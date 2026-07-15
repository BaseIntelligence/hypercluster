"""Signed miner auth for marketplace write routes.

Headers required on mutating public routes:
  X-Hotkey, X-Signature, X-Nonce, X-Timestamp

Canonical message (HMAC-dev and substrate-compatible body binding)::

    hypercluster:{hotkey}:{nonce}:{timestamp}:{sha256(body)}

When ``HYPER_ALLOW_INSECURE_SIGNATURES=true`` (or Settings allow flag),
signatures may be HMAC-SHA256 of the canonical message using the challenge
shared token (test/dev). Production path verifies substrate hotkey signatures
when bittensor is available; otherwise reject unless insecure mode is on.
"""

from __future__ import annotations

import hmac
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import RequestNonce, utc_now

logger = logging.getLogger(__name__)

HOTKEY_HEADER = "X-Hotkey"
SIGNATURE_HEADER = "X-Signature"
NONCE_HEADER = "X-Nonce"
TIMESTAMP_HEADER = "X-Timestamp"

DEFAULT_SIGNATURE_TTL_SECONDS = 300
DEFAULT_NONCE_TTL_SECONDS = 86_400


@dataclass(frozen=True)
class MinerIdentity:
    """Authenticated miner identity bound to a request body hash."""

    hotkey: str
    nonce: str
    timestamp: int
    body_hash: str


def canonical_message(*, hotkey: str, nonce: str, timestamp: str, body: bytes) -> bytes:
    """Build canonical bytes for miner signature verification."""

    body_hash = sha256(body).hexdigest()
    return f"hypercluster:{hotkey}:{nonce}:{timestamp}:{body_hash}".encode()


def body_hash_hex(body: bytes) -> str:
    return sha256(body).hexdigest()


def sign_dev(secret: str, message: bytes) -> str:
    """HMAC-SHA256 hex digest used in tests / insecure mode."""

    return hmac.new(secret.encode(), message, sha256).hexdigest()


def verify_dev_signature(secret: str, message: bytes, signature: str) -> bool:
    expected = sign_dev(secret, message)
    return hmac.compare_digest(expected, signature.removeprefix("sha256="))


def verify_substrate_signature(hotkey: str, message: bytes, signature: str) -> bool:
    try:
        import bittensor as bt  # type: ignore

        keypair = bt.Keypair(ss58_address=hotkey)
        return bool(keypair.verify(message, _decode_signature(signature)))
    except Exception:
        return False


def _decode_signature(signature: str) -> bytes | str:
    value = signature.removeprefix("0x")
    try:
        return bytes.fromhex(value)
    except ValueError:
        return signature


def build_signed_headers(
    *,
    secret: str,
    hotkey: str,
    body: bytes,
    nonce: str | None = None,
    timestamp: int | None = None,
) -> dict[str, str]:
    """Helper for tests and CLI: produce valid insecure-mode auth headers."""

    import uuid

    ts = str(timestamp if timestamp is not None else int(time.time()))
    n = nonce if nonce is not None else uuid.uuid4().hex
    message = canonical_message(hotkey=hotkey, nonce=n, timestamp=ts, body=body)
    return {
        HOTKEY_HEADER: hotkey,
        SIGNATURE_HEADER: sign_dev(secret, message),
        NONCE_HEADER: n,
        TIMESTAMP_HEADER: ts,
    }


async def get_db_session(request: Request) -> Any:
    """Yield a request-scoped async session from app.state.database."""

    database = getattr(request.app.state, "database", None)
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "database_unavailable", "message": "database not configured"},
        )
    async with database.session() as session:
        yield session


async def authenticate_miner(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    x_hotkey: Annotated[str | None, Header(alias=HOTKEY_HEADER)] = None,
    x_signature: Annotated[str | None, Header(alias=SIGNATURE_HEADER)] = None,
    x_nonce: Annotated[str | None, Header(alias=NONCE_HEADER)] = None,
    x_timestamp: Annotated[str | None, Header(alias=TIMESTAMP_HEADER)] = None,
) -> MinerIdentity:
    """Fail-closed signed auth for marketplace write routes."""

    if not x_hotkey or not x_signature or not x_nonce or not x_timestamp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "missing_auth_headers", "message": "signed auth headers required"},
        )

    settings = getattr(request.app.state, "settings", None)
    hyper = getattr(request.app.state, "hyper_settings", None)
    ttl = DEFAULT_SIGNATURE_TTL_SECONDS
    if hyper is not None and getattr(hyper, "signature_ttl_seconds", None) is not None:
        ttl = int(hyper.signature_ttl_seconds)

    try:
        timestamp = int(x_timestamp)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_timestamp", "message": "timestamp must be unix seconds"},
        ) from exc

    now = int(time.time())
    if abs(now - timestamp) > ttl:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "stale_signature",
                "message": "signature timestamp outside skew window",
            },
        )

    body = await request.body()
    message = canonical_message(
        hotkey=x_hotkey,
        nonce=x_nonce,
        timestamp=x_timestamp,
        body=body,
    )
    body_hash = body_hash_hex(body)

    allow_insecure = False
    if hyper is not None:
        allow_insecure = bool(getattr(hyper, "allow_insecure_signatures", False))

    shared_token = ""
    if settings is not None:
        shared_token = getattr(settings, "shared_token", None) or ""
        if not shared_token:
            # Token may only live in file path; for HMAC we need the resolved value.
            token_file = getattr(settings, "shared_token_file", None)
            if token_file:
                try:
                    from pathlib import Path

                    shared_token = Path(token_file).read_text(encoding="utf-8").strip()
                except OSError:
                    shared_token = ""

    valid = verify_substrate_signature(x_hotkey, message, x_signature)
    if not valid and allow_insecure and shared_token:
        valid = verify_dev_signature(shared_token, message, x_signature)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_signature", "message": "signature verification failed"},
        )

    # Nonce replay protection
    expires_at = utc_now() + timedelta(seconds=DEFAULT_NONCE_TTL_SECONDS)
    try:
        # Prune expired nonces opportunistically
        await session.execute(
            delete(RequestNonce).where(RequestNonce.expires_at < datetime.now(UTC))
        )
        session.add(
            RequestNonce(
                nonce=x_nonce,
                hotkey=x_hotkey,
                purpose="request",
                body_hash=body_hash,
                used_at=utc_now(),
                expires_at=expires_at,
            )
        )
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "nonce_replay", "message": "nonce already used"},
        ) from exc

    return MinerIdentity(
        hotkey=x_hotkey,
        nonce=x_nonce,
        timestamp=timestamp,
        body_hash=body_hash,
    )


# Type alias for FastAPI Depends
RequireMiner = Annotated[MinerIdentity, Depends(authenticate_miner)]

# Optional session type for handlers
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


async def require_hotkey_header_only(
    x_hotkey: Annotated[str | None, Header(alias=HOTKEY_HEADER)] = None,
) -> str:
    """For list endpoints that scope by owner without full signature (policy)."""

    if not x_hotkey:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "missing_hotkey", "message": "X-Hotkey required for scoped list"},
        )
    return x_hotkey


async def optional_hotkey(
    x_hotkey: Annotated[str | None, Header(alias=HOTKEY_HEADER)] = None,
) -> str | None:
    return x_hotkey


__all__ = [
    "DbSession",
    "HOTKEY_HEADER",
    "MinerIdentity",
    "NONCE_HEADER",
    "RequireMiner",
    "SIGNATURE_HEADER",
    "TIMESTAMP_HEADER",
    "authenticate_miner",
    "build_signed_headers",
    "canonical_message",
    "get_db_session",
    "optional_hotkey",
    "require_hotkey_header_only",
    "sign_dev",
    "verify_dev_signature",
    "verify_substrate_signature",
]
