"""Mock Base master for local raw-weight push integration (port 3201).

Accepts ``POST /internal/v1/challenges/{slug}/raw-weights`` using the Base
``RawWeightPushRequest`` schema (digest + freshness structural checks) and
acks with ``RawWeightPushAcknowledgement``. Also serves ``GET /health``.

VAL-SCORE-015/025/030: stores snapshots by (epoch, revision) for idempotency;
rejects digest mismatches and inverted windows via pydantic validation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from base.challenge_sdk.schemas import (
    RawWeightPushAcknowledgement,
    RawWeightPushRequest,
)
from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import ValidationError

PROTOCOL_VERSION = "1.0"

app = FastAPI(title="hypercluster-mock-master", version="0.1.0")

# In-process durable ack store: (slug, epoch, revision) → snapshot record.
_SNAPSHOTS: dict[tuple[str, int, int], dict[str, Any]] = {}
# Shared token for optional signature check when CHALLENGE_SHARED_TOKEN set.
_EXPECTED_TOKEN: str | None = None


def reset_store() -> None:
    """Clear ack store (tests)."""

    _SNAPSHOTS.clear()


def configure_token(token: str | None) -> None:
    global _EXPECTED_TOKEN
    _EXPECTED_TOKEN = token


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        return ""
    return authorization.split(" ", 1)[1].strip()


def _canonical_sig(
    *,
    method: str,
    path: str,
    challenge_slug: str,
    timestamp: str,
    body: bytes,
) -> str:
    body_digest = hashlib.sha256(body).hexdigest()
    return f"{method.upper()}\n{path}\n{challenge_slug}\n{timestamp}\n{body_digest}"


def _hmac_hex(token: str, canonical: str) -> str:
    return hmac.new(
        token.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "mock-master", "role": "mock-master"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    return {"status": "ok", "ready": True, "service": "mock-master"}


@app.post("/internal/v1/challenges/{slug}/raw-weights")
async def raw_weights(
    slug: str,
    request: Request,
    authorization: str | None = Header(default=None),
    x_base_challenge_slug: str | None = Header(default=None, alias="X-Base-Challenge-Slug"),
    x_signature: str | None = Header(default=None, alias="X-Signature"),
    x_timestamp: str | None = Header(default=None, alias="X-Timestamp"),
) -> dict[str, Any]:
    """Accept authenticated raw-weight push and return durable ack."""

    raw = await request.body()
    if len(raw) > 256_000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="payload too large",
        )

    token = _bearer(authorization)
    if _EXPECTED_TOKEN:
        if not token or not hmac.compare_digest(token, _EXPECTED_TOKEN):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )
        if x_base_challenge_slug and x_base_challenge_slug != slug:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden",
            )
        if x_signature and x_timestamp:
            path = f"/internal/v1/challenges/{slug}/raw-weights"
            expected = _hmac_hex(
                _EXPECTED_TOKEN,
                _canonical_sig(
                    method="POST",
                    path=path,
                    challenge_slug=slug,
                    timestamp=str(x_timestamp),
                    body=raw,
                ),
            )
            if not hmac.compare_digest(expected, x_signature):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized",
                )

    try:
        payload = RawWeightPushRequest.model_validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_raw_weight_payload", "errors": json.loads(exc.json())},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_raw_weight_payload", "detail": str(exc)},
        ) from exc

    if payload.challenge_slug != slug:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="challenge_slug mismatch",
        )

    # Structural freshness already enforced by schema (expires_at > computed_at).
    # Receipt-time: reject clearly inverted (defensive).
    if _as_utc(payload.expires_at) <= _as_utc(payload.computed_at):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "inverted_window", "detail": "expires_at must be after computed_at"},
        )

    key = (slug, int(payload.epoch), int(payload.revision))
    existing = _SNAPSHOTS.get(key)
    if existing is not None:
        if existing["payload_digest"] != payload.payload_digest:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="conflicting raw weight payload",
            )
        # Idempotent replay of exact same identity.
        ack = RawWeightPushAcknowledgement(
            protocol_version=PROTOCOL_VERSION,
            challenge_slug=slug,
            epoch=payload.epoch,
            revision=payload.revision,
            snapshot_id=existing["snapshot_id"],
            payload_digest=payload.payload_digest,
            accepted=True,
            idempotent=True,
        )
        return json.loads(ack.model_dump_json())

    snapshot_id = f"mock-{uuid.uuid4().hex}"
    record = {
        "snapshot_id": snapshot_id,
        "payload_digest": payload.payload_digest,
        "epoch": int(payload.epoch),
        "revision": int(payload.revision),
        "challenge_slug": slug,
        "weights": dict(payload.weights),
        "nonce": payload.nonce,
        "computed_at": payload.computed_at.isoformat(),
        "expires_at": payload.expires_at.isoformat(),
        "received_at": datetime.now(UTC).isoformat(),
    }
    _SNAPSHOTS[key] = record
    ack = RawWeightPushAcknowledgement(
        protocol_version=PROTOCOL_VERSION,
        challenge_slug=slug,
        epoch=payload.epoch,
        revision=payload.revision,
        snapshot_id=snapshot_id,
        payload_digest=payload.payload_digest,
        accepted=True,
        idempotent=False,
    )
    return json.loads(ack.model_dump_json())


@app.get("/internal/v1/challenges/{slug}/raw-weights")
async def list_raw_weights(slug: str) -> dict[str, Any]:
    """Debug: list accepted snapshots for a slug."""

    items = [
        rec
        for (s, _e, _r), rec in sorted(_SNAPSHOTS.items(), key=lambda kv: (kv[0][1], kv[0][2]))
        if s == slug
    ]
    return {"slug": slug, "items": items, "count": len(items)}


@app.get("/debug/snapshots")
async def debug_snapshots() -> dict[str, Any]:
    items = [
        {"key": {"slug": s, "epoch": e, "revision": r}, **rec}
        for (s, e, r), rec in _SNAPSHOTS.items()
    ]
    return {"count": len(items), "items": items}


__all__ = ["app", "configure_token", "reset_store"]
