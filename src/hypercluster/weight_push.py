"""Raw-weight push: monochronic snapshots, digest, mock-master ack.

Fulfills VAL-SCORE-013, 014, 015, 017, 023, 024, 030 and VAL-WGT-015.

Challenge builds ``RawWeightPushRequest`` (protocol 1.x), stores
``weight_snapshots`` with UNIQUE(epoch, revision), POSTs to Base master
``POST /internal/v1/challenges/{slug}/raw-weights`` (mock-master :3201 in
mission). Posted ``weights`` are the **sum-normalized** incentive map
(sum ≈ 1.0 when non-empty); ``raw_mass_json`` retains pre-normalize mass.
Never calls on-chain ``set_weights`` and never product-Verda egress. Push
loop is cooperative async and MUST NOT block ``/health``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from base.challenge_sdk.schemas import (
    RawWeightPushAcknowledgement,
    RawWeightPushRequest,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import WeightSnapshot, isoformat_utc, utc_now
from hypercluster.domain.aggregation import (
    compute_mass_map,
    compute_raw_weights,
    sanitize_weights_map,
)
from hypercluster.domain.incentive import (
    finalize_incentives,
    finalize_incentives_with_settings,
)
from hypercluster.no_verda import (
    VerdaForbiddenError,
    assert_challenge_outbound_allowed,
)
from hypercluster.settings import HyperSettings, get_hyper_settings

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "1.0"
DEFAULT_FRESHNESS_SECONDS = 300
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_EPOCH_SECONDS = 3600

# Base SDK hotkey key grammar (ss58-like); and sim-scenario first-char-digit variants.
# Mock-master relaxes to also accept sim keys; challenge push validates ss58-like when
# calling Base schema. For local sim we use Base-compatible keys for real pushes.
_SS58_KEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{3,64}$")

# Forbidden product surface: never import or call wallet set_weights.
_FORBIDDEN_SET_WEIGHTS = "set_weights"


class WeightPushValidationError(ValueError):
    """Illegal push window or payload (inverted/expired expires_at, bad digest)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class PushAttemptResult:
    """Outcome of one push attempt (ack may or may not land)."""

    status: str
    epoch: int
    revision: int
    payload_digest: str
    snapshot_id: str | None
    local_id: str | None = None
    push_status: str | None = None
    error: str | None = None
    cursor_advanced: bool = False
    idempotent: bool = False


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def epoch_from_now(
    now: datetime | None = None,
    *,
    epoch_seconds: int = DEFAULT_EPOCH_SECONDS,
) -> int:
    """Monotonic hour-bucket (or configured) epoch from wall/sim clock."""

    when = as_utc(now or utc_now())
    seconds = max(1, int(epoch_seconds))
    return int(when.timestamp()) // seconds


def is_ss58_like_hotkey(key: str) -> bool:
    """True when key looks like Base ss58 hotkey, not a bare UID integer."""

    if not key or not isinstance(key, str):
        return False
    if not _SS58_KEY_RE.fullmatch(key):
        return False
    if not any(ch.isalpha() for ch in key):
        return False
    return True


def filter_ss58_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Drop non-ss58 keys (VAL-SCORE-024: raw map is hotkey-keyed, never UIDs)."""

    clean = sanitize_weights_map(weights)
    return {k: v for k, v in clean.items() if is_ss58_like_hotkey(k)}


def canonical_challenge_push_request(
    *,
    method: str,
    path: str,
    challenge_slug: str,
    timestamp: str,
    body: bytes,
) -> str:
    """Mirror master's challenge-push binding (method/path/slug/ts/body)."""

    body_digest = hashlib.sha256(body).hexdigest()
    return f"{method.upper()}\n{path}\n{challenge_slug}\n{timestamp}\n{body_digest}"


def sign_challenge_push_request(*, token: str, canonical: str) -> str:
    return hmac.new(
        token.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def compute_payload_digest_for_body(body: dict[str, Any]) -> str:
    """Independent SHA-256 of canonical push payload (VAL-SCORE-014)."""

    return RawWeightPushRequest.compute_digest(body)


def validate_freshness_window(
    *,
    computed_at: datetime,
    expires_at: datetime,
    now: datetime | None = None,
    allow_expired_vs_wall: bool = False,
) -> None:
    """Reject inverted or already-expired expires_at (VAL-SCORE-030).

    Structural: expires_at must be strictly after computed_at (tz-aware).
    Receipt: expires_at must not be already in the past vs challenge clock at
    push construction time (unless ``allow_expired_vs_wall`` for transport
    retries of a durable pending that was valid at construction).
    """

    c = as_utc(computed_at)
    e = as_utc(expires_at)
    if e <= c:
        raise WeightPushValidationError(
            "inverted_window",
            "expires_at must be strictly after computed_at",
        )
    if not allow_expired_vs_wall:
        wall = as_utc(now or utc_now())
        if e <= wall:
            raise WeightPushValidationError(
                "expired_window",
                "expires_at is already in the past relative to challenge clock",
            )


def build_raw_weight_push_body(
    *,
    challenge_slug: str,
    epoch: int,
    revision: int,
    weights: Mapping[str, float],
    nonce: str,
    computed_at: datetime,
    expires_at: datetime,
    protocol_version: str = PROTOCOL_VERSION,
    already_normalized: bool = False,
) -> tuple[RawWeightPushRequest, bytes]:
    """Build digest-bound Base RawWeightPushRequest bytes.

    Egress maps are sum-normalized incentives (VAL-WGT-015): when mass > 0 the
    posted ``weights`` sum ≈ 1.0; empty remains burn-safe (no push). Callers
    that already finalized a unit-sum map may pass ``already_normalized=True``
    to skip a second normalize (idempotent: unit maps stay unit).

    Raises WeightPushValidationError on inverted/expired window or empty map.
    """

    validate_freshness_window(computed_at=computed_at, expires_at=expires_at, now=computed_at)
    cleaned = filter_ss58_weights(weights)
    if not cleaned:
        raise WeightPushValidationError(
            "empty_weights",
            "empty weight map has no push surface",
        )
    # VAL-WGT-015: mock-master / master raw-weights body carries unit-sum
    # incentives, never absolute composite mass alone.
    if already_normalized:
        emission = cleaned
    else:
        emission = filter_ss58_weights(finalize_incentives(cleaned, sum_normalize=True))
    if not emission:
        raise WeightPushValidationError(
            "empty_weights",
            "empty weight map has no push surface",
        )
    # Ensure finite ≥0 (already sanitized) and Base schema alphabet keys.
    body: dict[str, Any] = {
        "protocol_version": protocol_version,
        "challenge_slug": challenge_slug,
        "epoch": int(epoch),
        "revision": int(revision),
        "computed_at": as_utc(computed_at).replace(microsecond=0),
        "expires_at": as_utc(expires_at).replace(microsecond=0),
        "nonce": str(nonce),
        "weights": {str(k): float(v) for k, v in emission.items()},
    }
    # Digests exclude payload_digest field.
    digest_src = {
        "protocol_version": protocol_version,
        "challenge_slug": challenge_slug,
        "epoch": int(epoch),
        "revision": int(revision),
        "computed_at": isoformat_utc(as_utc(computed_at).replace(microsecond=0)),
        "expires_at": isoformat_utc(as_utc(expires_at).replace(microsecond=0)),
        "nonce": str(nonce),
        "weights": {str(k): float(v) for k, v in emission.items()},
    }
    digest = RawWeightPushRequest.compute_digest(digest_src)
    body["payload_digest"] = digest
    # Re-encode dates as ISO strings for pydantic validate.
    body_jsonlike = {
        **body,
        "computed_at": digest_src["computed_at"],
        "expires_at": digest_src["expires_at"],
    }
    try:
        payload = RawWeightPushRequest.model_validate(body_jsonlike)
    except Exception as exc:  # noqa: BLE001
        raise WeightPushValidationError("schema_error", str(exc)) from exc
    return payload, payload.canonical_bytes()


async def next_revision(
    session: AsyncSession,
    *,
    epoch: int,
) -> int:
    """Bump revision within epoch; start at 1 for a new epoch (monochronic)."""

    result = await session.execute(
        select(WeightSnapshot)
        .where(WeightSnapshot.epoch == int(epoch))
        .order_by(WeightSnapshot.revision.desc())
        .limit(1)
    )
    latest = result.scalar_one_or_none()
    if latest is None:
        return 1
    return int(latest.revision) + 1


async def get_latest_snapshot(
    session: AsyncSession,
    *,
    prefer_acked: bool = True,
) -> WeightSnapshot | None:
    """Latest snapshot (acked preferred when available)."""

    if prefer_acked:
        result = await session.execute(
            select(WeightSnapshot)
            .where(WeightSnapshot.push_status.in_(("acked", "sim")))
            .order_by(
                WeightSnapshot.epoch.desc(),
                WeightSnapshot.revision.desc(),
            )
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            return row
    result = await session.execute(
        select(WeightSnapshot)
        .order_by(WeightSnapshot.epoch.desc(), WeightSnapshot.revision.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_snapshot_by_epoch_revision(
    session: AsyncSession,
    *,
    epoch: int,
    revision: int,
) -> WeightSnapshot | None:
    result = await session.execute(
        select(WeightSnapshot).where(
            WeightSnapshot.epoch == int(epoch),
            WeightSnapshot.revision == int(revision),
        )
    )
    return result.scalar_one_or_none()


async def list_snapshots(
    session: AsyncSession,
    *,
    limit: int = 50,
) -> list[WeightSnapshot]:
    result = await session.execute(
        select(WeightSnapshot)
        .order_by(WeightSnapshot.epoch.desc(), WeightSnapshot.revision.desc())
        .limit(max(1, int(limit)))
    )
    return list(result.scalars().all())


def _snapshot_from_payload(
    *,
    local_id: str,
    payload: RawWeightPushRequest,
    push_status: str,
    canonical: bytes,
    raw_mass: Mapping[str, float] | None = None,
) -> WeightSnapshot:
    raw_mass_payload = (
        {str(k): float(v) for k, v in sanitize_weights_map(raw_mass).items()}
        if raw_mass is not None
        else {}
    )
    return WeightSnapshot(
        id=local_id,
        epoch=int(payload.epoch),
        revision=int(payload.revision),
        computed_at=as_utc(payload.computed_at),
        expires_at=as_utc(payload.expires_at),
        nonce=str(payload.nonce),
        payload_digest=str(payload.payload_digest),
        weights_json=json.dumps(
            {str(k): float(v) for k, v in payload.weights.items()},
            sort_keys=True,
        ),
        # VAL-WGT-013: retain pre-normalize absolute mass for audit.
        raw_mass_json=json.dumps(raw_mass_payload, sort_keys=True),
        push_status=push_status,
        canonical_payload=canonical.decode("utf-8"),
        master_ack_json=None,
        master_snapshot_id=None,
    )


async def create_pending_snapshot(
    session: AsyncSession,
    *,
    challenge_slug: str,
    epoch: int | None = None,
    revision: int | None = None,
    weights: Mapping[str, float] | None = None,
    hyper: HyperSettings | None = None,
    now: datetime | None = None,
    force_computed_at: datetime | None = None,
    force_expires_at: datetime | None = None,
    nonce: str | None = None,
) -> WeightSnapshot:
    """Persist a pending snapshot after window validation (VAL-SCORE-013/030).

    Does **not** call master. Caller may pass forced timestamps to exercise
    inverted/expired rejection paths — those raise without write.
    """

    settings = hyper if hyper is not None else get_hyper_settings()
    wall = as_utc(now or utc_now())
    computed = as_utc(force_computed_at) if force_computed_at is not None else wall
    freshness = int(getattr(settings, "weight_push_freshness_s", DEFAULT_FRESHNESS_SECONDS))
    expires = (
        as_utc(force_expires_at)
        if force_expires_at is not None
        else computed + timedelta(seconds=max(30, freshness))
    )
    # Validate before any DB write — illegal windows leave no acked/pending row.
    validate_freshness_window(computed_at=computed, expires_at=expires, now=wall)

    # Resolve absolute mass + unit-sum incentive map (VAL-WGT-011/013).
    # Caller-supplied ``weights`` is treated as mass input and re-normalized so
    # push / snapshot always share the same emission family as get_weights.
    if weights is not None:
        raw_mass_map = sanitize_weights_map(weights)
        weight_map = filter_ss58_weights(
            finalize_incentives_with_settings(raw_mass_map, hyper=settings)
        )
        # Retain ss58-filtered raw mass for snapshot audit (pre-normalize).
        raw_mass_retained = filter_ss58_weights(raw_mass_map)
    else:
        raw_mass_map = await compute_mass_map(session, hyper=settings)
        raw_mass_retained = filter_ss58_weights(raw_mass_map)
        weight_map = filter_ss58_weights(await compute_raw_weights(session, hyper=settings))
    if not weight_map:
        raise WeightPushValidationError(
            "empty_weights",
            "empty weight map has no push surface",
        )

    resolved_epoch = (
        int(epoch)
        if epoch is not None
        else epoch_from_now(
            wall, epoch_seconds=int(getattr(settings, "epoch_seconds", DEFAULT_EPOCH_SECONDS))
        )
    )
    if revision is not None:
        resolved_revision = int(revision)
    else:
        resolved_revision = await next_revision(session, epoch=resolved_epoch)
    # Idempotent reuse: if exact epoch/revision already exists, return it
    # rather than rewriting illegally (VAL-SCORE-013 monochronic unique).
    existing = await get_snapshot_by_epoch_revision(
        session, epoch=resolved_epoch, revision=resolved_revision
    )
    if existing is not None:
        return existing

    local_id = str(uuid.uuid4())
    n = nonce or f"hyper-{uuid.uuid4().hex}"
    # weight_map is already unit-sum from finalize_incentives_with_settings
    # (or live compute_raw_weights); flag already_normalized to keep digest
    # stable and avoid a redundant second pass (still idempotent if reused).
    payload, raw = build_raw_weight_push_body(
        challenge_slug=challenge_slug,
        epoch=resolved_epoch,
        revision=resolved_revision,
        weights=weight_map,
        nonce=n,
        computed_at=computed,
        expires_at=expires,
        already_normalized=True,
    )
    row = _snapshot_from_payload(
        local_id=local_id,
        payload=payload,
        push_status="pending",
        canonical=raw,
        raw_mass=raw_mass_retained,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def mark_snapshot_status(
    session: AsyncSession,
    snapshot: WeightSnapshot,
    *,
    push_status: str,
    master_ack: dict[str, Any] | None = None,
    master_snapshot_id: str | None = None,
) -> WeightSnapshot:
    snapshot.push_status = push_status
    if master_ack is not None:
        snapshot.master_ack_json = json.dumps(master_ack, sort_keys=True)
    if master_snapshot_id is not None:
        snapshot.master_snapshot_id = master_snapshot_id
    snapshot.updated_at = utc_now()
    await session.commit()
    await session.refresh(snapshot)
    return snapshot


def resolve_master_base_url(hyper: HyperSettings | None = None) -> str | None:
    """Return configured master base URL after Verda outbound allowlist check.

    VAL-LIVE-011: challenge weight push must never target api.verda.com even if
    ``HYPER_MASTER_BASE_URL`` is mis-set to a commercial cloud control plane.
    """

    product = hyper if hyper is not None else get_hyper_settings()
    master = getattr(product, "master_base_url", None)
    if not master:
        return None
    url = str(master).strip()
    if not url:
        return None
    try:
        assert_challenge_outbound_allowed(url)
    except VerdaForbiddenError as exc:
        raise WeightPushValidationError(exc.code, exc.message) from exc
    return url.rstrip("/")


class WeightPushClient:
    """Build, sign, and POST raw weights; persist wait/ack on weight_snapshots."""

    def __init__(
        self,
        *,
        database: Any,
        challenge_slug: str = "hypercluster",
        master_base_url: str,
        shared_token: str,
        hyper: HyperSettings | None = None,
        freshness_seconds: int | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        now_fn: Callable[[], datetime] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        # Fail closed on Verda (or other forbidden) control-plane URLs.
        assert_challenge_outbound_allowed(master_base_url)
        self.database = database
        self.challenge_slug = challenge_slug
        self.master_base_url = master_base_url.rstrip("/")
        self.shared_token = shared_token
        self.hyper = hyper if hyper is not None else get_hyper_settings()
        self.freshness_seconds = int(
            freshness_seconds
            if freshness_seconds is not None
            else getattr(self.hyper, "weight_push_freshness_s", DEFAULT_FRESHNESS_SECONDS)
        )
        self.timeout_seconds = float(timeout_seconds)
        self._now_fn = now_fn or (lambda: datetime.now(UTC))
        self._http = http_client

    def _path(self) -> str:
        return f"/internal/v1/challenges/{self.challenge_slug}/raw-weights"

    def _headers(self, *, path: str, body: bytes, timestamp: int) -> dict[str, str]:
        canonical = canonical_challenge_push_request(
            method="POST",
            path=path,
            challenge_slug=self.challenge_slug,
            timestamp=str(timestamp),
            body=body,
        )
        signature = sign_challenge_push_request(token=self.shared_token, canonical=canonical)
        return {
            "Authorization": f"Bearer {self.shared_token}",
            "Content-Type": "application/json",
            "X-Base-Challenge-Slug": self.challenge_slug,
            "X-Signature": signature,
            "X-Timestamp": str(timestamp),
            "Accept": "application/json",
        }

    @staticmethod
    def _ack_matches(ack: RawWeightPushAcknowledgement, *, payload: RawWeightPushRequest) -> bool:
        return (
            ack.accepted is True
            and ack.challenge_slug == payload.challenge_slug
            and ack.epoch == payload.epoch
            and ack.revision == payload.revision
            and ack.payload_digest == payload.payload_digest
            and bool(ack.snapshot_id)
        )

    async def push_once(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        epoch: int | None = None,
        revision: int | None = None,
        force_computed_at: datetime | None = None,
        force_expires_at: datetime | None = None,
        reuse_snapshot_id: str | None = None,
    ) -> PushAttemptResult:
        """Push one monochronic snapshot. Illegal windows raise/return rejected.

        VAL-SCORE-030: inverted/expired expires_at never becomes push_status=acked.
        VAL-SCORE-015: happy path stores push_status=acked; re-push is idempotent.
        """

        # Guard: product must never reference chain set_weights.
        assert _FORBIDDEN_SET_WEIGHTS not in dir(self)

        now = as_utc(self._now_fn())
        snapshot: WeightSnapshot | None = None
        payload: RawWeightPushRequest | None = None
        raw_bytes: bytes | None = None

        try:
            async with self.database.session() as session:
                if reuse_snapshot_id is not None:
                    result = await session.execute(
                        select(WeightSnapshot).where(WeightSnapshot.id == reuse_snapshot_id)
                    )
                    snapshot = result.scalar_one_or_none()
                    if snapshot is None:
                        return PushAttemptResult(
                            status="missing_snapshot",
                            epoch=0,
                            revision=0,
                            payload_digest="",
                            snapshot_id=None,
                            error=f"snapshot {reuse_snapshot_id} not found",
                        )
                    # Re-validate stored window (expired prior good snapshot —
                    # do not overwrite acked with illegal re-send).
                    try:
                        validate_freshness_window(
                            computed_at=snapshot.computed_at,
                            expires_at=snapshot.expires_at,
                            now=now,
                            allow_expired_vs_wall=snapshot.push_status
                            in {"pending", "failed", "rejected"},
                        )
                    except WeightPushValidationError as exc:
                        if snapshot.push_status not in {"acked", "sim"}:
                            await mark_snapshot_status(
                                session,
                                snapshot,
                                push_status="invalid_window",
                            )
                        return PushAttemptResult(
                            status="invalid_window",
                            epoch=int(snapshot.epoch),
                            revision=int(snapshot.revision),
                            payload_digest=snapshot.payload_digest,
                            snapshot_id=None,
                            local_id=snapshot.id,
                            push_status="invalid_window",
                            error=exc.message,
                        )
                    if snapshot.canonical_payload:
                        try:
                            payload = RawWeightPushRequest.model_validate_json(
                                snapshot.canonical_payload
                            )
                            raw_bytes = payload.canonical_bytes()
                        except Exception:  # noqa: BLE001
                            payload = None
                            raw_bytes = None

                if payload is None or raw_bytes is None:
                    # Build + store pending (validates window fail-closed).
                    try:
                        snapshot = await create_pending_snapshot(
                            session,
                            challenge_slug=self.challenge_slug,
                            epoch=epoch,
                            revision=revision,
                            weights=weights,
                            hyper=self.hyper,
                            now=now,
                            force_computed_at=force_computed_at,
                            force_expires_at=force_expires_at,
                        )
                    except WeightPushValidationError as exc:
                        return PushAttemptResult(
                            status="invalid_window"
                            if exc.code in {"inverted_window", "expired_window"}
                            else exc.code,
                            epoch=int(epoch or 0),
                            revision=int(revision or 0),
                            payload_digest="",
                            snapshot_id=None,
                            push_status="invalid_window"
                            if exc.code in {"inverted_window", "expired_window"}
                            else None,
                            error=exc.message,
                        )
                    payload = RawWeightPushRequest.model_validate_json(
                        snapshot.canonical_payload or "{}"
                    )
                    raw_bytes = payload.canonical_bytes()

                # Already acked same identity → idempotent replay ok (VAL-SCORE-015).
                if snapshot is not None and snapshot.push_status in {"acked", "sim"}:
                    if snapshot.canonical_payload:
                        return PushAttemptResult(
                            status="acknowledged",
                            epoch=int(snapshot.epoch),
                            revision=int(snapshot.revision),
                            payload_digest=snapshot.payload_digest,
                            snapshot_id=snapshot.master_snapshot_id,
                            local_id=snapshot.id,
                            push_status=snapshot.push_status,
                            cursor_advanced=False,
                            idempotent=True,
                        )
        except WeightPushValidationError as exc:
            return PushAttemptResult(
                status=exc.code,
                epoch=int(epoch or 0),
                revision=int(revision or 0),
                payload_digest="",
                snapshot_id=None,
                error=exc.message,
            )

        assert snapshot is not None and payload is not None and raw_bytes is not None

        path = self._path()
        url = f"{self.master_base_url}{path}"
        headers = self._headers(path=path, body=raw_bytes, timestamp=int(now.timestamp()))
        client = self._http
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout_seconds)
        assert client is not None
        try:
            response = await client.post(url, content=raw_bytes, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            async with self.database.session() as session:
                row = await get_snapshot_by_epoch_revision(
                    session, epoch=payload.epoch, revision=payload.revision
                )
                if row is not None and row.push_status == "pending":
                    await mark_snapshot_status(session, row, push_status="failed")
            return PushAttemptResult(
                status="transport_error",
                epoch=payload.epoch,
                revision=payload.revision,
                payload_digest=payload.payload_digest,
                snapshot_id=None,
                local_id=snapshot.id,
                push_status="failed",
                error=str(exc),
            )
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code not in {200, 201}:
            status_name = "rejected" if response.status_code < 500 else "server_error"
            async with self.database.session() as session:
                row = await get_snapshot_by_epoch_revision(
                    session, epoch=payload.epoch, revision=payload.revision
                )
                if row is not None and row.push_status == "pending":
                    await mark_snapshot_status(session, row, push_status="rejected")
            return PushAttemptResult(
                status=status_name,
                epoch=payload.epoch,
                revision=payload.revision,
                payload_digest=payload.payload_digest,
                snapshot_id=None,
                local_id=snapshot.id,
                push_status="rejected",
                error=f"status={response.status_code}",
            )

        try:
            ack = RawWeightPushAcknowledgement.model_validate(response.json())
        except Exception as exc:  # noqa: BLE001
            return PushAttemptResult(
                status="malformed_ack",
                epoch=payload.epoch,
                revision=payload.revision,
                payload_digest=payload.payload_digest,
                snapshot_id=None,
                local_id=snapshot.id,
                push_status="failed",
                error=str(exc),
            )

        if not self._ack_matches(ack, payload=payload):
            return PushAttemptResult(
                status="ack_mismatch",
                epoch=payload.epoch,
                revision=payload.revision,
                payload_digest=payload.payload_digest,
                snapshot_id=getattr(ack, "snapshot_id", None),
                local_id=snapshot.id,
                push_status="failed",
                error="acknowledgement identity mismatch",
            )

        ack_body = {
            "protocol_version": ack.protocol_version,
            "challenge_slug": ack.challenge_slug,
            "epoch": ack.epoch,
            "revision": ack.revision,
            "snapshot_id": ack.snapshot_id,
            "payload_digest": ack.payload_digest,
            "accepted": True,
            "idempotent": bool(ack.idempotent),
        }
        async with self.database.session() as session:
            row = await get_snapshot_by_epoch_revision(
                session, epoch=payload.epoch, revision=payload.revision
            )
            if row is None:
                return PushAttemptResult(
                    status="missing_snapshot",
                    epoch=payload.epoch,
                    revision=payload.revision,
                    payload_digest=payload.payload_digest,
                    snapshot_id=ack.snapshot_id,
                    error="local snapshot missing after ack",
                )
            already = row.push_status in {"acked", "sim"}
            await mark_snapshot_status(
                session,
                row,
                push_status="acked",
                master_ack=ack_body,
                master_snapshot_id=ack.snapshot_id,
            )
        return PushAttemptResult(
            status="acknowledged",
            epoch=payload.epoch,
            revision=payload.revision,
            payload_digest=payload.payload_digest,
            snapshot_id=ack.snapshot_id,
            local_id=snapshot.id,
            push_status="acked",
            cursor_advanced=not already,
            idempotent=bool(ack.idempotent) or already,
        )


async def run_weight_push_loop(
    client: WeightPushClient,
    *,
    interval_seconds: float = 120.0,
    resilient: bool = True,
) -> None:
    """Background push worker. Cooperative asyncio; never blocks /health."""

    logger.info(
        "weight push loop started master=%s slug=%s",
        client.master_base_url,
        client.challenge_slug,
    )
    while True:
        try:
            result = await client.push_once()
            logger.info(
                "weight push attempt status=%s epoch=%s revision=%s",
                result.status,
                result.epoch,
                result.revision,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            if not resilient:
                raise
            logger.exception("weight push loop iteration failed")
        await asyncio.sleep(max(float(interval_seconds), 0.1))


def resolve_shared_token(settings: Any) -> str | None:
    """Resolve challenge shared token from settings (env or file)."""

    direct = getattr(settings, "shared_token", None)
    if direct:
        return str(direct)
    path = getattr(settings, "shared_token_file", None)
    if path:
        try:
            text = open(path, encoding="utf-8").read().strip()  # noqa: SIM115
        except OSError:
            return None
        return text or None
    return None


def maybe_build_push_client(
    *,
    database: Any,
    settings: Any,
    hyper: HyperSettings | None = None,
) -> WeightPushClient | None:
    """Construct push client when master URL + token enable raw-weight push."""

    product = hyper if hyper is not None else get_hyper_settings()
    if not bool(getattr(product, "weight_push_enabled", True)):
        return None
    try:
        master = resolve_master_base_url(product)
    except WeightPushValidationError:
        # Misconfiguration toward Verda (or other forbidden hosts) must not
        # partially construct a client that could dial those endpoints.
        logger.error("weight push disabled: master base URL fails outbound allowlist")
        return None
    if not master:
        return None
    token = resolve_shared_token(settings)
    if not token:
        return None
    slug = str(getattr(settings, "slug", "hypercluster") or "hypercluster")
    return WeightPushClient(
        database=database,
        challenge_slug=slug,
        master_base_url=str(master),
        shared_token=token,
        hyper=product,
        freshness_seconds=int(
            getattr(product, "weight_push_freshness_s", DEFAULT_FRESHNESS_SECONDS)
        ),
        timeout_seconds=float(getattr(product, "weight_push_timeout_s", DEFAULT_TIMEOUT_SECONDS)),
    )


__all__ = [
    "DEFAULT_EPOCH_SECONDS",
    "DEFAULT_FRESHNESS_SECONDS",
    "PROTOCOL_VERSION",
    "PushAttemptResult",
    "WeightPushClient",
    "WeightPushValidationError",
    "as_utc",
    "build_raw_weight_push_body",
    "canonical_challenge_push_request",
    "compute_payload_digest_for_body",
    "create_pending_snapshot",
    "epoch_from_now",
    "filter_ss58_weights",
    "get_latest_snapshot",
    "get_snapshot_by_epoch_revision",
    "is_ss58_like_hotkey",
    "list_snapshots",
    "mark_snapshot_status",
    "maybe_build_push_client",
    "next_revision",
    "resolve_master_base_url",
    "resolve_shared_token",
    "run_weight_push_loop",
    "sign_challenge_push_request",
    "validate_freshness_window",
]
