"""Challenge-local points ledger earn from scored attempts (M10 + M11).

Earn rule (VAL-WGT-002 / 003 / 004, library/points-incentive.md)::

    if composite > 0 and points_enabled:
        delta = composite * HYPER_POINTS_SCALE   # default scale 1.0
    else:
        no positive score_earn mint

Optional M11 competitiveness (VAL-PRICE-060..063, default OFF)::

    if HYPER_PRICE_WEIGHT_IN_EARN:
        price_weight = clamp(P_cat / P_list, floor, ceil)
                       or HYPER_PRICE_WEIGHT_MISSING when prices unknown
        delta = composite * scale * price_weight
    else:
        delta = composite * scale   # M10 parity

Idempotency: at most one ``score_earn`` ledger row per ``attempt_id``
(unique constraint + pre-check no-op). Balance rollup is updated in the same
session flush.

Downstream of the fixed four-factor product only — never a 5th scoring factor.
price_weight multiplies ledger mint only; composite identity is untouched.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import (
    Job,
    JobAttempt,
    Lease,
    Offer,
    PointsBalance,
    PointsLedger,
    Score,
    utc_now,
)
from hypercluster.settings import HyperSettings, get_hyper_settings

logger = logging.getLogger(__name__)

REASON_SCORE_EARN = "score_earn"
REASON_ADMIN_ADJUST = "admin_adjust"

# Tolerance: composites this close to zero never mint positive mass.
_COMPOSITE_EPS = 0.0

PRICE_WEIGHT_MODE_OFF = "off"
PRICE_WEIGHT_MODE_MISSING = "missing"
PRICE_WEIGHT_MODE_CATALOG_RATIO = "catalog_ratio"


def compute_score_earn_delta(
    composite: float,
    *,
    scale: float = 1.0,
    price_weight: float = 1.0,
) -> float:
    """Return the points delta for a score_earn, or 0.0 when no mint.

    Positive finite composite × non-negative finite scale × price_weight → delta.
    Non-positive / non-finite composite, scale, or weight → 0 for bad inputs.
    ``price_weight`` defaults to 1.0 (M10 parity / flag-off / missing prices).
    """

    try:
        c = float(composite)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(c) or c <= _COMPOSITE_EPS:
        return 0.0
    try:
        s = float(scale)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(s) or s <= 0.0:
        return 0.0
    try:
        w = float(price_weight)
    except (TypeError, ValueError):
        w = 1.0
    if not math.isfinite(w) or w <= 0.0:
        return 0.0
    delta = c * s * w
    if not math.isfinite(delta) or delta <= 0.0:
        return 0.0
    return float(delta)


def compute_price_weight(
    *,
    list_price: float | None,
    catalog_price: float | None,
    enabled: bool = False,
    floor: float = 0.85,
    ceil: float = 1.15,
    missing: float = 1.0,
) -> float:
    """Competitiveness weight for optional points earn (VAL-PRICE-060..062).

    * Flag off → 1.0 (M10 parity; no price term).
    * Missing / non-positive list or catalog → ``missing`` (default 1.0 neutral).
    * Else ``clamp(P_cat / P_list, floor, ceil)`` — bargain list < catalog
      yields weight in (1, ceil]; gouge floors at floor so honest work is not zeroed.
    """

    if not enabled:
        return 1.0

    # Coerce missing default safely.
    try:
        miss = float(missing)
    except (TypeError, ValueError):
        miss = 1.0
    if not math.isfinite(miss) or miss <= 0.0:
        miss = 1.0

    list_p = _positive_price_or_none(list_price)
    cat_p = _positive_price_or_none(catalog_price)
    if list_p is None or cat_p is None:
        return float(miss)

    # Guarantee floor <= ceil (swap if misconfigured); both must be positive finite.
    try:
        lo = float(floor)
        hi = float(ceil)
    except (TypeError, ValueError):
        return float(miss)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0.0 or hi <= 0.0:
        return float(miss)
    if lo > hi:
        lo, hi = hi, lo

    ratio = cat_p / list_p
    if not math.isfinite(ratio) or ratio <= 0.0:
        return float(miss)
    if ratio < lo:
        return float(lo)
    if ratio > hi:
        return float(hi)
    return float(ratio)


def _positive_price_or_none(value: Any) -> float | None:
    """Parse a single price field; missing / non-positive / non-finite → None."""

    if value is None:
        return None
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or p <= 0.0:
        return None
    return float(p)


def price_weight_mode(
    *,
    enabled: bool,
    list_price: float | None,
    catalog_price: float | None,
) -> str:
    """Mode tag persisted in ledger details_json for forensics."""

    if not enabled:
        return PRICE_WEIGHT_MODE_OFF
    list_missing = _positive_price_or_none(list_price) is None
    cat_missing = _positive_price_or_none(catalog_price) is None
    if list_missing or cat_missing:
        return PRICE_WEIGHT_MODE_MISSING
    return PRICE_WEIGHT_MODE_CATALOG_RATIO


def _safe_balance(value: Any) -> float:
    """Coerce balance-like value to finite float (else 0.0)."""

    try:
        bal = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(bal):
        return 0.0
    return bal


def ledger_row_to_public(row: PointsLedger) -> dict[str, Any]:
    """Public forensic shape for a ledger row (no secrets).

    Includes attempt_id / score_id when present for earn forensics (VAL-WGT-007).
    """

    public = row.to_dict()
    public["delta"] = float(row.delta) if math.isfinite(float(row.delta)) else 0.0
    public["balance_after"] = _safe_balance(row.balance_after)
    # Never leak raw env/token keys if a misconfigured details blob carried them.
    details = public.get("details")
    if isinstance(details, dict):
        scrubbed = {
            k: v
            for k, v in details.items()
            if str(k).lower()
            not in {
                "token",
                "shared_token",
                "password",
                "private_key",
                "api_key",
                "authorization",
                "secret",
            }
        }
        public["details"] = scrubbed
    return public


def balance_row_to_public(row: PointsBalance) -> dict[str, Any]:
    """Public balance rollup shape for list/balance endpoints."""

    body = row.to_dict()
    body["balance"] = _safe_balance(row.balance)
    return body


async def get_points_balance(session: AsyncSession, hotkey: str) -> float:
    """Current denormalized balance for hotkey (0.0 if never seen)."""

    result = await session.execute(
        select(PointsBalance).where(PointsBalance.hotkey == hotkey)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return 0.0
    return _safe_balance(row.balance)


async def get_balance_row(
    session: AsyncSession,
    hotkey: str,
) -> PointsBalance | None:
    """Return points_balances row for hotkey, if any."""

    result = await session.execute(
        select(PointsBalance).where(PointsBalance.hotkey == hotkey)
    )
    return result.scalar_one_or_none()


async def list_points_balances(
    session: AsyncSession,
    *,
    limit: int = 100,
) -> list[PointsBalance]:
    """Enumerate balance rollups ordered by balance desc (VAL-WGT-006).

    Empty DB → ``[]`` (never crash). Only rows that exist are returned;
    never-seen hotkeys are omitted from list (use balance endpoint for 0).
    """

    lim = max(1, min(int(limit), 1000))
    result = await session.execute(
        select(PointsBalance)
        .order_by(PointsBalance.balance.desc(), PointsBalance.hotkey.asc())
        .limit(lim)
    )
    return list(result.scalars().all())


async def list_points_history(
    session: AsyncSession,
    hotkey: str,
    *,
    limit: int = 100,
) -> list[PointsLedger]:
    """Ordered ledger history for a hotkey, newest first (VAL-WGT-007).

    Never-seen hotkey → empty list (empty-safe).
    """

    lim = max(1, min(int(limit), 1000))
    hk = str(hotkey or "").strip()
    if not hk:
        return []
    result = await session.execute(
        select(PointsLedger)
        .where(PointsLedger.hotkey == hk)
        .order_by(PointsLedger.created_at.desc())
        .limit(lim)
    )
    return list(result.scalars().all())


async def get_ledger_for_attempt(
    session: AsyncSession,
    attempt_id: str,
) -> PointsLedger | None:
    """Return existing ledger row for attempt_id, if any."""

    if not attempt_id:
        return None
    result = await session.execute(
        select(PointsLedger).where(PointsLedger.attempt_id == attempt_id)
    )
    return result.scalar_one_or_none()


async def _upsert_balance(
    session: AsyncSession,
    *,
    hotkey: str,
    delta: float,
) -> float:
    """Apply ``delta`` to denormalized balance; return balance_after."""

    result = await session.execute(
        select(PointsBalance).where(PointsBalance.hotkey == hotkey)
    )
    bal = result.scalar_one_or_none()
    now = utc_now()
    if bal is None:
        new_balance = float(delta)
        bal = PointsBalance(hotkey=hotkey, balance=new_balance, updated_at=now)
        session.add(bal)
    else:
        try:
            prev = float(bal.balance)
        except (TypeError, ValueError):
            prev = 0.0
        if not math.isfinite(prev):
            prev = 0.0
        new_balance = prev + float(delta)
        bal.balance = new_balance
        bal.updated_at = now
    await session.flush()
    return float(new_balance)


async def resolve_earn_price_snapshot(
    session: AsyncSession,
    attempt_id: str,
) -> dict[str, Any]:
    """Resolve list/catalog prices from attempt → job → lease → offer snaps.

    Preference (design §7.2):
    - ``P_list``: lease listed price else offer listed price.
    - ``P_cat``: offer catalog snap (``catalog_price_per_hour``) when present.
    Returns keys ``list_price_per_hour``, ``catalog_price_per_hour``,
    ``catalog_model_key`` (values may be None when unbound / unsnapped).
    """

    empty: dict[str, Any] = {
        "list_price_per_hour": None,
        "catalog_price_per_hour": None,
        "catalog_model_key": None,
    }
    aid = str(attempt_id or "").strip()
    if not aid:
        return empty

    attempt = (
        await session.execute(select(JobAttempt).where(JobAttempt.id == aid))
    ).scalar_one_or_none()
    if attempt is None:
        return empty

    job = (
        await session.execute(select(Job).where(Job.id == attempt.job_id))
    ).scalar_one_or_none()
    if job is None:
        return empty

    list_price: float | None = None
    catalog_price: float | None = None
    catalog_model_key: str | None = None
    offer: Offer | None = None

    lease_id = getattr(job, "lease_id", None)
    if lease_id:
        lease = (
            await session.execute(select(Lease).where(Lease.id == str(lease_id)))
        ).scalar_one_or_none()
        if lease is not None:
            list_price = _positive_price_or_none(lease.price_per_hour)
            offer = (
                await session.execute(select(Offer).where(Offer.id == lease.offer_id))
            ).scalar_one_or_none()

    if offer is not None:
        if list_price is None:
            list_price = _positive_price_or_none(offer.price_per_hour)
        catalog_price = _positive_price_or_none(getattr(offer, "catalog_price_per_hour", None))
        key = getattr(offer, "catalog_model_key", None)
        if key:
            catalog_model_key = str(key)

    return {
        "list_price_per_hour": list_price,
        "catalog_price_per_hour": catalog_price,
        "catalog_model_key": catalog_model_key,
    }


async def earn_from_score(
    session: AsyncSession,
    score: Score,
    *,
    hyper: HyperSettings | None = None,
    list_price_per_hour: float | None = None,
    catalog_price_per_hour: float | None = None,
    catalog_model_key: str | None = None,
) -> PointsLedger | None:
    """Mint ``score_earn`` points from a fully scored attempt (idempotent).

    VAL-WGT-002: composite > 0 → positive ledger delta & balance increase.
    VAL-WGT-003: composite ≤ 0 / integrity zero → no positive mint (None).
    VAL-WGT-004: same ``attempt_id`` replay is a no-op (returns existing row).
    VAL-PRICE-060: flag off → delta == composite * scale (no price term).
    VAL-PRICE-061: flag on + both prices > 0 → clamp(catalog/list, floor, ceil).
    VAL-PRICE-062: missing/≤0 prices → HYPER_PRICE_WEIGHT_MISSING (default 1.0).
    VAL-PRICE-063: price_weight multiplies ledger only; composite stays pure.

    Optional explicit price kwargs override attempt→lease→offer resolve (tests /
    callers that already hold snaps). When flag is off, prices are ignored for
    math (delta parity with M10).

    Returns the ledger row created or the existing earn row; ``None`` when no
    positive mint and no prior earn for the attempt.
    """

    settings = hyper if hyper is not None else get_hyper_settings()
    attempt_id = str(score.attempt_id or "").strip()
    hotkey = str(score.hotkey or "").strip()
    if not attempt_id or not hotkey:
        return None

    # Idempotent short-circuit before any balance mutation.
    existing = await get_ledger_for_attempt(session, attempt_id)
    if existing is not None:
        return existing

    if not bool(getattr(settings, "points_enabled", True)):
        return None

    scale = float(getattr(settings, "points_scale", 1.0))
    weight_enabled = bool(getattr(settings, "price_weight_in_earn", False))
    floor = float(getattr(settings, "price_weight_floor", 0.85))
    ceil = float(getattr(settings, "price_weight_ceil", 1.15))
    missing = float(getattr(settings, "price_weight_missing", 1.0))

    list_p = list_price_per_hour
    cat_p = catalog_price_per_hour
    model_key = catalog_model_key
    # Auto-resolve capacity snaps only when weight path is on and caller did not
    # supply both sides (avoid extra SQL when flag off / fully explicit).
    if weight_enabled and (list_p is None or cat_p is None):
        try:
            snap = await resolve_earn_price_snapshot(session, attempt_id)
        except Exception:  # noqa: BLE001 — resolution is best-effort forensics
            logger.exception(
                "price_weight resolve failed for attempt_id=%s",
                attempt_id,
            )
            snap = {
                "list_price_per_hour": None,
                "catalog_price_per_hour": None,
                "catalog_model_key": None,
            }
        if list_p is None:
            list_p = snap.get("list_price_per_hour")
        if cat_p is None:
            cat_p = snap.get("catalog_price_per_hour")
        if not model_key:
            model_key = snap.get("catalog_model_key")

    weight = compute_price_weight(
        list_price=list_p,
        catalog_price=cat_p,
        enabled=weight_enabled,
        floor=floor,
        ceil=ceil,
        missing=missing,
    )
    mode = price_weight_mode(
        enabled=weight_enabled,
        list_price=list_p,
        catalog_price=cat_p,
    )

    delta = compute_score_earn_delta(
        float(score.composite),
        scale=scale,
        price_weight=weight,
    )
    if delta <= 0.0:
        # Zero / non-positive composite: never mint positive points (VAL-WGT-003).
        return None

    details: dict[str, Any] = {
        "composite": float(score.composite),
        "scale": float(scale),
        "delta": float(delta),
        "price_weight": float(weight),
        "price_weight_mode": mode,
        "factors": {
            "correctness": float(score.correctness),
            "efficiency": float(score.efficiency),
            "fabric_gate": float(score.fabric_gate),
            "tee_bonus": float(score.tee_bonus),
        },
    }
    # Forensic price fields when present (VAL-PRICE-061 assessment keys).
    list_pos = _positive_price_or_none(list_p)
    cat_pos = _positive_price_or_none(cat_p)
    if list_pos is not None:
        details["list_price_per_hour"] = float(list_pos)
    if cat_pos is not None:
        details["catalog_price_per_hour"] = float(cat_pos)
    if model_key:
        details["catalog_model_key"] = str(model_key)

    # SAVEPOINT so a unique-attempt race does not abort the outer score seal txn.
    try:
        async with session.begin_nested():
            balance_after = await _upsert_balance(session, hotkey=hotkey, delta=delta)
            row = PointsLedger(
                id=str(uuid.uuid4()),
                hotkey=hotkey,
                role=str(score.role) if score.role else None,
                delta=float(delta),
                balance_after=float(balance_after),
                reason=REASON_SCORE_EARN,
                score_id=str(score.id) if score.id else None,
                attempt_id=attempt_id,
                details_json=json.dumps(details, sort_keys=True),
                created_at=utc_now(),
            )
            session.add(row)
            await session.flush()
            return row
    except IntegrityError:
        # Concurrent seal won attempt_id unique — return winner, no double mint.
        existing = await get_ledger_for_attempt(session, attempt_id)
        if existing is not None:
            return existing
        logger.exception(
            "points earn IntegrityError for attempt_id=%s without winner row",
            attempt_id,
        )
        return None


async def earn_from_score_id(
    session: AsyncSession,
    score_id: str,
    *,
    hyper: HyperSettings | None = None,
    list_price_per_hour: float | None = None,
    catalog_price_per_hour: float | None = None,
    catalog_model_key: str | None = None,
) -> PointsLedger | None:
    """Lookup Score by id then earn (helper for seals that only have score id)."""

    result = await session.execute(select(Score).where(Score.id == score_id))
    score = result.scalar_one_or_none()
    if score is None:
        return None
    return await earn_from_score(
        session,
        score,
        hyper=hyper,
        list_price_per_hour=list_price_per_hour,
        catalog_price_per_hour=catalog_price_per_hour,
        catalog_model_key=catalog_model_key,
    )


__all__ = [
    "PRICE_WEIGHT_MODE_CATALOG_RATIO",
    "PRICE_WEIGHT_MODE_MISSING",
    "PRICE_WEIGHT_MODE_OFF",
    "REASON_ADMIN_ADJUST",
    "REASON_SCORE_EARN",
    "balance_row_to_public",
    "compute_price_weight",
    "compute_score_earn_delta",
    "earn_from_score",
    "earn_from_score_id",
    "get_balance_row",
    "get_ledger_for_attempt",
    "get_points_balance",
    "ledger_row_to_public",
    "list_points_balances",
    "list_points_history",
    "price_weight_mode",
    "resolve_earn_price_snapshot",
]
