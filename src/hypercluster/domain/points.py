"""Challenge-local points ledger earn from scored attempts (M10).

Earn rule (VAL-WGT-002 / 003 / 004, library/points-incentive.md)::

    if composite > 0 and points_enabled:
        delta = composite * HYPER_POINTS_SCALE   # default scale 1.0
    else:
        no positive score_earn mint

Idempotency: at most one ``score_earn`` ledger row per ``attempt_id``
(unique constraint + pre-check no-op). Balance rollup is updated in the same
session flush.

Downstream of the fixed four-factor product only — never a 5th scoring factor.
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

from hypercluster.db.models import PointsBalance, PointsLedger, Score, utc_now
from hypercluster.settings import HyperSettings, get_hyper_settings

logger = logging.getLogger(__name__)

REASON_SCORE_EARN = "score_earn"
REASON_ADMIN_ADJUST = "admin_adjust"

# Tolerance: composites this close to zero never mint positive mass.
_COMPOSITE_EPS = 0.0


def compute_score_earn_delta(
    composite: float,
    *,
    scale: float = 1.0,
) -> float:
    """Return the points delta for a score_earn, or 0.0 when no mint.

    Positive finite composite × non-negative finite scale → positive delta.
    Non-positive / non-finite composite, or non-positive / non-finite scale → 0.
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
    delta = c * s
    if not math.isfinite(delta) or delta <= 0.0:
        return 0.0
    return float(delta)


async def get_points_balance(session: AsyncSession, hotkey: str) -> float:
    """Current denormalized balance for hotkey (0.0 if never seen)."""

    result = await session.execute(
        select(PointsBalance).where(PointsBalance.hotkey == hotkey)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return 0.0
    try:
        bal = float(row.balance)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(bal):
        return 0.0
    return bal


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


async def earn_from_score(
    session: AsyncSession,
    score: Score,
    *,
    hyper: HyperSettings | None = None,
) -> PointsLedger | None:
    """Mint ``score_earn`` points from a fully scored attempt (idempotent).

    VAL-WGT-002: composite > 0 → positive ledger delta & balance increase.
    VAL-WGT-003: composite ≤ 0 / integrity zero → no positive mint (None).
    VAL-WGT-004: same ``attempt_id`` replay is a no-op (returns existing row).

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
    delta = compute_score_earn_delta(float(score.composite), scale=scale)
    if delta <= 0.0:
        # Zero / non-positive composite: never mint positive points (VAL-WGT-003).
        return None

    details: dict[str, Any] = {
        "composite": float(score.composite),
        "scale": float(scale),
        "delta": float(delta),
        "factors": {
            "correctness": float(score.correctness),
            "efficiency": float(score.efficiency),
            "fabric_gate": float(score.fabric_gate),
            "tee_bonus": float(score.tee_bonus),
        },
    }

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
) -> PointsLedger | None:
    """Lookup Score by id then earn (helper for seals that only have score id)."""

    result = await session.execute(select(Score).where(Score.id == score_id))
    score = result.scalar_one_or_none()
    if score is None:
        return None
    return await earn_from_score(session, score, hyper=hyper)


__all__ = [
    "REASON_ADMIN_ADJUST",
    "REASON_SCORE_EARN",
    "compute_score_earn_delta",
    "earn_from_score",
    "earn_from_score_id",
    "get_ledger_for_attempt",
    "get_points_balance",
]
