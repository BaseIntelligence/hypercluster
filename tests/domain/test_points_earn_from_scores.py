"""VAL-WGT-002 / 003 / 004: earn points from fully scored attempts.

M10 earn slice:
- Positive composite → positive ledger delta + balance increase.
- Composite ≤ 0 / integrity zero → no positive score_earn mint.
- Replay same attempt_id is idempotent (no double count).
- HYPER_POINTS_SCALE optional knob; formula remains four-factor product only.
"""

from __future__ import annotations

import json
import math
import uuid
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hypercluster.db.database import Base
from hypercluster.db.models import Job, JobAttempt, PointsBalance, PointsLedger, Score, utc_now
from hypercluster.domain.points import (
    REASON_SCORE_EARN,
    compute_score_earn_delta,
    earn_from_score,
    get_ledger_for_attempt,
    get_points_balance,
)
from hypercluster.domain.scoring import compute_four_factor
from hypercluster.settings import HyperSettings

HOTKEY_A = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
HOTKEY_B = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def _hyper(**overrides: Any) -> HyperSettings:
    base: dict[str, Any] = {
        "allow_insecure_signatures": True,
        "points_enabled": True,
        "points_scale": 1.0,
    }
    base.update(overrides)
    return HyperSettings(**base)


@pytest.fixture
async def db_session(tmp_path) -> AsyncSession:
    path = tmp_path / "points-earn.sqlite3"
    url = f"sqlite+aiosqlite:///{path}"
    engine = create_async_engine(url, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session
        await session.rollback()
    await engine.dispose()


def _score(
    *,
    hotkey: str = HOTKEY_A,
    composite: float = 2.5,
    correctness: float = 1.0,
    efficiency: float | None = None,
    fabric_gate: float = 1.0,
    tee_bonus: float = 1.0,
    attempt_id: str | None = None,
    role: str = "demand",
    score_id: str | None = None,
) -> Score:
    if efficiency is None:
        # Reconstruct a plausible efficiency when composite is positive.
        if correctness > 0 and fabric_gate > 0 and tee_bonus > 0 and composite > 0:
            efficiency = float(composite) / (
                float(correctness) * float(fabric_gate) * float(tee_bonus)
            )
        else:
            efficiency = 0.0
    return Score(
        id=score_id or str(uuid.uuid4()),
        attempt_id=attempt_id or str(uuid.uuid4()),
        hotkey=hotkey,
        role=role,
        correctness=float(correctness),
        efficiency=float(efficiency),
        fabric_gate=float(fabric_gate),
        tee_bonus=float(tee_bonus),
        composite=float(composite),
        details_json=None,
        created_at=utc_now(),
    )


# ---------------------------------------------------------------------------
# Pure delta helper
# ---------------------------------------------------------------------------


def test_compute_score_earn_delta_positive() -> None:
    assert compute_score_earn_delta(2.5, scale=1.0) == pytest.approx(2.5)
    assert compute_score_earn_delta(2.5, scale=2.0) == pytest.approx(5.0)
    assert compute_score_earn_delta(0.1, scale=1.0) == pytest.approx(0.1)


def test_compute_score_earn_delta_non_positive_composite() -> None:
    assert compute_score_earn_delta(0.0) == 0.0
    assert compute_score_earn_delta(-1.0) == 0.0
    assert compute_score_earn_delta(float("nan")) == 0.0
    assert compute_score_earn_delta(float("inf")) == 0.0


def test_compute_score_earn_delta_bad_scale() -> None:
    assert compute_score_earn_delta(3.0, scale=0.0) == 0.0
    assert compute_score_earn_delta(3.0, scale=-2.0) == 0.0
    assert compute_score_earn_delta(3.0, scale=float("nan")) == 0.0


# ---------------------------------------------------------------------------
# VAL-WGT-002: positive composite mints positive points
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_positive_composite_earns_positive_points(db_session: AsyncSession) -> None:
    """VAL-WGT-002: composite > 0 → positive ledger delta and balance increase."""

    score = _score(composite=2.5, correctness=1.0, fabric_gate=1.0, tee_bonus=1.0)
    before = await get_points_balance(db_session, HOTKEY_A)
    assert before == 0.0

    row = await earn_from_score(db_session, score, hyper=_hyper())
    await db_session.commit()

    assert row is not None
    assert row.reason == REASON_SCORE_EARN
    assert row.delta == pytest.approx(2.5)
    assert row.delta > 0.0
    assert row.balance_after == pytest.approx(2.5)
    assert row.attempt_id == score.attempt_id
    assert row.score_id == score.id
    assert row.hotkey == HOTKEY_A

    bal = await get_points_balance(db_session, HOTKEY_A)
    assert bal == pytest.approx(2.5)
    assert bal > before

    stored = (
        await db_session.execute(
            select(PointsLedger).where(PointsLedger.attempt_id == score.attempt_id)
        )
    ).scalar_one()
    assert stored.delta == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_points_scale_multiplies_earn(db_session: AsyncSession) -> None:
    """HYPER_POINTS_SCALE multiplies composite; formula remains four-factor only."""

    score = _score(composite=1.25)
    row = await earn_from_score(db_session, score, hyper=_hyper(points_scale=4.0))
    await db_session.commit()
    assert row is not None
    assert row.delta == pytest.approx(5.0)
    details = json.loads(row.details_json or "{}")
    assert details["scale"] == pytest.approx(4.0)
    assert details["composite"] == pytest.approx(1.25)
    # Scale is a ledger knob only — factors dump keeps exactly four product terms.
    factors = details["factors"]
    assert set(factors.keys()) == {
        "correctness",
        "efficiency",
        "fabric_gate",
        "tee_bonus",
    }


@pytest.mark.asyncio
async def test_points_disabled_skips_mint(db_session: AsyncSession) -> None:
    score = _score(composite=9.0)
    row = await earn_from_score(db_session, score, hyper=_hyper(points_enabled=False))
    await db_session.commit()
    assert row is None
    assert await get_points_balance(db_session, HOTKEY_A) == 0.0


# ---------------------------------------------------------------------------
# VAL-WGT-003: non-positive composite never mints positive points
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_composite_does_not_mint(db_session: AsyncSession) -> None:
    """VAL-WGT-003: composite 0 → no score_earn row, balance unchanged."""

    score = _score(
        composite=0.0,
        correctness=0.0,
        efficiency=5.0,
        fabric_gate=1.0,
        tee_bonus=1.0,
    )
    row = await earn_from_score(db_session, score, hyper=_hyper())
    await db_session.commit()
    assert row is None
    assert await get_points_balance(db_session, HOTKEY_A) == 0.0
    count = (
        (
            await db_session.execute(
                select(PointsLedger).where(PointsLedger.attempt_id == score.attempt_id)
            )
        )
        .scalars()
        .all()
    )
    assert count == []


@pytest.mark.asyncio
async def test_integrity_zero_does_not_mint_positive(db_session: AsyncSession) -> None:
    """VAL-WGT-003: integrity path composite 0 never mints positive mass."""

    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=10.0,
        fabric_gate=1.0,
        tee_bonus=1.2,
        integrity_codes=["inventory_spoof"],
        hyper=_hyper(),
    )
    assert breakdown.composite == pytest.approx(0.0)
    score = _score(
        composite=breakdown.composite,
        correctness=breakdown.correctness,
        efficiency=breakdown.efficiency,
        fabric_gate=breakdown.fabric_gate,
        tee_bonus=breakdown.tee_bonus,
    )
    row = await earn_from_score(db_session, score, hyper=_hyper())
    await db_session.commit()
    assert row is None
    assert await get_points_balance(db_session, HOTKEY_A) == 0.0


@pytest.mark.asyncio
async def test_negative_composite_does_not_mint(db_session: AsyncSession) -> None:
    # Defensive: engine should never write negatives, but earn must not mint.
    score = _score(composite=-3.0, correctness=1.0, efficiency=1.0)
    row = await earn_from_score(db_session, score, hyper=_hyper())
    await db_session.commit()
    assert row is None
    assert await get_points_balance(db_session, HOTKEY_A) == 0.0


# ---------------------------------------------------------------------------
# VAL-WGT-004: idempotent per attempt_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_earn_idempotent_per_attempt_id(db_session: AsyncSession) -> None:
    """VAL-WGT-004: 2× earn for same attempt_id does not double balance."""

    attempt_id = str(uuid.uuid4())
    score = _score(composite=3.0, attempt_id=attempt_id)
    first = await earn_from_score(db_session, score, hyper=_hyper())
    await db_session.commit()
    assert first is not None
    assert first.delta == pytest.approx(3.0)

    # Replay with a new Score instance / same attempt_id (score seal re-write path).
    score2 = _score(
        composite=3.0,
        attempt_id=attempt_id,
        score_id=score.id,
        hotkey=HOTKEY_A,
    )
    second = await earn_from_score(db_session, score2, hyper=_hyper())
    await db_session.commit()
    assert second is not None
    assert second.id == first.id
    assert second.delta == pytest.approx(3.0)

    bal = await get_points_balance(db_session, HOTKEY_A)
    assert bal == pytest.approx(3.0)

    rows = (
        (
            await db_session.execute(
                select(PointsLedger).where(PointsLedger.attempt_id == attempt_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_distinct_attempts_accumulate(db_session: AsyncSession) -> None:
    a1 = _score(composite=1.0, attempt_id=str(uuid.uuid4()))
    a2 = _score(composite=2.0, attempt_id=str(uuid.uuid4()))
    await earn_from_score(db_session, a1, hyper=_hyper())
    await earn_from_score(db_session, a2, hyper=_hyper())
    await db_session.commit()
    assert await get_points_balance(db_session, HOTKEY_A) == pytest.approx(3.0)
    ledger = (
        (await db_session.execute(select(PointsLedger).where(PointsLedger.hotkey == HOTKEY_A)))
        .scalars()
        .all()
    )
    assert len(ledger) == 2


# ---------------------------------------------------------------------------
# Hook: persist_score_for_attempt triggers earn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_score_earns_positive_composite(
    settings_factory: Any,
    tmp_path: Any,
) -> None:
    """Score seal path: positive composite → durable score_earn (VAL-WGT-002)."""

    from hypercluster.app import create_app
    from hypercluster.domain.scoring_tee import persist_score_for_attempt

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'earn-hook.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    hyper = _hyper(points_scale=1.0)
    app = create_app(settings, hyper_settings=hyper)
    attempt_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            session.add(
                Job(
                    id=job_id,
                    submitter_hotkey=HOTKEY_A,
                    status="succeeded",
                    image_digest=(
                        "sha256:sim000000000000000000000000000000000000000000000000000000000001"
                    ),
                    entrypoint_json='["python","-c","pass"]',
                    world_size=1,
                    nnodes=1,
                    nproc_per_node=1,
                    backend="nccl",
                    fabric_mode="auto",
                    tee_mode="none",
                    resource_json="{}",
                    timeout_s=60,
                )
            )
            session.add(
                JobAttempt(
                    id=attempt_id,
                    job_id=job_id,
                    attempt_no=1,
                    status="succeeded",
                )
            )
            await session.flush()
            row = await persist_score_for_attempt(
                session,
                attempt_id=attempt_id,
                hotkey=HOTKEY_A,
                role="demand",
                correctness=1.0,
                efficiency=2.0,
                fabric_gate=1.0,
                proof=None,
                tee_mode="none",
                hyper=hyper,
            )
            await session.commit()
            assert row.composite == pytest.approx(2.0)

            ledger = await get_ledger_for_attempt(session, attempt_id)
            assert ledger is not None
            assert ledger.reason == REASON_SCORE_EARN
            assert ledger.delta == pytest.approx(2.0)
            assert ledger.balance_after == pytest.approx(2.0)
            assert await get_points_balance(session, HOTKEY_A) == pytest.approx(2.0)

            # Replay seal: score updates, points stay single mint (VAL-WGT-004).
            row2 = await persist_score_for_attempt(
                session,
                attempt_id=attempt_id,
                hotkey=HOTKEY_A,
                role="demand",
                correctness=1.0,
                efficiency=2.0,
                fabric_gate=1.0,
                proof=None,
                tee_mode="none",
                hyper=hyper,
            )
            await session.commit()
            assert row2.id == row.id
            ledger2 = await get_ledger_for_attempt(session, attempt_id)
            assert ledger2 is not None
            assert ledger2.id == ledger.id
            assert await get_points_balance(session, HOTKEY_A) == pytest.approx(2.0)
            all_rows = (
                (
                    await session.execute(
                        select(PointsLedger).where(PointsLedger.hotkey == HOTKEY_A)
                    )
                )
                .scalars()
                .all()
            )
            assert len(all_rows) == 1


@pytest.mark.asyncio
async def test_persist_score_integrity_zero_no_points(
    settings_factory: Any,
    tmp_path: Any,
) -> None:
    """Score seal with integrity zero: composite 0, no positive mint (VAL-WGT-003)."""

    from hypercluster.app import create_app
    from hypercluster.domain.scoring_tee import persist_score_for_attempt

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'earn-zero.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    hyper = _hyper()
    app = create_app(settings, hyper_settings=hyper)
    attempt_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            session.add(
                Job(
                    id=job_id,
                    submitter_hotkey=HOTKEY_A,
                    status="failed",
                    image_digest=(
                        "sha256:sim000000000000000000000000000000000000000000000000000000000001"
                    ),
                    entrypoint_json='["python","-c","pass"]',
                    world_size=1,
                    nnodes=1,
                    nproc_per_node=1,
                    backend="nccl",
                    fabric_mode="auto",
                    tee_mode="none",
                    resource_json="{}",
                    timeout_s=60,
                )
            )
            session.add(
                JobAttempt(
                    id=attempt_id,
                    job_id=job_id,
                    attempt_no=1,
                    status="failed",
                )
            )
            await session.flush()
            row = await persist_score_for_attempt(
                session,
                attempt_id=attempt_id,
                hotkey=HOTKEY_A,
                role="demand",
                correctness=1.0,
                efficiency=5.0,
                fabric_gate=1.0,
                proof=None,
                tee_mode="none",
                hyper=hyper,
                details={"integrity_codes": ["inventory_spoof"]},
            )
            await session.commit()
            assert row.composite == pytest.approx(0.0)
            assert await get_ledger_for_attempt(session, attempt_id) is None
            assert await get_points_balance(session, HOTKEY_A) == 0.0
            balances = (await session.execute(select(PointsBalance))).scalars().all()
            assert balances == []


def test_four_factor_product_unchanged_under_points_scale() -> None:
    """VAL-WGT-022 regression guard: scale must not enter the product formula."""

    hyper = _hyper(points_scale=100.0)
    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=2.0,
        fabric_gate=1.0,
        tee_bonus=1.1,
        hyper=hyper,
    )
    assert breakdown.composite == pytest.approx(1.0 * 2.0 * 1.0 * 1.1)
    # Earn delta multiplies after product, not inside it.
    assert compute_score_earn_delta(breakdown.composite, scale=100.0) == pytest.approx(
        breakdown.composite * 100.0
    )
    assert math.isfinite(breakdown.composite)


def test_package_exports_points_module() -> None:
    from hypercluster.domain import points as p

    assert hasattr(p, "earn_from_score")
    assert hasattr(p, "compute_score_earn_delta")
    assert p.REASON_SCORE_EARN == "score_earn"
