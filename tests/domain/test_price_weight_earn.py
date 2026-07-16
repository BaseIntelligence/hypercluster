"""VAL-PRICE-060..063: optional price_weight on score_earn (default OFF).

M11 earn slice:
- Flag off: delta == composite * HYPER_POINTS_SCALE (M10 parity).
- Flag on + bargain list < catalog → weight in (1, ceil]; gouge floors.
- Missing/≤0 list or catalog → HYPER_PRICE_WEIGHT_MISSING (default 1.0).
- compute_four_factor identity pure (no fifth factor); integrity zero → no mint.

Ledger details_json records price_weight / list / catalog when weight path applied.
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
from hypercluster.db.models import PointsLedger, Score, utc_now
from hypercluster.domain.points import (
    REASON_SCORE_EARN,
    compute_price_weight,
    compute_score_earn_delta,
    earn_from_score,
    get_ledger_for_attempt,
    get_points_balance,
)
from hypercluster.domain.scoring import compute_four_factor
from hypercluster.settings import HyperSettings

HOTKEY_A = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def _hyper(**overrides: Any) -> HyperSettings:
    base: dict[str, Any] = {
        "allow_insecure_signatures": True,
        "points_enabled": True,
        "points_scale": 1.0,
        "price_weight_in_earn": False,
        "price_weight_floor": 0.85,
        "price_weight_ceil": 1.15,
        "price_weight_missing": 1.0,
    }
    base.update(overrides)
    return HyperSettings(**base)


@pytest.fixture
async def db_session(tmp_path) -> AsyncSession:
    path = tmp_path / "price-weight-earn.sqlite3"
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
    composite: float = 2.0,
    correctness: float = 1.0,
    efficiency: float | None = None,
    fabric_gate: float = 1.0,
    tee_bonus: float = 1.0,
    attempt_id: str | None = None,
    role: str = "demand",
    score_id: str | None = None,
) -> Score:
    if efficiency is None:
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
# Pure compute_price_weight
# ---------------------------------------------------------------------------


def test_compute_price_weight_flag_off_is_one() -> None:
    """When disabled, weight is always 1.0 regardless of prices."""

    w = compute_price_weight(
        list_price=1.0,
        catalog_price=10.0,
        enabled=False,
        floor=0.85,
        ceil=1.15,
        missing=1.0,
    )
    assert w == pytest.approx(1.0)


def test_compute_price_weight_bargain_clamped_to_ceil() -> None:
    """P_list < P_cat → ratio > 1, clamped to ceil (VAL-PRICE-061)."""

    # catalog 2.49 / list 2.00 = 1.245 → clamp to 1.15
    w = compute_price_weight(
        list_price=2.0,
        catalog_price=2.49,
        enabled=True,
        floor=0.85,
        ceil=1.15,
        missing=1.0,
    )
    assert w == pytest.approx(1.15)
    assert 1.0 < w <= 1.15 + 1e-12


def test_compute_price_weight_bargain_inside_band() -> None:
    """Mild bargain stays between 1 and ceil."""

    # 2.49 / 2.40 ≈ 1.0375
    w = compute_price_weight(
        list_price=2.40,
        catalog_price=2.49,
        enabled=True,
        floor=0.85,
        ceil=1.15,
        missing=1.0,
    )
    assert 1.0 < w < 1.15
    assert w == pytest.approx(2.49 / 2.40)


def test_compute_price_weight_gouge_floors() -> None:
    """P_list >> P_cat → ratio < floor; weight floors at floor."""

    # 1.0 / 10.0 = 0.1 → floor 0.85
    w = compute_price_weight(
        list_price=10.0,
        catalog_price=1.0,
        enabled=True,
        floor=0.85,
        ceil=1.15,
        missing=1.0,
    )
    assert w == pytest.approx(0.85)


def test_compute_price_weight_missing_neutral() -> None:
    """Missing/≤0 list or catalog → missing default 1.0 (VAL-PRICE-062)."""

    for list_p, cat_p in (
        (None, 2.0),
        (2.0, None),
        (None, None),
        (0.0, 2.0),
        (2.0, 0.0),
        (-1.0, 2.0),
        (2.0, -1.0),
    ):
        w = compute_price_weight(
            list_price=list_p,
            catalog_price=cat_p,
            enabled=True,
            floor=0.85,
            ceil=1.15,
            missing=1.0,
        )
        assert w == pytest.approx(1.0), (list_p, cat_p, w)


def test_compute_price_weight_custom_missing() -> None:
    w = compute_price_weight(
        list_price=None,
        catalog_price=None,
        enabled=True,
        floor=0.85,
        ceil=1.15,
        missing=0.95,
    )
    assert w == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# VAL-PRICE-060: flag off parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_delta_is_composite_times_scale(db_session: AsyncSession) -> None:
    """VAL-PRICE-060: price_weight_in_earn=false → delta == composite * scale."""

    score = _score(composite=2.5)
    row = await earn_from_score(
        db_session,
        score,
        hyper=_hyper(price_weight_in_earn=False, points_scale=2.0),
        list_price_per_hour=1.0,
        catalog_price_per_hour=10.0,  # would be huge ratio if applied
    )
    await db_session.commit()
    assert row is not None
    assert row.delta == pytest.approx(5.0)  # 2.5 * 2.0, no price term
    details = json.loads(row.details_json or "{}")
    assert details["composite"] == pytest.approx(2.5)
    assert details["scale"] == pytest.approx(2.0)
    # Flag off: no applied price_weight boosts/cuts (mode off or absent *effect*)
    # If recorded, weight must be neutral 1.0 / mode off.
    if "price_weight" in details:
        assert details["price_weight"] == pytest.approx(1.0)
        assert details.get("price_weight_mode") in (None, "off", "disabled")
    assert await get_points_balance(db_session, HOTKEY_A) == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_flag_off_default_settings_parity(db_session: AsyncSession) -> None:
    """Default HyperSettings.price_weight_in_earn is False (M10 parity)."""

    hyper = HyperSettings(
        allow_insecure_signatures=True,
        points_enabled=True,
        points_scale=1.0,
    )
    assert hyper.price_weight_in_earn is False
    score = _score(composite=3.0)
    row = await earn_from_score(db_session, score, hyper=hyper)
    await db_session.commit()
    assert row is not None
    assert row.delta == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# VAL-PRICE-061: bargain + gouge clamp when flag on
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bargain_list_earns_weight_above_one(db_session: AsyncSession) -> None:
    """VAL-PRICE-061: P_list < P_cat → weight in (1, ceil]; delta multiplies."""

    composite = 2.0
    scale = 1.0
    list_p = 2.0
    cat_p = 2.49
    expected_w = min(1.15, cat_p / list_p)  # 1.15 (ceil)
    assert 1.0 < expected_w <= 1.15

    score = _score(composite=composite)
    row = await earn_from_score(
        db_session,
        score,
        hyper=_hyper(price_weight_in_earn=True, points_scale=scale),
        list_price_per_hour=list_p,
        catalog_price_per_hour=cat_p,
        catalog_model_key="H100_80GB",
    )
    await db_session.commit()
    assert row is not None
    assert row.reason == REASON_SCORE_EARN
    assert row.delta == pytest.approx(composite * scale * expected_w)
    assert row.delta > composite * scale  # bargain bonus

    details = json.loads(row.details_json or "{}")
    assert details["price_weight"] == pytest.approx(expected_w)
    assert details["list_price_per_hour"] == pytest.approx(list_p)
    assert details["catalog_price_per_hour"] == pytest.approx(cat_p)
    assert details["catalog_model_key"] == "H100_80GB"
    assert details["price_weight_mode"] == "catalog_ratio"
    # Four factors only in factors dump — price is ledger-only.
    assert set(details["factors"].keys()) == {
        "correctness",
        "efficiency",
        "fabric_gate",
        "tee_bonus",
    }


@pytest.mark.asyncio
async def test_gouge_list_floors_weight(db_session: AsyncSession) -> None:
    """Gouge floors at HYPER_PRICE_WEIGHT_FLOOR (does not zero honest work)."""

    composite = 4.0
    list_p = 10.0
    cat_p = 1.0
    floor = 0.85
    expected_w = floor  # ratio 0.1 → floor

    score = _score(composite=composite)
    row = await earn_from_score(
        db_session,
        score,
        hyper=_hyper(price_weight_in_earn=True, price_weight_floor=floor),
        list_price_per_hour=list_p,
        catalog_price_per_hour=cat_p,
    )
    await db_session.commit()
    assert row is not None
    assert row.delta == pytest.approx(composite * 1.0 * expected_w)
    details = json.loads(row.details_json or "{}")
    assert details["price_weight"] == pytest.approx(floor)
    assert details["price_weight_mode"] == "catalog_ratio"


# ---------------------------------------------------------------------------
# VAL-PRICE-062: missing prices neutral
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_prices_use_missing_default(db_session: AsyncSession) -> None:
    """VAL-PRICE-062: missing list/catalog → weight == missing (1.0 default)."""

    composite = 1.5
    score = _score(composite=composite)
    row = await earn_from_score(
        db_session,
        score,
        hyper=_hyper(price_weight_in_earn=True),
        # omit price kwargs → missing
    )
    await db_session.commit()
    assert row is not None
    assert row.delta == pytest.approx(composite * 1.0 * 1.0)
    details = json.loads(row.details_json or "{}")
    assert details["price_weight"] == pytest.approx(1.0)
    assert details["price_weight_mode"] == "missing"


@pytest.mark.asyncio
async def test_zero_catalog_price_is_missing_neutral(db_session: AsyncSession) -> None:
    score = _score(composite=2.0)
    row = await earn_from_score(
        db_session,
        score,
        hyper=_hyper(price_weight_in_earn=True),
        list_price_per_hour=2.0,
        catalog_price_per_hour=0.0,
    )
    await db_session.commit()
    assert row is not None
    assert row.delta == pytest.approx(2.0)
    details = json.loads(row.details_json or "{}")
    assert details["price_weight"] == pytest.approx(1.0)
    assert details["price_weight_mode"] == "missing"


# ---------------------------------------------------------------------------
# VAL-PRICE-063: four-factor pure + integrity zero no mint
# ---------------------------------------------------------------------------


def test_compute_four_factor_identity_pure() -> None:
    """VAL-PRICE-063: composite remains product of four factors only."""

    hyper = _hyper(price_weight_in_earn=True, points_scale=50.0)
    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=2.0,
        fabric_gate=1.0,
        tee_bonus=1.1,
        hyper=hyper,
    )
    assert breakdown.composite == pytest.approx(1.0 * 2.0 * 1.0 * 1.1)
    # Earn multiplies price AFTER the product, never inside formula API.
    base = compute_score_earn_delta(breakdown.composite, scale=50.0)
    assert base == pytest.approx(breakdown.composite * 50.0)
    w = compute_price_weight(
        list_price=1.0,
        catalog_price=1.1,
        enabled=True,
        floor=0.85,
        ceil=1.15,
        missing=1.0,
    )
    assert base * w == pytest.approx(breakdown.composite * 50.0 * w)
    # Product identity holds regardless of catalog trauma.
    assert math.isfinite(breakdown.composite)


@pytest.mark.asyncio
async def test_integrity_zero_still_no_positive_mint(db_session: AsyncSession) -> None:
    """VAL-PRICE-063: integrity composite 0 never mints even with bargain prices."""

    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=10.0,
        fabric_gate=1.0,
        tee_bonus=1.2,
        integrity_codes=["inventory_spoof"],
        hyper=_hyper(price_weight_in_earn=True),
    )
    assert breakdown.composite == pytest.approx(0.0)
    score = _score(
        composite=breakdown.composite,
        correctness=breakdown.correctness,
        efficiency=breakdown.efficiency,
        fabric_gate=breakdown.fabric_gate,
        tee_bonus=breakdown.tee_bonus,
    )
    row = await earn_from_score(
        db_session,
        score,
        hyper=_hyper(price_weight_in_earn=True),
        list_price_per_hour=1.0,
        catalog_price_per_hour=10.0,  # would be big bonus if composite > 0
    )
    await db_session.commit()
    assert row is None
    assert await get_points_balance(db_session, HOTKEY_A) == 0.0
    rows = (
        (
            await db_session.execute(
                select(PointsLedger).where(PointsLedger.attempt_id == score.attempt_id)
            )
        )
        .scalars()
        .all()
    )
    assert rows == []


@pytest.mark.asyncio
async def test_price_weight_earn_idempotent(db_session: AsyncSession) -> None:
    """Idempotency holds when price_weight ≠ 1 (VAL-PRICE-072 preview)."""

    attempt_id = str(uuid.uuid4())
    score = _score(composite=2.0, attempt_id=attempt_id)
    hyper = _hyper(price_weight_in_earn=True)
    first = await earn_from_score(
        db_session,
        score,
        hyper=hyper,
        list_price_per_hour=2.0,
        catalog_price_per_hour=2.49,
    )
    await db_session.commit()
    assert first is not None
    expected = 2.0 * min(1.15, 2.49 / 2.0)
    assert first.delta == pytest.approx(expected)

    second = await earn_from_score(
        db_session,
        _score(composite=2.0, attempt_id=attempt_id, score_id=score.id),
        hyper=hyper,
        list_price_per_hour=2.0,
        catalog_price_per_hour=2.49,
    )
    await db_session.commit()
    assert second is not None
    assert second.id == first.id
    assert await get_points_balance(db_session, HOTKEY_A) == pytest.approx(expected)
    assert await get_ledger_for_attempt(db_session, attempt_id) is not None
    all_rows = (
        (await db_session.execute(select(PointsLedger).where(PointsLedger.hotkey == HOTKEY_A)))
        .scalars()
        .all()
    )
    assert len(all_rows) == 1


def test_package_exports_price_weight() -> None:
    from hypercluster.domain import points as p

    assert hasattr(p, "compute_price_weight")
    assert callable(p.compute_price_weight)
