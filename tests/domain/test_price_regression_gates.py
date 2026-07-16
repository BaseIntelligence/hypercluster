"""M11 regression gates VAL-PRICE-070 / 071 / 072.

Pure preservative: pricing must not rewrite FakeSsh / M9 probe bank,
must leave M10 unit-sum incentives intact, and must keep attempt_id
earn idempotency even when price_weight ≠ 1.

Four-factor composite identity and M9 integrity zeros remain untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hypercluster.db.database import Base
from hypercluster.db.models import PointsLedger, Score, utc_now
from hypercluster.domain.incentive import (
    UNIT_SUM_TOLERANCE,
    finalize_incentives,
    normalize_sum_to_unit,
    weight_sum,
)
from hypercluster.domain.points import (
    REASON_SCORE_EARN,
    compute_price_weight,
    compute_score_earn_delta,
    earn_from_score,
    get_ledger_for_attempt,
    get_points_balance,
)
from hypercluster.domain.scoring import compute_four_factor
from hypercluster.probe.fixtures import (
    KNOWN_FIXTURE_NAMES,
    get_fixture,
    list_fixtures,
    load_fixture_json,
    package_fixture_dir,
)
from hypercluster.probe.pipeline import GpuProbeConfig, GpuProbeContext, run_gpu_probe
from hypercluster.probe.transport import FakeSshTransport
from hypercluster.settings import HyperSettings

# Stable digests of the M9 FakeSsh JSON bank as of fixture-bank ship
# (commit fcae051). Pricing work MUST NOT rewrite these files.
# VAL-PRICE-070 locks path + content fingerprint.
_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "gpu_probe"
_FIXTURE_SHA256: dict[str, str] = {
    "bench_fail.json": ("103165461202b078e67aeec4d52ec04ea01063e16d5bc380fb84f1f3a8c97238"),
    "docker_missing.json": ("3206c298af514713d24ff98a549abcaae1077c62b9056d36d78a885413075621"),
    "fingerprint_churn.json": ("3199ae29a3dbf0daa4cbe171640df7d065396c9e58341d4a1ec84ebe95eb1361"),
    "no_gpu.json": ("be883881cc61f207fc555d8da55d865eef101c4d43c9fc420be511634248c5a6"),
    "ssh_timeout.json": ("9fcb60feafe23f779bff35f4e13a986d27490813c98e243147f82fb948fa2bbf"),
    "uuid_clone.json": ("fef22ad9daa16435924aaa2edffc260695ca4b48d8c1be89b413d2b349e639f1"),
    "v100_pass_all.json": ("5cd49c2008a11386db6256f2c878b6898d045ee783b346af3059947fa28f970e"),
    "vram_lie.json": ("7ebc03ec4b2e400e1e5c1f0fbbe8171fba71ae8036638c1eae4ed1d1cf643223"),
    "wrong_model.json": ("cecc3d252ef5d2a00441337ab6baa82f700ad2aa84a3ff28ed757f77474d0029"),
}

_EXPECTED_FIXTURE_NAMES = frozenset(
    {
        "pass_all",
        "no_gpu",
        "wrong_model",
        "uuid_clone",
        "vram_lie",
        "bench_fail",
        "docker_missing",
        "ssh_timeout",
        "fingerprint_churn",
    }
)

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
        "incentive_sum_normalize": True,
        "weight_dust": 1e-12,
    }
    base.update(overrides)
    return HyperSettings(**base)


@pytest.fixture
async def db_session(tmp_path) -> AsyncSession:
    path = tmp_path / "price-regression.sqlite3"
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
# VAL-PRICE-070: FakeSsh fixture bank path + content stable, suite still runs
# ---------------------------------------------------------------------------


def test_val_price_070_fixture_bank_path_and_names_stable() -> None:
    """VAL-PRICE-070: gpu_probe fixture bank path + names unchanged by pricing."""

    assert _FIXTURE_DIR.is_dir(), f"missing fixture bank dir {_FIXTURE_DIR}"
    assert _FIXTURE_DIR.name == "gpu_probe"
    assert _FIXTURE_DIR.parent.name == "fixtures"
    # package helper must still resolve to the same bank (not a pricing rewrite).
    package = package_fixture_dir()
    assert package.resolve() == _FIXTURE_DIR.resolve()
    assert set(list_fixtures()) == _EXPECTED_FIXTURE_NAMES
    assert KNOWN_FIXTURE_NAMES == _EXPECTED_FIXTURE_NAMES


def test_val_price_070_fixture_json_sha256_unmodified() -> None:
    """VAL-PRICE-070: JSON content fingerprints locked; no pricing rewrite."""

    on_disk = sorted(p.name for p in _FIXTURE_DIR.glob("*.json"))
    assert on_disk == sorted(_FIXTURE_SHA256.keys()), f"fixture bank membership changed: {on_disk}"
    for name, expected in _FIXTURE_SHA256.items():
        path = _FIXTURE_DIR / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, (
            f"{name} content changed by pricing work (got {digest}, expected {expected})"
        )
        # Round-trip still loads without catalog dependency.
        loaded = load_fixture_json(path)
        assert loaded.scripted  # non-empty command scripts


def test_val_price_070_fake_ssh_matrix_still_green_smoke() -> None:
    """VAL-PRICE-070: representative FakeSsh matrix paths still pass/fail as designed.

    Full matrix stays in tests/probe/test_fake_ssh_transport_matrix.py; this gate
    re-runs pass_all + a fatal bank member so pricing cannot silently break CI.
    """

    # pass_all → verified pipeline under FakeSsh scripts
    fx_ok = get_fixture("pass_all")
    transport_ok = FakeSshTransport(scripted=fx_ok.scripted)
    evidence_ok = run_gpu_probe(
        transport_ok,
        GpuProbeContext(
            node_id="node-price-reg-ok",
            provider_hotkey="hk-provider-reg",
            ssh_endpoint="10.0.0.9:22",
            claimed=fx_ok.claimed,
            key_fingerprint="sha256:fake-key-fingerprint",
            occupied_uuids=set(fx_ok.occupied_uuids),
            prior_verified_uuids=(
                None if fx_ok.prior_verified_uuids is None else set(fx_ok.prior_verified_uuids)
            ),
        ),
        config=GpuProbeConfig(require_docker_runtime=fx_ok.require_docker_runtime),
    )
    assert evidence_ok.status == "passed"
    assert evidence_ok.failure_code is None

    # no_gpu fatal still fails without needing catalog
    fx_bad = get_fixture("no_gpu")
    transport_bad = FakeSshTransport(scripted=fx_bad.scripted)
    evidence_bad = run_gpu_probe(
        transport_bad,
        GpuProbeContext(
            node_id="node-price-reg-bad",
            provider_hotkey="hk-provider-reg",
            ssh_endpoint="10.0.0.9:22",
            claimed=fx_bad.claimed,
            key_fingerprint="sha256:fake-key-fingerprint",
            occupied_uuids=set(fx_bad.occupied_uuids),
        ),
        config=GpuProbeConfig(require_docker_runtime=fx_bad.require_docker_runtime),
    )
    assert evidence_bad.status == "failed"
    assert evidence_bad.failure_code == "nvidia_smi_list"


# ---------------------------------------------------------------------------
# VAL-PRICE-071: incentive unit-sum + empty {} still hold
# ---------------------------------------------------------------------------


def test_val_price_071_unit_sum_and_empty_burn_safe() -> None:
    """VAL-PRICE-071: M10 unit-sum + empty {} green under pricing defaults.

    mass_from_points remains deferred/default-off (not invoiced into weight
    mass); incentive_sum_normalize stays True; price_weight never fabricates
    unit mass when the composite window is empty.
    """

    hyper = HyperSettings(
        allow_insecure_signatures=True,
        points_enabled=True,
    )
    assert hyper.incentive_sum_normalize is True
    # Default price_weight off: ledger weight never injects into normalize defaults.
    assert hyper.price_weight_in_earn is False
    # Deferred flag not present on ship settings: mass stays composite-window sourced.
    assert "mass_from_points" not in type(hyper).model_fields

    mass = {"a": 15.0, "b": 3.0, "c": 2.0}
    weights = normalize_sum_to_unit(mass)
    assert weight_sum(weights) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
    assert weights["a"] > weights["b"] > weights["c"]

    # Empty / toxin mass stays burn-safe {}
    assert finalize_incentives({}) == {}
    assert finalize_incentives({"x": 0.0, "y": -1.0, "z": float("nan")}) == {}
    assert normalize_sum_to_unit({}) == {}

    # Multi-hotkey after clamp still unit-sums under default knobs.
    out = finalize_incentives(
        {"a": 10.0, "b": float("nan"), "c": -3.0, "d": 5.0},
        sum_normalize=True,
        dust=hyper.weight_dust,
    )
    assert set(out) == {"a", "d"}
    assert weight_sum(out) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)


def test_val_price_071_price_weight_does_not_fabricate_incentive_mass() -> None:
    """VAL-PRICE-071: ledger price_weight is not a 5th composite / raw weight factor.

    Mass still comes from composite window by default (mass_from_points=false);
    unit-sum stays intact even if earn tracts pick non-1.0 weight.
    """

    # Pure composite identity still four factors only.
    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=2.0,
        fabric_gate=1.0,
        tee_bonus=1.1,
        hyper=_hyper(price_weight_in_earn=True),
    )
    assert breakdown.composite == pytest.approx(1.0 * 2.0 * 1.0 * 1.1)

    # price_weight multiplies earn only AFTER the product.
    w = compute_price_weight(
        list_price=2.0,
        catalog_price=2.49,
        enabled=True,
        floor=0.85,
        ceil=1.15,
        missing=1.0,
    )
    assert 1.0 < w <= 1.15 + 1e-12
    earn_delta = compute_score_earn_delta(breakdown.composite, scale=1.0, price_weight=w)
    assert earn_delta == pytest.approx(breakdown.composite * w)
    # Incentive path on composite-like mass still unit-sums; price does not add key.
    composite_mass = {"hkA": breakdown.composite, "hkB": 1.0}
    incentives = finalize_incentives(composite_mass, sum_normalize=True)
    assert weight_sum(incentives) == pytest.approx(1.0, abs=UNIT_SUM_TOLERANCE)
    assert set(incentives) == {"hkA", "hkB"}


# ---------------------------------------------------------------------------
# VAL-PRICE-072: double earn same attempt_id single mint under price_weight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_val_price_072_idempotent_earn_with_price_weight_on(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-072: two earn calls same attempt_id → single ledger mint."""

    attempt_id = str(uuid.uuid4())
    score_id = str(uuid.uuid4())
    composite = 3.0
    list_p = 2.0
    cat_p = 2.49
    expected_w = min(1.15, cat_p / list_p)
    assert expected_w != pytest.approx(1.0)  # non-neutral weight
    expected_delta = composite * 1.0 * expected_w

    hyper = _hyper(price_weight_in_earn=True, points_scale=1.0)
    score = _score(composite=composite, attempt_id=attempt_id, score_id=score_id)

    first = await earn_from_score(
        db_session,
        score,
        hyper=hyper,
        list_price_per_hour=list_p,
        catalog_price_per_hour=cat_p,
        catalog_model_key="H100_80GB",
    )
    await db_session.commit()
    assert first is not None
    assert first.reason == REASON_SCORE_EARN
    assert first.delta == pytest.approx(expected_delta)
    assert first.delta > composite  # bargain boost applied once
    assert first.attempt_id == attempt_id
    details = json.loads(first.details_json or "{}")
    assert details["price_weight"] == pytest.approx(expected_w)
    assert details["price_weight_mode"] == "catalog_ratio"

    # Second seal with the same attempt_id must not double-mint (even with weight ≠ 1).
    second = await earn_from_score(
        db_session,
        _score(composite=composite, attempt_id=attempt_id, score_id=score_id),
        hyper=hyper,
        list_price_per_hour=list_p,
        catalog_price_per_hour=cat_p,
        catalog_model_key="H100_80GB",
    )
    await db_session.commit()
    assert second is not None
    assert second.id == first.id
    assert second.delta == pytest.approx(expected_delta)

    bal = await get_points_balance(db_session, HOTKEY_A)
    assert bal == pytest.approx(expected_delta)

    ledger = await get_ledger_for_attempt(db_session, attempt_id)
    assert ledger is not None
    assert ledger.id == first.id

    all_rows = (
        (
            await db_session.execute(
                select(PointsLedger).where(PointsLedger.attempt_id == attempt_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(all_rows) == 1
    # Hotkey total rows still one positive score_earn for this attempt
    hotkey_rows = (
        (await db_session.execute(select(PointsLedger).where(PointsLedger.hotkey == HOTKEY_A)))
        .scalars()
        .all()
    )
    assert len(hotkey_rows) == 1


@pytest.mark.asyncio
async def test_val_price_072_idempotent_under_gouge_floor(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-072: gouge floor (weight < 1) still single-mints on retry."""

    attempt_id = str(uuid.uuid4())
    composite = 4.0
    list_p = 10.0
    cat_p = 1.0
    floor = 0.85
    expected = composite * floor
    hyper = _hyper(price_weight_in_earn=True, price_weight_floor=floor)

    score = _score(composite=composite, attempt_id=attempt_id)
    a = await earn_from_score(
        db_session,
        score,
        hyper=hyper,
        list_price_per_hour=list_p,
        catalog_price_per_hour=cat_p,
    )
    await db_session.commit()
    b = await earn_from_score(
        db_session,
        _score(composite=composite, attempt_id=attempt_id, score_id=score.id),
        hyper=hyper,
        list_price_per_hour=list_p,
        catalog_price_per_hour=cat_p,
    )
    await db_session.commit()
    assert a is not None and b is not None
    assert a.id == b.id
    assert a.delta == pytest.approx(expected)
    assert await get_points_balance(db_session, HOTKEY_A) == pytest.approx(expected)
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


def test_val_price_070_072_four_factor_and_math_guards() -> None:
    """Cross-check: pricing knobs never reintroduce non-finite mass into formula."""

    hyp = _hyper(price_weight_in_earn=True, points_scale=7.0)
    bd = compute_four_factor(
        correctness=1.0,
        efficiency=1.5,
        fabric_gate=1.0,
        tee_bonus=1.0,
        hyper=hyp,
    )
    assert math.isfinite(bd.composite)
    assert bd.composite == pytest.approx(1.5)
    delta = compute_score_earn_delta(bd.composite, scale=7.0, price_weight=1.15)
    assert math.isfinite(delta)
    assert delta == pytest.approx(1.5 * 7.0 * 1.15)
