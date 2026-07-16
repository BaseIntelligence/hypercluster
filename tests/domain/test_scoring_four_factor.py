"""VAL-SCORE-001..007, 021, 026: four-factor composite product engine."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.db.models import Job, JobAttempt
from hypercluster.domain.scoring import (
    CHEAT_REASON_CODES,
    ScoreBreakdown,
    apply_efficiency_floor,
    coerce_fabric_gate,
    coerce_gate01,
    coerce_tee_bonus,
    compute_four_factor,
    score_breakdown_to_public,
)
from hypercluster.domain.scoring_tee import persist_score_for_attempt
from hypercluster.settings import HyperSettings

# ----- helpers ---------------------------------------------------------------


def _hyper(**overrides: Any) -> HyperSettings:
    base: dict[str, Any] = {
        "tee_bonus_tdx": 1.08,
        "tee_bonus_tdx_gpu": 1.20,
        "efficiency_floor": 0.0,
        "score_window_attempts": 50,
    }
    base.update(overrides)
    return HyperSettings(**base)


def _approx_product(c: float, e: float, f: float, t: float) -> float:
    return float(c) * float(e) * float(f) * float(t)


def _job(*, hotkey: str, job_id: str | None = None) -> Job:
    return Job(
        id=job_id or str(uuid.uuid4()),
        submitter_hotkey=hotkey,
        status="succeeded",
        image_digest=("sha256:sim000000000000000000000000000000000000000000000000000000000001"),
        entrypoint_json=json.dumps(["python", "-m", "train"]),
        world_size=1,
        nnodes=1,
        nproc_per_node=1,
        backend="nccl",
        fabric_mode="auto",
        tee_mode="none",
        resource_json=json.dumps({"gpus": 1}),
        timeout_s=60,
    )


# ----- VAL-SCORE-001 ---------------------------------------------------------


def test_composite_is_product_of_exactly_four_factors() -> None:
    """VAL-SCORE-001: composite == correctness * efficiency * fabric_gate * tee_bonus."""

    vectors = [
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 0.42, 1.0, 1.08),
        (1.0, 2.5, 1.0, 1.20),
        (0.0, 9.0, 1.0, 1.25),
        (1.0, 0.5, 0.0, 1.20),
    ]
    for correctness, efficiency, fabric_gate, tee_bonus in vectors:
        breakdown = compute_four_factor(
            correctness=correctness,
            efficiency=efficiency,
            fabric_gate=fabric_gate,
            tee_bonus=tee_bonus,
            hyper=_hyper(),
        )
        expected = _approx_product(
            breakdown.correctness,
            breakdown.efficiency,
            breakdown.fabric_gate,
            breakdown.tee_bonus,
        )
        if breakdown.integrity_zero:
            assert breakdown.composite == pytest.approx(0.0)
        else:
            assert breakdown.composite == pytest.approx(expected, rel=1e-9, abs=1e-12)
        public = score_breakdown_to_public(breakdown)
        factor_keys = {
            "correctness",
            "efficiency",
            "fabric_gate",
            "tee_bonus",
            "composite",
        }
        assert factor_keys.issubset(public.keys())
        assert "hidden_factor" not in public
        assert "extra_multiplier" not in public


# ----- VAL-SCORE-002 ---------------------------------------------------------


def test_correctness_is_gate_in_zero_one() -> None:
    """VAL-SCORE-002: correctness residual is 0.0 or 1.0 only (v1 gate)."""

    assert coerce_gate01(1.0) == 1.0
    assert coerce_gate01(0.0) == 0.0
    # Partial credit collapses to fail (gate, not continuous).
    assert coerce_gate01(0.5) == 0.0
    assert coerce_gate01(0.99) == 0.0
    # Overshoot still full pass.
    assert coerce_gate01(1.01) == 1.0
    assert coerce_gate01(True) == 1.0
    assert coerce_gate01(False) == 0.0

    bad = compute_four_factor(
        correctness=0.0,
        efficiency=1.0,
        fabric_gate=1.0,
        tee_bonus=1.0,
    )
    good = compute_four_factor(
        correctness=1.0,
        efficiency=1.0,
        fabric_gate=1.0,
        tee_bonus=1.0,
    )
    assert bad.correctness in {0.0, 1.0}
    assert good.correctness in {0.0, 1.0}
    assert bad.correctness == 0.0
    assert good.correctness == 1.0


# ----- VAL-SCORE-003 ---------------------------------------------------------


def test_correctness_zero_zeroes_composite_regardless_of_other_factors() -> None:
    """VAL-SCORE-003: correctness=0 → composite=0 even with huge efficiency/bonus."""

    breakdown = compute_four_factor(
        correctness=0.0,
        efficiency=1e9,
        fabric_gate=1.0,
        tee_bonus=1.25,
        hyper=_hyper(),
    )
    assert breakdown.correctness == 0.0
    assert breakdown.composite == pytest.approx(0.0)
    assert breakdown.efficiency > 0  # efficiency still recorded
    assert breakdown.tee_bonus >= 1.0


# ----- VAL-SCORE-004 ---------------------------------------------------------


def test_efficiency_is_continuous_non_negative_compute_normalized() -> None:
    """VAL-SCORE-004: efficiency continuous ≥0; wall-clock delay twin doesn't dominate."""

    # Equal compute work, different artificial wall delays → same efficiency core.
    base = compute_four_factor(
        correctness=1.0,
        efficiency=0.85,  # compute-normalized allreduce figure
        fabric_gate=1.0,
        tee_bonus=1.0,
        wall_seconds=10.0,
        compute_metric=0.85,
    )
    delayed = compute_four_factor(
        correctness=1.0,
        efficiency=0.85,  # same compute metric despite longer wall
        fabric_gate=1.0,
        tee_bonus=1.0,
        wall_seconds=100.0,
        compute_metric=0.85,
    )
    assert base.efficiency == pytest.approx(delayed.efficiency)
    assert base.efficiency >= 0.0
    assert delayed.efficiency >= 0.0
    # Negative input rejected into non-negative storage.
    neg = compute_four_factor(
        correctness=1.0,
        efficiency=-3.0,
        fabric_gate=1.0,
        tee_bonus=1.0,
    )
    assert neg.efficiency >= 0.0
    assert neg.efficiency == pytest.approx(0.0)
    # Continuous: intermediate values survive (not forced to 0/1).
    mid = compute_four_factor(
        correctness=1.0,
        efficiency=0.333,
        fabric_gate=1.0,
        tee_bonus=1.0,
    )
    assert mid.efficiency == pytest.approx(0.333)
    assert mid.composite == pytest.approx(0.333)


# ----- VAL-SCORE-005 ---------------------------------------------------------


def test_fabric_gate_is_gate_in_zero_one() -> None:
    """VAL-SCORE-005: fabric_gate ∈ {0, 1}; multiplies composite."""

    ok = compute_four_factor(
        correctness=1.0,
        efficiency=2.0,
        fabric_gate=1.0,
        tee_bonus=1.0,
    )
    bad = compute_four_factor(
        correctness=1.0,
        efficiency=2.0,
        fabric_gate=0.0,
        tee_bonus=1.0,
    )
    partial = compute_four_factor(
        correctness=1.0,
        efficiency=2.0,
        fabric_gate=0.7,  # v1 hard gate → 0
        tee_bonus=1.0,
    )
    assert ok.fabric_gate == 1.0
    assert bad.fabric_gate == 0.0
    assert partial.fabric_gate == 0.0
    assert ok.composite == pytest.approx(2.0)
    assert bad.composite == pytest.approx(0.0)
    assert coerce_fabric_gate(1) == 1.0
    assert coerce_fabric_gate(0) == 0.0


# ----- VAL-SCORE-006 ---------------------------------------------------------


def test_tee_bonus_multiplier_never_below_one_for_honest_modes() -> None:
    """VAL-SCORE-006: tee_bonus ≥ 1 for honest; penalty never uses sub-1 tee_bonus."""

    none_mode = compute_four_factor(
        correctness=1.0,
        efficiency=1.0,
        fabric_gate=1.0,
        tee_bonus=1.0,
    )
    tdx = compute_four_factor(
        correctness=1.0,
        efficiency=1.0,
        fabric_gate=1.0,
        tee_bonus=1.08,
    )
    gpu = compute_four_factor(
        correctness=1.0,
        efficiency=1.0,
        fabric_gate=1.0,
        tee_bonus=1.20,
    )
    # Attempted sub-1 "penalty" path must clamp to ≥1 (penalties use integrity zero).
    attempted_penalty = compute_four_factor(
        correctness=1.0,
        efficiency=1.0,
        fabric_gate=1.0,
        tee_bonus=0.5,
    )
    assert none_mode.tee_bonus == pytest.approx(1.0)
    assert tdx.tee_bonus == pytest.approx(1.08)
    assert gpu.tee_bonus == pytest.approx(1.20)
    assert attempted_penalty.tee_bonus >= 1.0
    assert coerce_tee_bonus(0.5) >= 1.0
    assert coerce_tee_bonus(1.08) == pytest.approx(1.08)
    for row in (none_mode, tdx, gpu, attempted_penalty):
        assert row.tee_bonus >= 1.0


# ----- VAL-SCORE-007 ---------------------------------------------------------


@pytest.mark.parametrize(
    "cheat_code",
    sorted(CHEAT_REASON_CODES),
)
def test_cheat_integrity_fail_forces_composite_zero(cheat_code: str) -> None:
    """VAL-SCORE-007: integrity/cheat injects zero composite regardless of factors."""

    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=50.0,
        fabric_gate=1.0,
        tee_bonus=1.25,
        integrity_codes=[cheat_code],
    )
    assert breakdown.composite == pytest.approx(0.0)
    assert breakdown.integrity_zero is True
    # Factors remain for forensics (VAL-SCORE-026 linkage).
    assert breakdown.efficiency == pytest.approx(50.0)
    assert breakdown.tee_bonus >= 1.0


def test_integrity_zero_flag_alone_zeros_composite() -> None:
    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=3.0,
        fabric_gate=1.0,
        tee_bonus=1.1,
        integrity_zero=True,
    )
    assert breakdown.composite == pytest.approx(0.0)


# ----- VAL-SCORE-021 ---------------------------------------------------------


def test_efficiency_floor_default_zero_preserves_tiny_positive() -> None:
    """VAL-SCORE-021: default floor 0.0 keeps tiny efficiency → composite > 0 when gates 1."""

    tiny = 1e-6
    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=tiny,
        fabric_gate=1.0,
        tee_bonus=1.0,
        hyper=_hyper(efficiency_floor=0.0),
    )
    assert breakdown.efficiency == pytest.approx(tiny)
    assert breakdown.composite > 0.0
    assert breakdown.composite == pytest.approx(tiny)


def test_efficiency_floor_knocks_below_floor_from_product() -> None:
    """VAL-SCORE-021: efficiency below floor knocks product; store never negative."""

    hyper = _hyper(efficiency_floor=0.1)
    below = compute_four_factor(
        correctness=1.0,
        efficiency=0.05,
        fabric_gate=1.0,
        tee_bonus=1.0,
        hyper=hyper,
    )
    # Stored continuous metric is non-negative measured value.
    assert below.efficiency == pytest.approx(0.05)
    # Product contribution knocked to 0 via floor policy.
    assert below.composite == pytest.approx(0.0)
    assert below.below_efficiency_floor is True

    above = compute_four_factor(
        correctness=1.0,
        efficiency=0.5,
        fabric_gate=1.0,
        tee_bonus=1.0,
        hyper=hyper,
    )
    assert above.composite == pytest.approx(0.5)
    assert above.below_efficiency_floor is False

    # Direct helper: floor application never produces negatives.
    assert apply_efficiency_floor(-1.0, floor=0.0).stored == 0.0
    assert apply_efficiency_floor(0.01, floor=0.05).for_product == 0.0


def test_efficiency_never_stored_negative() -> None:
    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=-0.2,
        fabric_gate=1.0,
        tee_bonus=1.0,
        hyper=_hyper(efficiency_floor=0.0),
    )
    assert breakdown.efficiency >= 0.0


# ----- VAL-SCORE-026 ---------------------------------------------------------


def test_factor_fields_visible_when_composite_zero() -> None:
    """VAL-SCORE-026: zero composite still exposes distinct component factors."""

    breakdown = compute_four_factor(
        correctness=0.0,
        efficiency=1.5,
        fabric_gate=0.0,
        tee_bonus=1.08,
        integrity_codes=["inventory_spoof"],
    )
    assert breakdown.composite == pytest.approx(0.0)
    public = score_breakdown_to_public(breakdown)
    assert public["correctness"] == 0.0
    assert public["fabric_gate"] == 0.0
    assert public["efficiency"] == pytest.approx(1.5)
    assert public["tee_bonus"] == pytest.approx(1.08)
    assert public["composite"] == pytest.approx(0.0)
    # Forensic detail survives.
    assert breakdown.integrity_zero is True
    assert "inventory_spoof" in breakdown.reason_codes


@pytest.mark.asyncio
async def test_persisted_score_row_keeps_factors_when_composite_zero(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-SCORE-026 (+001): Score ORM row exposes factors after product zero."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'score-factor.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    app = create_app(settings, hyper_settings=_hyper())
    hotkey = "5FakeHotkeyCompositeFactorsAAA"
    attempt_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            session.add(_job(hotkey=hotkey, job_id=job_id))
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
                hotkey=hotkey,
                role="demand",
                correctness=0.0,
                efficiency=12.0,
                fabric_gate=1.0,
                proof=None,
                tee_mode="none",
                hyper=_hyper(),
                details={"integrity_codes": ["attestation_fail"]},
            )
            await session.commit()
            assert row.composite == pytest.approx(0.0)
            assert row.correctness == pytest.approx(0.0)
            assert row.efficiency == pytest.approx(12.0)
            assert row.fabric_gate == pytest.approx(1.0)
            assert row.tee_bonus >= 1.0
            public = row.to_dict()
            assert public["correctness"] == 0.0
            assert public["efficiency"] == pytest.approx(12.0)
            assert public["composite"] == 0.0
            assert public["fabric_gate"] == 1.0
            assert "tee_bonus" in public


@pytest.mark.asyncio
async def test_scores_api_exposes_factor_fields_on_zero_composite(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-SCORE-026 via GET /v1/scores/{hotkey} after a zeroed score row exists."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'score-api.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    app = create_app(settings, hyper_settings=_hyper())
    hotkey = "5ScoreFactorsVisibleHotkeyAAAAAA"
    attempt_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            session.add(_job(hotkey=hotkey, job_id=job_id))
            session.add(
                JobAttempt(
                    id=attempt_id,
                    job_id=job_id,
                    attempt_no=1,
                    status="succeeded",
                )
            )
            await session.flush()
            await persist_score_for_attempt(
                session,
                attempt_id=attempt_id,
                hotkey=hotkey,
                role="demand",
                correctness=0.0,
                efficiency=7.5,
                fabric_gate=0.0,
                proof=None,
                details={
                    "integrity_codes": ["rank_desync"],
                    "reason_codes": ["rank_desync"],
                },
                hyper=_hyper(),
            )
            await session.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get(f"/v1/scores/{hotkey}")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            items = body.get("items") or body.get("scores") or body
            assert isinstance(items, list)
            assert len(items) >= 1
            row = items[0]
            assert row["correctness"] == 0.0
            assert row["fabric_gate"] == 0.0
            assert row["efficiency"] == pytest.approx(7.5)
            assert row["tee_bonus"] >= 1.0
            assert row["composite"] == pytest.approx(0.0)


# ----- product identity regression with ScoreBreakdown dataclass -------------


def test_score_breakdown_is_frozen_and_finite() -> None:
    bd = compute_four_factor(
        correctness=1.0,
        efficiency=1.1,
        fabric_gate=1.0,
        tee_bonus=1.08,
    )
    assert isinstance(bd, ScoreBreakdown)
    assert bd.composite >= 0.0
    assert bd.composite == pytest.approx(1.1 * 1.08)


def test_efficiency_floor_setting_default_is_zero() -> None:
    assert HyperSettings().efficiency_floor == pytest.approx(0.0)
