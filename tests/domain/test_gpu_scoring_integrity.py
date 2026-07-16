"""VAL-GPU-050/051/052: GPU probe integrity maps into four-factor zeros.

Fixed formula remains::

    composite = correctness × efficiency × fabric_gate × tee_bonus

No 5th factor. Spoof / fatal probe / HYPER_SIM_GPU_PROBE_FAIL force composite 0
via existing integrity / correctness / fabric_gate paths.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from hypercluster.db.models import Job, JobAttempt
from hypercluster.domain.gpu_scoring_integrity import (
    GpuIntegrityDecision,
    apply_gpu_integrity,
    evaluate_claim_vs_evidence,
    evaluate_gpu_probe_integrity,
    sim_gpu_probe_fail_active,
)
from hypercluster.domain.scoring import (
    compute_four_factor,
    score_breakdown_to_public,
)
from hypercluster.domain.scoring_tee import persist_score_for_attempt
from hypercluster.settings import HyperSettings


def _hyper(**overrides: Any) -> HyperSettings:
    base: dict[str, Any] = {
        "tee_bonus_tdx": 1.08,
        "tee_bonus_tdx_gpu": 1.20,
        "efficiency_floor": 0.0,
        "score_window_attempts": 50,
        "sim_gpu_probe_fail": False,
        "require_gpu_evidence_for_live": False,
    }
    base.update(overrides)
    return HyperSettings(**base)


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


# ----- VAL-GPU-050 -----------------------------------------------------------


def test_claim_model_mismatch_vs_passed_evidence_is_integrity_zero() -> None:
    """VAL-GPU-050: claimed H100 vs measured V100 evidence → inventory_spoof zero."""

    decision = evaluate_claim_vs_evidence(
        claimed_gpu_model="H100",
        claimed_gpu_count=1,
        evidence_status="passed",
        measured_gpu_model="Tesla V100-SXM2-16GB",
        measured_gpu_count=1,
    )
    assert decision.integrity_zero is True
    assert "inventory_spoof" in decision.integrity_codes
    assert decision.correctness == 0.0
    assert decision.fabric_gate == 0.0

    c, e, f, t = apply_gpu_integrity(
        correctness=1.0,
        efficiency=2.5,
        fabric_gate=1.0,
        tee_bonus=1.0,
        decision=decision,
        hyper=_hyper(),
    )
    breakdown = compute_four_factor(
        correctness=c,
        efficiency=e,
        fabric_gate=f,
        tee_bonus=t,
        integrity_zero=decision.integrity_zero,
        integrity_codes=decision.integrity_codes,
        hyper=_hyper(),
    )
    assert breakdown.composite == pytest.approx(0.0)
    public = score_breakdown_to_public(breakdown)
    # Exactly four factors — no gpu_gate key (VAL-GPU-050 formula fence).
    assert set(public.keys()) >= {
        "correctness",
        "efficiency",
        "fabric_gate",
        "tee_bonus",
        "composite",
    }
    assert "gpu_gate" not in public
    assert public["composite"] == pytest.approx(0.0)
    # Product identity still holds for factors still published.
    assert public["composite"] == pytest.approx(
        public["correctness"] * public["efficiency"] * public["fabric_gate"] * public["tee_bonus"]
    )


def test_claim_count_mismatch_inventory_spoof() -> None:
    decision = evaluate_claim_vs_evidence(
        claimed_gpu_model="1V100.6V",
        claimed_gpu_count=8,
        evidence_status="passed",
        measured_gpu_model="Tesla V100-SXM2-16GB",
        measured_gpu_count=1,
    )
    assert decision.integrity_zero is True
    assert "inventory_spoof" in decision.integrity_codes
    assert decision.fabric_gate == 0.0


def test_failed_evidence_on_live_required_path_zeros_composite() -> None:
    decision = evaluate_gpu_probe_integrity(
        claimed_gpu_model="1V100.6V",
        claimed_gpu_count=1,
        evidence_status="failed",
        measured_gpu_model=None,
        measured_gpu_count=0,
        proof_tier="ordinary",
        requires_live_gpu_evidence=True,
        hyper=_hyper(),
    )
    assert decision.integrity_zero is True
    codes = set(decision.integrity_codes)
    assert codes & {"inventory_spoof", "gpu_probe_fail", "integrity_fail"}
    breakdown = compute_four_factor(
        correctness=decision.correctness if decision.correctness is not None else 1.0,
        efficiency=1.0,
        fabric_gate=decision.fabric_gate if decision.fabric_gate is not None else 1.0,
        tee_bonus=1.0,
        integrity_zero=decision.integrity_zero,
        integrity_codes=decision.integrity_codes,
        hyper=_hyper(),
    )
    assert breakdown.composite == pytest.approx(0.0)


def test_matching_claim_and_passed_evidence_keeps_positive_product() -> None:
    decision = evaluate_claim_vs_evidence(
        claimed_gpu_model="1V100.6V",
        claimed_gpu_count=1,
        evidence_status="passed",
        measured_gpu_model="Tesla V100-SXM2-16GB",
        measured_gpu_count=1,
    )
    assert decision.integrity_zero is False
    assert decision.integrity_codes == []
    c, e, f, t = apply_gpu_integrity(
        correctness=1.0,
        efficiency=0.42,
        fabric_gate=1.0,
        tee_bonus=1.08,
        decision=decision,
        hyper=_hyper(),
    )
    breakdown = compute_four_factor(
        correctness=c,
        efficiency=e,
        fabric_gate=f,
        tee_bonus=t,
        integrity_zero=False,
        hyper=_hyper(),
    )
    assert breakdown.composite == pytest.approx(0.42 * 1.08)
    assert breakdown.composite > 0.0


# ----- VAL-GPU-051 -----------------------------------------------------------


def test_unprobed_sim_path_unaffected_without_evidence_requirement() -> None:
    """VAL-GPU-051: proof_tier=sim / no silicon evidence still scores green."""

    decision = evaluate_gpu_probe_integrity(
        claimed_gpu_model="1V100.6V",
        claimed_gpu_count=1,
        evidence_status=None,
        measured_gpu_model=None,
        measured_gpu_count=None,
        proof_tier="sim",
        requires_live_gpu_evidence=False,
        hyper=_hyper(require_gpu_evidence_for_live=False, sim_gpu_probe_fail=False),
    )
    assert decision.integrity_zero is False
    assert decision.integrity_codes == []

    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=1.5,
        fabric_gate=1.0,
        tee_bonus=1.0,
        integrity_zero=decision.integrity_zero,
        integrity_codes=decision.integrity_codes or None,
        hyper=_hyper(),
    )
    assert breakdown.composite == pytest.approx(1.5)
    assert breakdown.integrity_zero is False


def test_sim_launcher_path_no_forced_probe() -> None:
    """VAL-GPU-051: sim_launcher / lack of evidence does not force zero."""

    decision = evaluate_gpu_probe_integrity(
        claimed_gpu_model=None,
        claimed_gpu_count=None,
        evidence_status=None,
        measured_gpu_model=None,
        measured_gpu_count=None,
        proof_tier="sim",
        execution_backend="sim_launcher",
        requires_live_gpu_evidence=False,
        hyper=_hyper(),
    )
    assert decision.integrity_zero is False
    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=2.0,
        fabric_gate=1.0,
        tee_bonus=1.0,
        hyper=_hyper(),
    )
    assert breakdown.composite == pytest.approx(2.0)


# ----- VAL-GPU-052 -----------------------------------------------------------


def test_hyper_sim_gpu_probe_fail_injects_integrity_zero() -> None:
    """VAL-GPU-052: HYPER_SIM_GPU_PROBE_FAIL → composite 0 without real SSH."""

    hyper = _hyper(sim_gpu_probe_fail=True)
    assert sim_gpu_probe_fail_active(hyper) is True

    decision = evaluate_gpu_probe_integrity(
        claimed_gpu_model="1V100.6V",
        claimed_gpu_count=1,
        evidence_status=None,
        measured_gpu_model=None,
        measured_gpu_count=None,
        proof_tier="sim",
        requires_live_gpu_evidence=False,
        hyper=hyper,
    )
    assert decision.integrity_zero is True
    assert "gpu_probe_fail" in decision.integrity_codes or "inventory_spoof" in (
        decision.integrity_codes
    )

    c, e, f, t = apply_gpu_integrity(
        correctness=1.0,
        efficiency=3.0,
        fabric_gate=1.0,
        tee_bonus=1.2,
        decision=decision,
        hyper=hyper,
    )
    breakdown = compute_four_factor(
        correctness=c,
        efficiency=e,
        fabric_gate=f,
        tee_bonus=t,
        integrity_zero=decision.integrity_zero,
        integrity_codes=decision.integrity_codes,
        hyper=hyper,
    )
    assert breakdown.composite == pytest.approx(0.0)
    # Factors still published for forensics; tee/eff residuals survive.
    assert breakdown.efficiency == pytest.approx(3.0)
    assert breakdown.tee_bonus >= 1.0
    assert "gpu_gate" not in score_breakdown_to_public(breakdown)


def test_sim_gpu_probe_fail_flag_defaults_false() -> None:
    hyper = HyperSettings()
    assert hyper.sim_gpu_probe_fail is False
    assert sim_gpu_probe_fail_active(hyper) is False


@pytest.mark.asyncio
async def test_persist_score_with_sim_gpu_probe_fail_details(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-GPU-052: integrity codes from GPU inject persist as composite 0."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'gpu-score.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    hyper = _hyper(sim_gpu_probe_fail=True)
    decision = evaluate_gpu_probe_integrity(hyper=hyper)
    app = create_app(settings, hyper_settings=hyper)
    hotkey = "5FAKEGPUINJECTHOTKEY0000000000000000000000000001"
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
            c, e, f, t = apply_gpu_integrity(
                correctness=1.0,
                efficiency=5.0,
                fabric_gate=1.0,
                tee_bonus=1.0,
                decision=decision,
                hyper=hyper,
            )
            row = await persist_score_for_attempt(
                session,
                attempt_id=attempt_id,
                hotkey=hotkey,
                role="demand",
                correctness=c,
                efficiency=e,
                fabric_gate=f if f is not None else 1.0,
                proof=None,
                tee_mode="none",
                hyper=hyper,
                details={
                    "integrity_codes": decision.integrity_codes,
                    "gpu_integrity": True,
                },
            )
            await session.commit()
            assert row.composite == pytest.approx(0.0)
            # Gate residues typically zero; integrity path always forces composite 0.
            assert float(row.correctness) == 0.0 or float(row.fabric_gate) == 0.0
            details = json.loads(row.details_json or "{}")
            tee = details.get("tee_decision") or {}
            codes = set(tee.get("reason_codes") or [])
            extra = details.get("extra") or {}
            codes |= set(extra.get("integrity_codes") or [])
            assert codes & {
                "gpu_probe_fail",
                "inventory_spoof",
                "integrity_fail",
            }


def test_four_factor_formula_signature_unchanged_with_gpu_codes() -> None:
    """Prior M6 math: spoof code still product identity, no 5th factor."""

    breakdown = compute_four_factor(
        correctness=1.0,
        efficiency=2.0,
        fabric_gate=1.0,
        tee_bonus=1.1,
        integrity_codes=["inventory_spoof", "gpu_probe_fail"],
        hyper=_hyper(),
    )
    assert breakdown.composite == pytest.approx(0.0)
    public = score_breakdown_to_public(breakdown)
    assert "gpu_gate" not in public
    assert (
        len([k for k in public if k in {"correctness", "efficiency", "fabric_gate", "tee_bonus"}])
        == 4
    )


def test_gpu_integrity_decision_dataclass_fields() -> None:
    d = GpuIntegrityDecision(integrity_zero=False, integrity_codes=[], reason="clean")
    assert d.correctness is None
    assert d.fabric_gate is None


@pytest.mark.asyncio
async def test_score_attempt_with_tee_applies_claim_spoof(
    settings_factory: Any, tmp_path: Path
) -> None:
    """VAL-GPU-050: score_attempt_with_tee zeros composite on claim vs evidence."""

    from hypercluster.app import create_app
    from hypercluster.domain.tee_proofs import score_attempt_with_tee

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'gpu-spoof.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    hyper = _hyper()
    app = create_app(settings, hyper_settings=hyper)
    hotkey = "5FAKEGPUSPOOFHOTKEY00000000000000000000000000002"
    attempt_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            job = _job(hotkey=hotkey, job_id=job_id)
            session.add(job)
            attempt = JobAttempt(
                id=attempt_id,
                job_id=job_id,
                attempt_no=1,
                status="succeeded",
            )
            session.add(attempt)
            await session.flush()
            score, _decision = await score_attempt_with_tee(
                session,
                job=job,
                attempt=attempt,
                correctness=1.0,
                efficiency=4.0,
                fabric_gate=1.0,
                hyper=hyper,
                integrity_fail=False,
                details={
                    "claimed_gpu_model": "H100",
                    "claimed_gpu_count": 1,
                    "evidence_status": "passed",
                    "measured_gpu_model": "Tesla V100-SXM2-16GB",
                    "measured_gpu_count": 1,
                    "proof_tier": "ordinary",
                    "execution_backend": "real_ssh",
                },
            )
            await session.commit()
            assert score.composite == pytest.approx(0.0)
            assert float(score.correctness) == 0.0 or float(score.fabric_gate) == 0.0
            details = json.loads(score.details_json or "{}")
            extra = details.get("extra") or {}
            codes = set(extra.get("integrity_codes") or [])
            codes |= set((details.get("tee_decision") or {}).get("reason_codes") or [])
            assert "inventory_spoof" in codes or "gpu_probe_mismatch" in codes
            # Formula fence: no gpu_gate published factor.
            factors = details.get("factors") or {}
            assert "gpu_gate" not in factors
            assert set(factors.keys()) >= {
                "correctness",
                "efficiency",
                "fabric_gate",
                "tee_bonus",
                "composite",
            }


@pytest.mark.asyncio
async def test_score_attempt_with_tee_sim_unaffected(settings_factory: Any, tmp_path: Path) -> None:
    """VAL-GPU-051: pure sim score path stays positive without GPU evidence."""

    from hypercluster.app import create_app
    from hypercluster.domain.tee_proofs import score_attempt_with_tee

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'gpu-sim.sqlite3'}",
        shared_token="test-challenge-shared-token",
        shared_token_file=None,
    )
    hyper = _hyper(sim_gpu_probe_fail=False, require_gpu_evidence_for_live=False)
    app = create_app(settings, hyper_settings=hyper)
    hotkey = "5FAKEGPUSIMHOTKEY000000000000000000000000000003"
    attempt_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            job = _job(hotkey=hotkey, job_id=job_id)
            session.add(job)
            attempt = JobAttempt(
                id=attempt_id,
                job_id=job_id,
                attempt_no=1,
                status="succeeded",
            )
            session.add(attempt)
            await session.flush()
            score, _decision = await score_attempt_with_tee(
                session,
                job=job,
                attempt=attempt,
                correctness=1.0,
                efficiency=2.5,
                fabric_gate=1.0,
                hyper=hyper,
                integrity_fail=False,
                details={"proof_tier": "sim", "execution_backend": "sim_launcher"},
            )
            await session.commit()
            assert score.composite == pytest.approx(2.5)
            assert score.composite > 0.0
