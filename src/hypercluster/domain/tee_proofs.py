"""TEE proof attachment + verdict persistence (VAL-TEE-009/015/020).

Creates/updates ``job_proofs`` rows with verify_mode, verified flag, and
dstack_verdict_json. Non-TEE (tee=none) jobs stay on ordinary/sim path without
requiring the verifier (VAL-TEE-015).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.attest.models import TeeVerifyRequest, TeeVerifyResult
from hypercluster.attest.policy import TeeVerifyPolicy, default_policy_from_settings
from hypercluster.attest.verify import verify_tee
from hypercluster.db.models import Job, JobAttempt, JobProof, utc_now
from hypercluster.domain.job_lifecycle import get_proofs_for_attempt
from hypercluster.domain.scoring_tee import (
    TeeBonusDecision,
    compute_tee_bonus,
    persist_score_for_attempt,
)
from hypercluster.settings import HyperSettings, get_hyper_settings

SIM_PROOF_TIER = "sim"
ORDINARY_PROOF_TIER = "ordinary"
TDX_PROOF_TIER = "tdx"
TDX_GPU_PROOF_TIER = "tdx+gpu_cc"


def tier_for_tee_mode(tee_mode: str | None, *, verified_offline: bool = False) -> str:
    """Map job.tee_mode → proof_tier label."""

    mode = (tee_mode or "none").strip().lower()
    if mode in {"tdx+gpu_cc", "tdx_gpu_cc", "tdx+gpu-cc"}:
        return TDX_GPU_PROOF_TIER
    if mode == "tdx":
        return TDX_PROOF_TIER
    if mode == "sim":
        return SIM_PROOF_TIER
    return ORDINARY_PROOF_TIER if mode in {"none", ""} else mode


def build_sim_proof(
    *,
    attempt_id: str,
    job: Job,
    integrity_fail: bool = False,
    fabric_gate: float = 1.0,
    composite_hint: float | None = None,
) -> JobProof:
    """Construct a sim-tier proof that never carries live tee_bonus eligibility.

    VAL-TEE-005 / VAL-TEE-015: tee=none and pure sim wait point.
    """

    tee_mode = (job.tee_mode or "none").strip().lower()
    # Non-TEE → ordinary; TEE job without offline fixture → still sim tier
    # so bonus refuses live constants until offline verification upgrades.
    if tee_mode in {"none", ""}:
        proof_tier = ORDINARY_PROOF_TIER
    elif tee_mode in {"tdx", "tdx+gpu_cc", "tdx_gpu_cc"}:
        # Sealed under sim never upgrades to verified tdx without offline path.
        proof_tier = SIM_PROOF_TIER
    else:
        proof_tier = SIM_PROOF_TIER

    payload = {
        "sim": True,
        "tier": proof_tier,
        "tee_mode": tee_mode,
        "fabric_gate": fabric_gate,
        "integrity_fail": integrity_fail,
    }
    if composite_hint is not None:
        payload["composite"] = composite_hint
    verdict = {
        "is_valid": False if integrity_fail else True,
        "quote_verified": False,
        "verify_mode": "sim",
        "reason_codes": ["sim_path"] + (["integrity_fail"] if integrity_fail else []),
        "sim": True,
    }
    return JobProof(
        id=str(uuid.uuid4()),
        attempt_id=attempt_id,
        proof_tier=proof_tier,
        payload_json=json.dumps(payload, sort_keys=True),
        tdx_quote_b64=None,
        gpu_evidence_json=None,
        dstack_verdict_json=json.dumps(verdict, sort_keys=True),
        verified=0 if integrity_fail else 1,
        verify_mode="sim",
        created_at=utc_now(),
    )


def attach_offline_proof(
    *,
    attempt_id: str,
    job: Job,
    result: TeeVerifyResult,
    quote_b64: str | None = None,
    gpu_evidence: dict[str, Any] | None = None,
    proof_tier: str | None = None,
    nonce: str | None = None,
) -> JobProof:
    """Persist offline_fixture (or Live) verify result onto job_proofs."""

    tier = proof_tier or tier_for_tee_mode(job.tee_mode)
    if result.is_valid and tier in {ORDINARY_PROOF_TIER, SIM_PROOF_TIER}:
        # Promote to job tee tier when offline verify validated.
        tier = tier_for_tee_mode(job.tee_mode, verified_offline=True)

    payload = {
        "sim": False,
        "tier": tier,
        "tee_mode": job.tee_mode,
        "nonce": nonce,
        "mode": result.verify_mode or "offline_fixture",
    }
    verdict = result.to_public()
    return JobProof(
        id=str(uuid.uuid4()),
        attempt_id=attempt_id,
        proof_tier=tier,
        payload_json=json.dumps(payload, sort_keys=True),
        tdx_quote_b64=quote_b64,
        gpu_evidence_json=(
            json.dumps(gpu_evidence, sort_keys=True) if gpu_evidence is not None else None
        ),
        dstack_verdict_json=json.dumps(verdict, sort_keys=True),
        verified=1 if result.is_valid else 0,
        verify_mode=result.verify_mode or "offline_fixture",
        created_at=utc_now(),
    )


def verify_and_build_proof(
    *,
    attempt_id: str,
    job: Job,
    quote_b64: str,
    report_data_expected: bytes,
    gpu_evidence: dict[str, Any] | None = None,
    mode: str = "offline_fixture",
    event_log: str | None = None,
    vm_config: dict[str, Any] | None = None,
    policy: TeeVerifyPolicy | None = None,
    require_gpu_evidence: bool | None = None,
    expected_gpu_nonce: str | None = None,
) -> tuple[JobProof, TeeVerifyResult]:
    """Run verify_tee and build a JobProof with verdict persistence.

    ``mode=offline_fixture`` never dials the network (VAL-TEE-019).
    """

    tee_mode = (job.tee_mode or "none").strip().lower()
    needs_gpu = require_gpu_evidence
    if needs_gpu is None:
        needs_gpu = tee_mode in {"tdx+gpu_cc", "tdx_gpu_cc"}

    req = TeeVerifyRequest(
        quote_b64=quote_b64,
        event_log=event_log,
        vm_config=vm_config,
        report_data_expected=report_data_expected,
        gpu_evidence=gpu_evidence,
        mode=mode,  # type: ignore[arg-type]
    )
    result = verify_tee(
        req,
        policy=policy if policy is not None else default_policy_from_settings(),
        require_gpu_evidence=bool(needs_gpu),
        expected_gpu_nonce=expected_gpu_nonce,
        httpx_client=object(),  # deliberate: should be discarded for offline
    )
    proof = attach_offline_proof(
        attempt_id=attempt_id,
        job=job,
        result=result,
        quote_b64=quote_b64,
        gpu_evidence=gpu_evidence,
        proof_tier=tier_for_tee_mode(job.tee_mode),
        nonce=expected_gpu_nonce,
    )
    return proof, result


async def ensure_attempt_proof(
    session: AsyncSession,
    *,
    job: Job,
    attempt: JobAttempt,
    integrity_fail: bool = False,
    fabric_gate: float = 1.0,
) -> JobProof:
    """Return existing proof or create sim/ordinary proof (VAL-TEE-015)."""

    existing = await get_proofs_for_attempt(session, attempt.id)
    if existing:
        return existing[0]
    proof = build_sim_proof(
        attempt_id=attempt.id,
        job=job,
        integrity_fail=integrity_fail,
        fabric_gate=fabric_gate,
    )
    session.add(proof)
    await session.flush()
    return proof


async def score_attempt_with_tee(
    session: AsyncSession,
    *,
    job: Job,
    attempt: JobAttempt,
    correctness: float = 1.0,
    efficiency: float = 1.0,
    fabric_gate: float = 1.0,
    hyper: HyperSettings | None = None,
    integrity_fail: bool = False,
    details: dict[str, Any] | None = None,
) -> tuple[Any, TeeBonusDecision]:
    """Persist score row for attempt using proof-driven tee_bonus rules."""

    proof = await ensure_attempt_proof(
        session,
        job=job,
        attempt=attempt,
        integrity_fail=integrity_fail,
        fabric_gate=fabric_gate,
    )
    settings = hyper if hyper is not None else get_hyper_settings()
    is_valid_verdict: bool | None = None
    if proof.dstack_verdict_json:
        try:
            parsed = json.loads(proof.dstack_verdict_json)
            if isinstance(parsed, dict) and "is_valid" in parsed:
                is_valid_verdict = bool(parsed.get("is_valid"))
        except (TypeError, ValueError):
            is_valid_verdict = False

    decision = compute_tee_bonus(
        proof_tier=proof.proof_tier,
        verified=bool(proof.verified),
        verify_mode=proof.verify_mode,
        tee_mode=job.tee_mode or "none",
        # Sim seal of a claimed TEE job is not attestation_fail by itself —
        # bonus simply stays 1.0 (VAL-TEE-005). Hard zero only on integrity_fail
        # or garbage-quote path (unverified claim capillary).
        attestation_fail=integrity_fail,
        hyper=settings,
        is_valid_verdict=is_valid_verdict,
    )
    score = await persist_score_for_attempt(
        session,
        attempt_id=attempt.id,
        hotkey=job.submitter_hotkey,
        role="demand",
        correctness=correctness if not integrity_fail else 0.0,
        efficiency=efficiency,
        fabric_gate=fabric_gate,
        proof=proof,
        tee_mode=job.tee_mode or "none",
        hyper=settings,
        details=details,
    )
    return score, decision


__all__ = [
    "ORDINARY_PROOF_TIER",
    "SIM_PROOF_TIER",
    "TDX_GPU_PROOF_TIER",
    "TDX_PROOF_TIER",
    "attach_offline_proof",
    "build_sim_proof",
    "ensure_attempt_proof",
    "score_attempt_with_tee",
    "tier_for_tee_mode",
    "verify_and_build_proof",
]
