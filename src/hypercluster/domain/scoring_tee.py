"""TEE bonus application for the four-factor score product (VAL-TEE-005..008,020).

``composite = correctness × efficiency × fabric_gate × tee_bonus``

This module owns **only** the tee_bonus multiplier rules and attempt score row
construction. Full weight aggregation arrives in M6; here we pin firm rules:

* ``proof_tier=sim`` never receives a live TEE bonus (always 1.0).
* Bonus > 1.0 only when ``verified`` is true and mode is offline_fixture/live.
* Unverified TEE claims stay at 1.0; hard attestation fails may zero composite.
* TDX vs tdx+gpu_cc pick configured HYPER_TEE_BONUS_* constants.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import JobProof, Score, utc_now
from hypercluster.settings import HyperSettings, get_hyper_settings

ProofTier = Literal["ordinary", "sim", "tdx", "tdx_gpu_cc", "tdx+gpu_cc"]
VerifyMode = Literal["sim", "offline_fixture", "live"]

# Normalization map so callers can use either underscore or plus form.
_TIER_GPU = frozenset({"tdx_gpu_cc", "tdx+gpu_cc", "tdx+gpu-cc"})
_TIER_TDX = frozenset({"tdx"})
_TIER_SIM = frozenset({"sim"})
_TIER_ORDINARY = frozenset({"ordinary", "none", ""})

_MODE_ELIGIBLE = frozenset({"offline_fixture", "live"})


@dataclass(frozen=True)
class TeeBonusDecision:
    """Result of tee_bonus policy application."""

    tee_bonus: float
    applied_tier: str
    reason_codes: list[str] = field(default_factory=list)
    integrity_zero: bool = False

    @property
    def bonus_applied(self) -> bool:
        return self.tee_bonus > 1.0 + 1e-12


def _normalize_tier(proof_tier: str | None, *, tee_mode: str | None = None) -> str:
    raw = (proof_tier or "").strip().lower()
    if not raw or raw in _TIER_ORDINARY:
        # Fall back to job tee_mode claim when proof tier is ordinary/empty.
        claim = (tee_mode or "none").strip().lower()
        if claim in _TIER_GPU:
            return "tdx+gpu_cc"
        if claim in _TIER_TDX:
            return "tdx"
        if claim in _TIER_SIM:
            return "sim"
        return "ordinary"
    if raw in _TIER_GPU:
        return "tdx+gpu_cc"
    if raw in _TIER_TDX:
        return "tdx"
    if raw in _TIER_SIM:
        return "sim"
    return raw


def compute_tee_bonus(
    *,
    proof_tier: str,
    verified: bool,
    verify_mode: str = "sim",
    tee_mode: str = "none",
    attestation_fail: bool = False,
    hyper: HyperSettings | None = None,
    is_valid_verdict: bool | None = None,
) -> TeeBonusDecision:
    """Compute tee_bonus multiplier from proof + verdict + settings.

    Invariants:
    - sim proof_tier → always 1.0 (VAL-TEE-005), even if verified flag set.
    - verified=False or is_valid_verdict=False → 1.0 (VAL-TEE-008/020).
    - offline/live verified tdx → HYPER_TEE_BONUS_TDX (VAL-TEE-006).
    - offline/live verified tdx+gpu_cc → HYPER_TEE_BONUS_TDX_GPU (VAL-TEE-007).
    - attestation_fail → integrity_zero True (preferred composite 0).
    """

    settings = hyper if hyper is not None else get_hyper_settings()
    tier = _normalize_tier(proof_tier, tee_mode=tee_mode)
    mode = (verify_mode or "sim").strip().lower()
    reasons: list[str] = []

    if attestation_fail:
        reasons.append("attestation_fail")
        return TeeBonusDecision(
            tee_bonus=1.0,
            applied_tier=tier,
            reason_codes=reasons,
            integrity_zero=True,
        )

    # Explicit invalid verdict beats a caller-supplied verified flag.
    effective_verified = bool(verified)
    if is_valid_verdict is not None:
        effective_verified = effective_verified and bool(is_valid_verdict)
    if not effective_verified and is_valid_verdict is False:
        reasons.append("verdict_invalid")

    if tier == "sim":
        reasons.append("sim_no_live_bonus")
        return TeeBonusDecision(
            tee_bonus=1.0,
            applied_tier="sim",
            reason_codes=reasons,
            integrity_zero=False,
        )

    if tier == "ordinary":
        reasons.append("ordinary_proof")
        return TeeBonusDecision(
            tee_bonus=1.0,
            applied_tier="ordinary",
            reason_codes=reasons,
        )

    # Claimed TEE tier without verification → no bonus (VAL-TEE-008).
    if not effective_verified:
        reasons.append("unverified_tee_claim")
        reasons.append("no_bonus")
        # Claimed TEE without valid proof: integrity soft-zero preference.
        hard_zero = (tee_mode or "").strip().lower() in {
            "tdx",
            "tdx+gpu_cc",
            "tdx_gpu_cc",
        }
        if hard_zero:
            reasons.append("attestation_fail")
        return TeeBonusDecision(
            tee_bonus=1.0,
            applied_tier=tier,
            reason_codes=reasons,
            integrity_zero=hard_zero,
        )

    if mode not in _MODE_ELIGIBLE:
        # verified + sim mode should not grant live constants.
        reasons.append("verify_mode_not_live_or_offline")
        reasons.append("no_bonus")
        return TeeBonusDecision(
            tee_bonus=1.0,
            applied_tier=tier,
            reason_codes=reasons,
        )

    if tier == "tdx":
        bonus = float(settings.tee_bonus_tdx)
        reasons.append("bonus_tdx")
        return TeeBonusDecision(
            tee_bonus=bonus,
            applied_tier="tdx",
            reason_codes=reasons,
        )

    if tier == "tdx+gpu_cc":
        bonus = float(settings.tee_bonus_tdx_gpu)
        reasons.append("bonus_tdx_gpu_cc")
        return TeeBonusDecision(
            tee_bonus=bonus,
            applied_tier="tdx+gpu_cc",
            reason_codes=reasons,
        )

    reasons.append("unknown_tier_no_bonus")
    return TeeBonusDecision(
        tee_bonus=1.0,
        applied_tier=tier,
        reason_codes=reasons,
    )


def four_factor_composite(
    *,
    correctness: float,
    efficiency: float,
    fabric_gate: float,
    tee_bonus: float,
    integrity_zero: bool = False,
) -> float:
    """Product of the four factors; integrity fails force 0."""

    if integrity_zero:
        return 0.0
    return float(correctness) * float(efficiency) * float(fabric_gate) * float(tee_bonus)


def decision_from_proof(
    proof: JobProof,
    *,
    tee_mode: str = "none",
    hyper: HyperSettings | None = None,
) -> TeeBonusDecision:
    """Derive tee_bonus decision from a persisted JobProof row."""

    is_valid: bool | None = None
    attestation_fail = False
    if proof.dstack_verdict_json:
        try:
            verdict = json.loads(proof.dstack_verdict_json)
            if isinstance(verdict, dict):
                if "is_valid" in verdict:
                    is_valid = bool(verdict.get("is_valid"))
                codes = verdict.get("reason_codes") or []
                if isinstance(codes, list) and any(
                    c in {"attestation_fail", "quote_invalid", "quote_sig_invalid"}
                    for c in codes
                ):
                    # Hard fail when garbage + claim path.
                    if is_valid is False and (tee_mode or "none") != "none":
                        attestation_fail = True
        except (TypeError, ValueError):
            is_valid = False

    return compute_tee_bonus(
        proof_tier=proof.proof_tier,
        verified=bool(proof.verified),
        verify_mode=proof.verify_mode,
        tee_mode=tee_mode,
        attestation_fail=attestation_fail,
        hyper=hyper,
        is_valid_verdict=is_valid,
    )


async def persist_score_for_attempt(
    session: AsyncSession,
    *,
    attempt_id: str,
    hotkey: str,
    role: str = "demand",
    correctness: float,
    efficiency: float,
    fabric_gate: float,
    proof: JobProof | None,
    tee_mode: str = "none",
    hyper: HyperSettings | None = None,
    details: dict[str, Any] | None = None,
) -> Score:
    """Insert or update a scores row for an attempt (VAL-TEE-005..008,020).

    ``tee_bonus > 1.0`` only when proof.verified and eligible tier/mode.
    """

    decision = (
        decision_from_proof(proof, tee_mode=tee_mode, hyper=hyper)
        if proof is not None
        else compute_tee_bonus(
            proof_tier="sim",
            verified=False,
            verify_mode="sim",
            tee_mode=tee_mode,
            hyper=hyper,
        )
    )
    bonus = decision.tee_bonus
    # Hard invariant: never bonus without verified flag on the proof.
    if proof is not None and not bool(proof.verified) and bonus > 1.0:
        bonus = 1.0
        decision = TeeBonusDecision(
            tee_bonus=1.0,
            applied_tier=decision.applied_tier,
            reason_codes=list(decision.reason_codes) + ["verified_flag_required"],
            integrity_zero=decision.integrity_zero,
        )

    composite = four_factor_composite(
        correctness=correctness,
        efficiency=efficiency,
        fabric_gate=fabric_gate,
        tee_bonus=bonus,
        integrity_zero=decision.integrity_zero,
    )

    detail_blob: dict[str, Any] = {
        "tee_decision": {
            "tee_bonus": bonus,
            "applied_tier": decision.applied_tier,
            "reason_codes": list(decision.reason_codes),
            "integrity_zero": decision.integrity_zero,
        },
        "proof": (
            {
                "id": proof.id,
                "proof_tier": proof.proof_tier,
                "verified": bool(proof.verified),
                "verify_mode": proof.verify_mode,
            }
            if proof is not None
            else None
        ),
    }
    if details:
        detail_blob["extra"] = details

    existing = await session.execute(select(Score).where(Score.attempt_id == attempt_id))
    row = existing.scalar_one_or_none()
    if row is None:
        row = Score(
            id=str(uuid.uuid4()),
            attempt_id=attempt_id,
            hotkey=hotkey,
            role=role,
            correctness=float(correctness),
            efficiency=float(efficiency),
            fabric_gate=float(fabric_gate),
            tee_bonus=float(bonus),
            composite=float(composite),
            details_json=json.dumps(detail_blob, sort_keys=True),
            created_at=utc_now(),
        )
        session.add(row)
    else:
        row.correctness = float(correctness)
        row.efficiency = float(efficiency)
        row.fabric_gate = float(fabric_gate)
        row.tee_bonus = float(bonus)
        row.composite = float(composite)
        row.details_json = json.dumps(detail_blob, sort_keys=True)
        row.hotkey = hotkey
        row.role = role
    await session.flush()
    return row


async def get_score_for_attempt(session: AsyncSession, attempt_id: str) -> Score | None:
    result = await session.execute(select(Score).where(Score.attempt_id == attempt_id))
    return result.scalar_one_or_none()


__all__ = [
    "TeeBonusDecision",
    "compute_tee_bonus",
    "decision_from_proof",
    "four_factor_composite",
    "get_score_for_attempt",
    "persist_score_for_attempt",
]
