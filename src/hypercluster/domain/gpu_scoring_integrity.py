"""Map GPU probe / inventory honesty into existing four-factor integrity zeros.

M9 design §8 / VAL-GPU-050..052 — **no 5th formula factor**::

    composite = correctness × efficiency × fabric_gate × tee_bonus

Probe failures and claim-vs-evidence mismatches reason into the same
integrity-zero family as ``inventory_spoof`` / ``integrity_fail`` (see
``library/scoring.md`` and ``CHEAT_REASON_CODES``).

Pure sim / ``proof_tier=sim`` paths without ``HYPER_REQUIRE_GPU_EVIDENCE_FOR_LIVE``
or ``HYPER_SIM_GPU_PROBE_FAIL`` remain unaffected (VAL-GPU-051).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from hypercluster.probe.model_table import models_match, normalize_gpu_model
from hypercluster.settings import HyperSettings, get_hyper_settings

# Codes attached to GPU probe honesty paths (subset of CHEAT_REASON_CODES family).
GPU_PROBE_FAIL_CODE = "gpu_probe_fail"
GPU_PROBE_MISMATCH_CODE = "gpu_probe_mismatch"
INVENTORY_SPOOF_CODE = "inventory_spoof"
INTEGRITY_FAIL_CODE = "integrity_fail"

EvidenceStatus = Literal["passed", "failed", "error"] | str | None


@dataclass(frozen=True, slots=True)
class GpuIntegrityDecision:
    """Outcome of GPU probe honesty evaluation for one attempt/score."""

    integrity_zero: bool
    integrity_codes: list[str] = field(default_factory=list)
    correctness: float | None = None  # when set, overrides caller's gate
    fabric_gate: float | None = None  # when set, overrides caller's gate
    reason: str = ""


def sim_gpu_probe_fail_active(hyper: HyperSettings | None = None) -> bool:
    """True when HYPER_SIM_GPU_PROBE_FAIL inject is on (VAL-GPU-052)."""

    settings = hyper if hyper is not None else get_hyper_settings()
    return bool(getattr(settings, "sim_gpu_probe_fail", False))


def require_gpu_evidence_for_live(hyper: HyperSettings | None = None) -> bool:
    settings = hyper if hyper is not None else get_hyper_settings()
    return bool(getattr(settings, "require_gpu_evidence_for_live", False))


def evaluate_claim_vs_evidence(
    *,
    claimed_gpu_model: str | None,
    claimed_gpu_count: int | None,
    evidence_status: EvidenceStatus,
    measured_gpu_model: str | None,
    measured_gpu_count: int | None,
) -> GpuIntegrityDecision:
    """Compare claimed node inventory against last good / measured evidence.

    On mismatch or non-passed evidence with a concrete claim, force integrity
    zero via inventory_spoof (+ fabric_gate/correctness 0) — VAL-GPU-050.
    """

    status = (str(evidence_status).strip().lower() if evidence_status else None) or None
    codes: list[str] = []
    reason_parts: list[str] = []

    # Passed evidence is required to accept claimed silicon as honest when we
    # actually have measured fields or an explicit status.
    if status in {"failed", "error"}:
        codes.extend([GPU_PROBE_FAIL_CODE, INVENTORY_SPOOF_CODE])
        reason_parts.append(f"evidence_status={status}")
        return GpuIntegrityDecision(
            integrity_zero=True,
            integrity_codes=_uniq(codes),
            correctness=0.0,
            fabric_gate=0.0,
            reason=";".join(reason_parts) or "gpu_evidence_failed",
        )

    if status is None:
        # No evidence object — pure evaluate_claim_vs only flags when both sides
        # of a comparison exist elsewhere (caller / live required path).
        return GpuIntegrityDecision(
            integrity_zero=False,
            integrity_codes=[],
            reason="no_evidence",
        )

    # status == passed (or unknown non-fail string treated as measured-present).
    mismatch = False
    if claimed_gpu_model and measured_gpu_model:
        if not models_match(claimed_gpu_model, measured_gpu_model):
            mismatch = True
            reason_parts.append(
                f"model_mismatch claimed={claimed_gpu_model!r} "
                f"measured={measured_gpu_model!r} "
                f"families=({normalize_gpu_model(claimed_gpu_model)},"
                f"{normalize_gpu_model(measured_gpu_model)})"
            )
    if claimed_gpu_count is not None and measured_gpu_count is not None:
        try:
            if int(claimed_gpu_count) != int(measured_gpu_count):
                mismatch = True
                reason_parts.append(
                    f"count_mismatch claimed={claimed_gpu_count} measured={measured_gpu_count}"
                )
        except (TypeError, ValueError):
            mismatch = True
            reason_parts.append("count_unparseable")

    if mismatch:
        codes.extend([INVENTORY_SPOOF_CODE, GPU_PROBE_MISMATCH_CODE])
        return GpuIntegrityDecision(
            integrity_zero=True,
            integrity_codes=_uniq(codes),
            correctness=0.0,
            fabric_gate=0.0,
            reason=";".join(reason_parts) or "claim_vs_evidence_mismatch",
        )

    return GpuIntegrityDecision(
        integrity_zero=False,
        integrity_codes=[],
        reason="claim_consistent",
    )


def evaluate_gpu_probe_integrity(
    *,
    claimed_gpu_model: str | None = None,
    claimed_gpu_count: int | None = None,
    evidence_status: EvidenceStatus = None,
    measured_gpu_model: str | None = None,
    measured_gpu_count: int | None = None,
    proof_tier: str | None = None,
    execution_backend: str | None = None,
    requires_live_gpu_evidence: bool | None = None,
    hyper: HyperSettings | None = None,
    gpu_probe_status: str | None = None,
) -> GpuIntegrityDecision:
    """Full GPU honesty evaluation for an attempt score path.

    Precedence:
    1. ``HYPER_SIM_GPU_PROBE_FAIL`` inject → always integrity zero (VAL-GPU-052)
    2. Live evidence required + missing/failed evidence → zero
    3. Claim vs measured mismatch (when evidence present) → zero (VAL-GPU-050)
    4. Unprobed sim path (default) → clean (VAL-GPU-051)
    """

    settings = hyper if hyper is not None else get_hyper_settings()
    codes: list[str] = []
    reasons: list[str] = []

    if sim_gpu_probe_fail_active(settings):
        codes.extend([GPU_PROBE_FAIL_CODE, INVENTORY_SPOOF_CODE, INTEGRITY_FAIL_CODE])
        return GpuIntegrityDecision(
            integrity_zero=True,
            integrity_codes=_uniq(codes),
            correctness=0.0,
            fabric_gate=0.0,
            reason="HYPER_SIM_GPU_PROBE_FAIL",
        )

    tier = (proof_tier or "").strip().lower()
    backend = (execution_backend or "").strip().lower()
    live_required = (
        bool(requires_live_gpu_evidence)
        if requires_live_gpu_evidence is not None
        else require_gpu_evidence_for_live(settings)
    )

    # Non-sim pure paths with inventory stamp may still claim silicon; sim is
    # opt-out unless live flag is forced on.
    is_sim_path = tier in {"sim", ""} or backend in {
        "sim",
        "sim_launcher",
        "local_sim",
        "fake",
    }
    # Empty tier with no backend still treated as sim-friendly CI default.
    if tier in {"sim"} or backend in {"sim_launcher", "local_sim"}:
        is_sim_path = True

    status = (
        str(evidence_status).strip().lower()
        if evidence_status
        else (str(gpu_probe_status).strip().lower() if gpu_probe_status else None)
    )
    # Normalize inventory merge labels (verified|failed|error|none) onto probe statuses.
    if status == "verified":
        status = "passed"
    if status == "none":
        status = None

    if live_required and not is_sim_path:
        if status is None:
            codes.extend([GPU_PROBE_FAIL_CODE, INTEGRITY_FAIL_CODE, INVENTORY_SPOOF_CODE])
            return GpuIntegrityDecision(
                integrity_zero=True,
                integrity_codes=_uniq(codes),
                correctness=0.0,
                fabric_gate=0.0,
                reason="live_gpu_evidence_required_missing",
            )
        if status in {"failed", "error"}:
            codes.extend([GPU_PROBE_FAIL_CODE, INVENTORY_SPOOF_CODE])
            return GpuIntegrityDecision(
                integrity_zero=True,
                integrity_codes=_uniq(codes),
                correctness=0.0,
                fabric_gate=0.0,
                reason=f"live_gpu_evidence_{status}",
            )

    # When evidence is present (even on sim with explicit attach), check claim.
    if status is not None or measured_gpu_model is not None or measured_gpu_count is not None:
        claim_decision = evaluate_claim_vs_evidence(
            claimed_gpu_model=claimed_gpu_model,
            claimed_gpu_count=claimed_gpu_count,
            evidence_status=status or ("passed" if measured_gpu_model else None),
            measured_gpu_model=measured_gpu_model,
            measured_gpu_count=measured_gpu_count,
        )
        if claim_decision.integrity_zero:
            return claim_decision

    # VAL-GPU-051: unprobed sim residual is clean.
    if is_sim_path or not live_required:
        return GpuIntegrityDecision(
            integrity_zero=False,
            integrity_codes=[],
            reason="sim_or_unrequired_unprobed_ok" if status is None else "clean",
        )

    return GpuIntegrityDecision(
        integrity_zero=False,
        integrity_codes=codes,
        reason=";".join(reasons) or "clean",
    )


def apply_gpu_integrity(
    *,
    correctness: float,
    efficiency: float,
    fabric_gate: float,
    tee_bonus: float,
    decision: GpuIntegrityDecision,
    hyper: HyperSettings | None = None,  # noqa: ARG001 — reserved for future floors
) -> tuple[float, float, float, float]:
    """Apply decision gate overrides to the four factors (tee_bonus untouched).

    Does **not** introduce a 5th factor. Returns correctness, efficiency,
    fabric_gate, tee_bonus ready for :func:`compute_four_factor`.
    """

    c = float(correctness)
    e = float(efficiency)
    g = float(fabric_gate)
    t = float(tee_bonus)
    if decision.correctness is not None:
        c = float(decision.correctness)
    if decision.fabric_gate is not None:
        g = float(decision.fabric_gate)
    if decision.integrity_zero:
        # Hard residual: force gate zeros so forensics and product both zero even
        # before integrity_zero composite override (matches fabric honesty style).
        if decision.correctness is None:
            c = 0.0
        if decision.fabric_gate is None:
            g = 0.0
    return c, e, g, t


def merge_gpu_codes(
    existing: list[str] | tuple[str, ...] | None,
    decision: GpuIntegrityDecision,
) -> list[str]:
    """Merge GPU integrity codes into an attempt's integrity code list."""

    out: list[str] = []
    if existing:
        for raw in existing:
            s = str(raw or "").strip()
            if s and s not in out:
                out.append(s)
    for code in decision.integrity_codes:
        if code and code not in out:
            out.append(code)
    return out


def decision_from_inventory_blob(
    inventory: dict[str, Any] | str | None,
    *,
    claimed_gpu_model: str | None = None,
    claimed_gpu_count: int | None = None,
    proof_tier: str | None = None,
    hyper: HyperSettings | None = None,
) -> GpuIntegrityDecision:
    """Build decision from node ``inventory_json`` merge fields (VAL-GPU-027)."""

    inv: dict[str, Any]
    if inventory is None:
        inv = {}
    elif isinstance(inventory, str):
        text = inventory.strip()
        if not text:
            inv = {}
        else:
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                inv = {}
            else:
                inv = parsed if isinstance(parsed, dict) else {}
    elif isinstance(inventory, dict):
        inv = inventory
    else:
        inv = {}

    status = inv.get("gpu_probe_status")
    measured_model = inv.get("measured_gpu_model") or inv.get("gpu_model")
    measured_count = inv.get("measured_gpu_count")
    if measured_count is None and isinstance(inv.get("gpu_uuids"), list):
        measured_count = len(inv["gpu_uuids"])

    return evaluate_gpu_probe_integrity(
        claimed_gpu_model=claimed_gpu_model,
        claimed_gpu_count=claimed_gpu_count,
        evidence_status=None,
        measured_gpu_model=str(measured_model) if measured_model else None,
        measured_gpu_count=int(measured_count) if measured_count is not None else None,
        proof_tier=proof_tier,
        gpu_probe_status=str(status) if status else None,
        hyper=hyper,
    )


def _uniq(codes: list[str]) -> list[str]:
    out: list[str] = []
    for c in codes:
        s = str(c or "").strip()
        if s and s not in out:
            out.append(s)
    return out


__all__ = [
    "GPU_PROBE_FAIL_CODE",
    "GPU_PROBE_MISMATCH_CODE",
    "INTEGRITY_FAIL_CODE",
    "INVENTORY_SPOOF_CODE",
    "GpuIntegrityDecision",
    "apply_gpu_integrity",
    "decision_from_inventory_blob",
    "evaluate_claim_vs_evidence",
    "evaluate_gpu_probe_integrity",
    "merge_gpu_codes",
    "require_gpu_evidence_for_live",
    "sim_gpu_probe_fail_active",
]
