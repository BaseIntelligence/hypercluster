"""Four-factor composite scoring engine (M6 / library/scoring.md).

Fixed product formula (do not invent alternates)::

    composite = correctness × efficiency × fabric_gate × tee_bonus

Fulfills VAL-SCORE-001..007, 021, 026 (per-attempt product + forensics).
Aggregation / leaderboard / weight push live in later M6 slices.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import Score
from hypercluster.settings import HyperSettings, get_hyper_settings

# Integrity / cheat catalogue that forces composite = 0 (VAL-SCORE-007).
# Matches architecture §10.1 and library/scoring.md inject surface.
CHEAT_REASON_CODES: frozenset[str] = frozenset(
    {
        "attestation_fail",
        "image_mutation",
        "image_compose_mutation",
        "rank_desync",
        "inventory_spoof",
        "fabric_lie",
        "forbidden_eth_fallback",
        "quote_invalid",
        "quote_sig_invalid",
        "integrity_fail",
        "allreduce_out_of_band",
        # M9 GPU probe honesty (VAL-GPU-050/052) — map into same integrity zero
        # path; never a fifth published scoring factor.
        "gpu_probe_fail",
        "gpu_probe_mismatch",
    }
)

_GATE_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class EfficiencyFloorResult:
    """Outcome of efficiency floor policy (VAL-SCORE-021)."""

    stored: float
    for_product: float
    below_floor: bool
    floor: float


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Visible four-factor breakdown for an attempt (VAL-SCORE-001/026)."""

    correctness: float
    efficiency: float
    fabric_gate: float
    tee_bonus: float
    composite: float
    integrity_zero: bool = False
    below_efficiency_floor: bool = False
    reason_codes: list[str] = field(default_factory=list)
    wall_seconds: float | None = None
    compute_metric: float | None = None

    def factors_dict(self) -> dict[str, float]:
        """Exactly the four factors + composite (no hidden multipliers)."""

        return {
            "correctness": float(self.correctness),
            "efficiency": float(self.efficiency),
            "fabric_gate": float(self.fabric_gate),
            "tee_bonus": float(self.tee_bonus),
            "composite": float(self.composite),
        }


def coerce_gate01(value: float | int | bool) -> float:
    """Map an input to a hard gate in {0.0, 1.0} (VAL-SCORE-002/005).

    Full pass requires a value that is effectively 1 (or True). Partial credit
    (e.g. 0.5, 0.99) collapses to 0 in v1.
    """

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return 1.0 if v + _GATE_EPS >= 1.0 else 0.0


def coerce_fabric_gate(value: float | int | bool) -> float:
    """fabric_gate hard gate in {0, 1} (VAL-SCORE-005)."""

    return coerce_gate01(value)


def coerce_tee_bonus(value: float | int) -> float:
    """Honest tee_bonus is a multiplier ≥ 1.0 (VAL-SCORE-006).

    Sub-1 "penalty" attempts clamp up to 1.0; penalties must use integrity zero
    or correctness/fabric gates instead.
    """

    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(v):
        return 1.0
    return max(1.0, v)


def apply_efficiency_floor(
    efficiency: float,
    *,
    floor: float = 0.0,
) -> EfficiencyFloorResult:
    """Normalize efficiency continuous ≥0 and apply HYPER_EFFICIENCY_FLOOR.

    VAL-SCORE-004 / VAL-SCORE-021:
    - Never store a negative efficiency.
    - Default floor 0.0 keeps tiny positives as positive product contributions.
    - When floor > 0 and measured efficiency is below floor, product contribution
      is knocked to 0 while the stored continuous metric remains the measured
      non-negative value (forensics).
    """

    try:
        raw = float(efficiency)
    except (TypeError, ValueError):
        raw = 0.0
    if not math.isfinite(raw):
        raw = 0.0
    stored = max(0.0, raw)
    floor_v = max(0.0, float(floor) if math.isfinite(float(floor)) else 0.0)
    if floor_v > 0.0 and stored + _GATE_EPS < floor_v:
        return EfficiencyFloorResult(
            stored=stored,
            for_product=0.0,
            below_floor=True,
            floor=floor_v,
        )
    return EfficiencyFloorResult(
        stored=stored,
        for_product=stored,
        below_floor=False,
        floor=floor_v,
    )


def _normalize_integrity_codes(
    integrity_codes: list[str] | tuple[str, ...] | set[str] | None,
    *,
    integrity_zero: bool,
) -> tuple[list[str], bool]:
    codes: list[str] = []
    if integrity_codes:
        for raw in integrity_codes:
            code = str(raw or "").strip().lower()
            if not code:
                continue
            if code not in codes:
                codes.append(code)
    forced = bool(integrity_zero) or any(c in CHEAT_REASON_CODES for c in codes)
    if forced and not any(c in CHEAT_REASON_CODES for c in codes) and "integrity_fail" not in codes:
        codes.append("integrity_fail")
    return codes, forced


def compute_four_factor(
    *,
    correctness: float | int | bool,
    efficiency: float,
    fabric_gate: float | int | bool,
    tee_bonus: float = 1.0,
    integrity_zero: bool = False,
    integrity_codes: list[str] | tuple[str, ...] | set[str] | None = None,
    hyper: HyperSettings | None = None,
    wall_seconds: float | None = None,
    compute_metric: float | None = None,
    efficiency_floor: float | None = None,
) -> ScoreBreakdown:
    """Compute the fixed four-factor product with gate/floor/integrity rules.

    Factors written to the breakdown are the *visible residual values* used for
    forensics (VAL-SCORE-026). Composite is zeroed when integrity fails even if
    factor residuals remain non-zero.
    """

    settings = hyper if hyper is not None else get_hyper_settings()
    floor = (
        float(efficiency_floor)
        if efficiency_floor is not None
        else float(settings.efficiency_floor)
    )

    c = coerce_gate01(correctness)
    g = coerce_fabric_gate(fabric_gate)
    t = coerce_tee_bonus(tee_bonus)

    # Prefer explicit compute_metric when provided (wall clock is cap-only).
    eff_input = float(compute_metric) if compute_metric is not None else float(efficiency)
    eff_res = apply_efficiency_floor(eff_input, floor=floor)
    e_stored = eff_res.stored
    e_product = eff_res.for_product

    codes, forced_zero = _normalize_integrity_codes(integrity_codes, integrity_zero=integrity_zero)
    if eff_res.below_floor:
        codes = list(dict.fromkeys([*codes, "below_efficiency_floor"]))

    # Product uses stored gates + floored efficiency contribution.
    product = float(c) * float(e_product) * float(g) * float(t)
    if forced_zero:
        composite = 0.0
    else:
        composite = product if math.isfinite(product) else 0.0
        if composite < 0.0:
            # Contract: finite non-negative (paranoia against future factor inversions).
            composite = 0.0

    return ScoreBreakdown(
        correctness=c,
        # Store continuous measured efficiency (e_stored), not the knocked product input,
        # so forensics can show e.g. 0.05 under floor 0.1 with composite 0.
        efficiency=e_stored,
        fabric_gate=g,
        tee_bonus=t,
        composite=float(composite),
        integrity_zero=forced_zero,
        below_efficiency_floor=eff_res.below_floor,
        reason_codes=codes,
        wall_seconds=wall_seconds,
        compute_metric=float(compute_metric) if compute_metric is not None else None,
    )


def four_factor_composite(
    *,
    correctness: float,
    efficiency: float,
    fabric_gate: float,
    tee_bonus: float,
    integrity_zero: bool = False,
    integrity_codes: list[str] | None = None,
    hyper: HyperSettings | None = None,
) -> float:
    """Convenience wrapper returning only the composite float."""

    return compute_four_factor(
        correctness=correctness,
        efficiency=efficiency,
        fabric_gate=fabric_gate,
        tee_bonus=tee_bonus,
        integrity_zero=integrity_zero,
        integrity_codes=integrity_codes,
        hyper=hyper,
    ).composite


def score_breakdown_to_public(breakdown: ScoreBreakdown) -> dict[str, Any]:
    """API/debug JSON: factors stay visible even when composite is 0 (VAL-SCORE-026)."""

    body: dict[str, Any] = breakdown.factors_dict()
    body["integrity_zero"] = bool(breakdown.integrity_zero)
    body["below_efficiency_floor"] = bool(breakdown.below_efficiency_floor)
    body["reason_codes"] = list(breakdown.reason_codes)
    if breakdown.wall_seconds is not None:
        body["wall_seconds"] = float(breakdown.wall_seconds)
    if breakdown.compute_metric is not None:
        body["compute_metric"] = float(breakdown.compute_metric)
    return body


def score_row_to_public(row: Score) -> dict[str, Any]:
    """Public shape for a persisted Score ORM row (factor-visible)."""

    public = row.to_dict()
    # Guarantee the four factor keys are always present as floats.
    public["correctness"] = float(row.correctness)
    public["efficiency"] = float(row.efficiency)
    public["fabric_gate"] = float(row.fabric_gate)
    public["tee_bonus"] = float(row.tee_bonus)
    public["composite"] = float(row.composite)
    return public


async def list_scores_for_hotkey(
    session: AsyncSession,
    hotkey: str,
    *,
    limit: int = 100,
) -> list[Score]:
    """Return score history for a hotkey (newest first)."""

    lim = max(1, min(int(limit), 1000))
    result = await session.execute(
        select(Score).where(Score.hotkey == hotkey).order_by(Score.created_at.desc()).limit(lim)
    )
    return list(result.scalars().all())


async def get_score_by_attempt(
    session: AsyncSession,
    attempt_id: str,
) -> Score | None:
    result = await session.execute(select(Score).where(Score.attempt_id == attempt_id))
    return result.scalar_one_or_none()


def merge_score_details(
    *,
    breakdown: ScoreBreakdown,
    extra: dict[str, Any] | None = None,
) -> str:
    """Canonical details_json for a scores row (forensic factors)."""

    payload: dict[str, Any] = {
        "factors": breakdown.factors_dict(),
        "integrity_zero": breakdown.integrity_zero,
        "below_efficiency_floor": breakdown.below_efficiency_floor,
        "reason_codes": list(breakdown.reason_codes),
    }
    if breakdown.wall_seconds is not None:
        payload["wall_seconds"] = breakdown.wall_seconds
    if breakdown.compute_metric is not None:
        payload["compute_metric"] = breakdown.compute_metric
    if extra:
        payload["extra"] = extra
    return json.dumps(payload, sort_keys=True)


__all__ = [
    "CHEAT_REASON_CODES",
    "EfficiencyFloorResult",
    "ScoreBreakdown",
    "apply_efficiency_floor",
    "coerce_fabric_gate",
    "coerce_gate01",
    "coerce_tee_bonus",
    "compute_four_factor",
    "four_factor_composite",
    "get_score_by_attempt",
    "list_scores_for_hotkey",
    "merge_score_details",
    "score_breakdown_to_public",
    "score_row_to_public",
]
