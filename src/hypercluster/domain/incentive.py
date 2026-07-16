"""Incentive mass clamp + optional top-k / max-fraction + sum-normalize.

M10 (VAL-WGT-010..014): finite ≥0 mass only; sum-normalize to unit map when
Σmass > 0; empty → ``{}`` burn-safe. Optional simple policies (top-k keepers,
max-fraction share cap) apply **before** re-normalizing.

Downstream of the four-factor composite product only — never a 5th factor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from hypercluster.domain.aggregation import finite_non_negative, sanitize_weights_map

# Absolute sum tolerance for unit-sum checks (VAL-WGT-011/014).
UNIT_SUM_TOLERANCE = 1e-6
_DEFAULT_DUST = 1e-12


def weight_sum(weights: Mapping[str, float] | None) -> float:
    """Finite sum of weight values (non-finite entries treated as 0)."""

    if not weights:
        return 0.0
    total = 0.0
    for val in weights.values():
        total += finite_non_negative(val)
    return float(total)


def clamp_mass_map(
    mass: Mapping[str, float] | dict[str, float] | None,
    *,
    dust: float = 0.0,
) -> dict[str, float]:
    """Finite ≥0 mass only; drop empty keys / NaN / ±Inf / negative (VAL-WGT-010).

    Entries ≤ ``dust`` are dropped when dust > 0.
    """

    clean = sanitize_weights_map(mass)
    if dust <= 0.0:
        # Drop pure zeros from mass source (they contribute nothing to normalize).
        return {k: v for k, v in clean.items() if v > 0.0}
    floor = float(dust)
    if not math.isfinite(floor) or floor < 0.0:
        floor = 0.0
    return {k: v for k, v in clean.items() if v > floor}


def apply_top_k(
    mass: Mapping[str, float],
    *,
    k: int | None,
) -> dict[str, float]:
    """Keep the ``k`` largest positive mass keys; pass-through when k is None/≤0."""

    cleaned = clamp_mass_map(mass)
    if k is None:
        return cleaned
    try:
        keep = int(k)
    except (TypeError, ValueError):
        return cleaned
    if keep <= 0 or keep >= len(cleaned):
        return cleaned
    ordered = sorted(cleaned.items(), key=lambda item: (-item[1], item[0]))
    return {key: val for key, val in ordered[:keep]}


def apply_max_fraction(
    mass: Mapping[str, float],
    *,
    max_fraction: float | None,
) -> dict[str, float]:
    """Clamp each key's share to ``max_fraction`` of current total (pre-norm).

    Tails keep residual mass so ranking among non-dominant keys is preserved;
    the result is **not** yet unit-sum — caller re-normalizes.
    """

    cleaned = clamp_mass_map(mass)
    if max_fraction is None:
        return cleaned
    try:
        cap = float(max_fraction)
    except (TypeError, ValueError):
        return cleaned
    if not math.isfinite(cap) or cap <= 0.0:
        return cleaned
    if cap >= 1.0:
        return cleaned

    total = weight_sum(cleaned)
    if total <= 0.0:
        return {}
    max_mass = cap * total
    capped: dict[str, float] = {}
    for key, val in cleaned.items():
        capped[key] = min(float(val), max_mass)
    # If everything was above the cap, each entry equals max_mass (equal after
    # later normalize); residual zeros dropped.
    return clamp_mass_map(capped)


def normalize_sum_to_unit(
    mass: Mapping[str, float] | dict[str, float] | None,
    *,
    dust: float = 0.0,
) -> dict[str, float]:
    """``W[h] = M[h] / ΣM`` when ΣM > 0 else ``{}`` (VAL-WGT-011/012)."""

    cleaned = clamp_mass_map(mass, dust=dust)
    total = weight_sum(cleaned)
    if total <= 0.0:
        return {}
    return {key: float(val) / total for key, val in cleaned.items()}


def finalize_incentives(
    mass: Mapping[str, float] | dict[str, float] | None,
    *,
    sum_normalize: bool = True,
    top_k: int | None = None,
    max_fraction: float | None = None,
    dust: float = _DEFAULT_DUST,
) -> dict[str, float]:
    """Full incentive pipeline: clamp → optional top-k / max-fraction → normalize.

    When ``sum_normalize`` is false, returns clamped (policy-applied) absolute
    mass. Empty / zero mass always yields ``{}``.
    """

    stage = clamp_mass_map(mass, dust=0.0)
    if top_k is not None:
        stage = apply_top_k(stage, k=top_k)
    if max_fraction is not None:
        stage = apply_max_fraction(stage, max_fraction=max_fraction)
    stage = clamp_mass_map(stage, dust=dust)
    if not sum_normalize:
        return stage
    return normalize_sum_to_unit(stage, dust=0.0)


def resolve_incentive_knobs(hyper: Any | None) -> dict[str, Any]:
    """Read HYPER_* incentive knobs with safe defaults (sum-normalize ON)."""

    if hyper is None:
        return {
            "sum_normalize": True,
            "top_k": None,
            "max_fraction": None,
            "dust": _DEFAULT_DUST,
        }
    top_k_raw = getattr(hyper, "weight_top_k", None)
    top_k: int | None
    if top_k_raw is None or top_k_raw == "" or top_k_raw == 0:
        top_k = None
    else:
        try:
            top_k = int(top_k_raw)
        except (TypeError, ValueError):
            top_k = None
        if top_k is not None and top_k <= 0:
            top_k = None

    max_frac_raw = getattr(hyper, "weight_max_fraction", None)
    max_fraction: float | None
    if max_frac_raw is None or max_frac_raw == "":
        max_fraction = None
    else:
        try:
            max_fraction = float(max_frac_raw)
        except (TypeError, ValueError):
            max_fraction = None
        if max_fraction is not None and (
            not math.isfinite(max_fraction) or max_fraction <= 0.0 or max_fraction >= 1.0
        ):
            # 1.0+ means no effective cap; treat as off.
            if max_fraction is not None and max_fraction >= 1.0:
                max_fraction = None
            elif max_fraction is not None and (
                not math.isfinite(max_fraction) or max_fraction <= 0.0
            ):
                max_fraction = None

    dust_raw = getattr(hyper, "weight_dust", _DEFAULT_DUST)
    try:
        dust = float(dust_raw)
    except (TypeError, ValueError):
        dust = _DEFAULT_DUST
    if not math.isfinite(dust) or dust < 0.0:
        dust = _DEFAULT_DUST

    sum_norm = bool(getattr(hyper, "incentive_sum_normalize", True))
    return {
        "sum_normalize": sum_norm,
        "top_k": top_k,
        "max_fraction": max_fraction,
        "dust": dust,
    }


def finalize_incentives_with_settings(
    mass: Mapping[str, float] | dict[str, float] | None,
    *,
    hyper: Any | None = None,
) -> dict[str, float]:
    """Apply settings-driven finalize for emission / weight-preview / snapshots."""

    knobs = resolve_incentive_knobs(hyper)
    return finalize_incentives(
        mass,
        sum_normalize=bool(knobs["sum_normalize"]),
        top_k=knobs["top_k"],  # type: ignore[arg-type]
        max_fraction=knobs["max_fraction"],  # type: ignore[arg-type]
        dust=float(knobs["dust"]),
    )


__all__ = [
    "UNIT_SUM_TOLERANCE",
    "apply_max_fraction",
    "apply_top_k",
    "clamp_mass_map",
    "finalize_incentives",
    "finalize_incentives_with_settings",
    "normalize_sum_to_unit",
    "resolve_incentive_knobs",
    "weight_sum",
]
