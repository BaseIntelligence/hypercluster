"""Score-row aggregation window, raw weights, leaderboard, self-deal damping.

Fulfills VAL-SCORE-008, 009, 010, 011, 012, 018, 022, 027, 029.
M10 incentive normalize (VAL-WGT-010..014) applied at ``compute_raw_weights``:
clamp finite ≥0 → optional top-k / max-fraction → sum-normalize to ~1.0 when
mass > 0; empty → ``{}``. Leaderboard remains absolute population mass for
observability; emission / get_weights / weight-preview use the unit-sum map.

Policy (architecture §10.2 / library/scoring.md):

1. Select last ``HYPER_SCORE_WINDOW_ATTEMPTS`` score rows (newest first).
2. Bind each row to ``(hotkey, role)``; roles ∈ {demand, supply, joint}.
3. Sum positive composites per hotkey (single v1 emission pool).
4. Soft-damp self-deal rows (details.self_deal or explicit flag) without NaN.
5. Emit finite ≥0 mass map then sum-normalize (M10 default); empty → ``{}``.
6. Leaderboard = aggregates sorted by mass descending; vacant → empty items.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import Score
from hypercluster.settings import HyperSettings, get_hyper_settings

# Architecture §10 / VAL-SCORE-008 role catalogue.
SCORE_ROLES: frozenset[str] = frozenset({"demand", "supply", "joint"})

_DEFAULT_SELF_DEAL_DAMPING = 0.5


@dataclass(frozen=True, slots=True)
class HotkeyAggregate:
    """Per-hotkey mass after window + self-deal policy."""

    hotkey: str
    aggregate: float
    roles: dict[str, float] = field(default_factory=dict)
    score_count: int = 0
    self_deal_count: int = 0

    def to_public(self, *, rank: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "hotkey": self.hotkey,
            "aggregate": float(self.aggregate),
            "roles": {k: float(v) for k, v in sorted(self.roles.items())},
            "score_count": int(self.score_count),
            "self_deal_count": int(self.self_deal_count),
        }
        if rank is not None:
            body["rank"] = int(rank)
        return body


def finite_non_negative(value: float | int | None) -> float:
    """Coerce to a finite non-negative float (burn-safe)."""

    try:
        v = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or v < 0.0:
        return 0.0
    return v


def sanitize_weights_map(
    weights: Mapping[str, float] | dict[str, float] | None,
) -> dict[str, float]:
    """Return only valid hotkey→finite≥0 entries (VAL-SCORE-009/010).

    Drops empty keys, NaN/Inf, and negatives. Empty input → ``{}`` (burn-safe).
    """

    if not weights:
        return {}
    clean: dict[str, float] = {}
    for raw_key, raw_val in weights.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        val = finite_non_negative(raw_val)
        # Preserve intentional zeros only when the original value was a finite 0;
        # skip keys that collapsed from NaN/Inf/negative unless raw was zero.
        try:
            original = float(raw_val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(original) or original < 0.0:
            continue
        clean[key] = val
    return clean


def apply_self_deal_damping(
    composite: float,
    *,
    self_deal: bool,
    damping: float = _DEFAULT_SELF_DEAL_DAMPING,
) -> float:
    """Soft self-deal penalty: reduce mass, never NaN (VAL-SCORE-012).

    ``mass' = mass * (1 - damping)`` when self_deal is true.
    ``damping`` is clamped to [0, 1]. Non-finite composites collapse to 0.
    """

    mass = finite_non_negative(composite)
    if not self_deal:
        return mass
    try:
        d = float(damping)
    except (TypeError, ValueError):
        d = _DEFAULT_SELF_DEAL_DAMPING
    if not math.isfinite(d):
        d = _DEFAULT_SELF_DEAL_DAMPING
    d = min(1.0, max(0.0, d))
    return finite_non_negative(mass * (1.0 - d))


def _parse_details(details_json: str | None) -> dict[str, Any]:
    if not details_json:
        return {}
    try:
        parsed = json.loads(details_json)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def is_self_deal_score(row: Score) -> bool:
    """Detect self-deal / collusion flags on a score row.

    Recognized shapes (sim flag surface for VAL-SCORE-012):
    - details.self_deal truthy
    - details.extra.self_deal truthy
    - details.tee_decision-style codes containing ``self_deal``
    """

    details = _parse_details(row.details_json)
    if details.get("self_deal") is True:
        return True
    extra = details.get("extra")
    if isinstance(extra, dict) and extra.get("self_deal") is True:
        return True
    codes = details.get("reason_codes") or []
    if isinstance(codes, list) and any(str(c).lower() == "self_deal" for c in codes):
        return True
    if isinstance(extra, dict):
        extra_codes = extra.get("reason_codes") or extra.get("integrity_codes") or []
        if isinstance(extra_codes, list) and any(
            str(c).lower() == "self_deal" for c in extra_codes
        ):
            return True
    return False


def normalize_role(role: str | None) -> str:
    raw = (role or "demand").strip().lower()
    if raw in SCORE_ROLES:
        return raw
    return "demand"


def score_rows_bind_hotkey_role(
    rows: Iterable[Score],
) -> tuple[bool, list[str]]:
    """Validate every score binds a non-empty hotkey + known role (VAL-SCORE-008).

    Returns (ok, missing_descriptions).
    """

    missing: list[str] = []
    for row in rows:
        if not str(row.hotkey or "").strip():
            missing.append(f"score {row.id}: missing hotkey")
        if normalize_role(row.role) not in SCORE_ROLES:
            missing.append(f"score {row.id}: invalid role {row.role!r}")
    return (len(missing) == 0, missing)


async def list_scores_in_window(
    session: AsyncSession,
    *,
    window: int,
) -> list[Score]:
    """Return newest score rows bounded by the aggregation window (VAL-SCORE-022)."""

    limit = max(1, int(window))
    result = await session.execute(
        select(Score).order_by(Score.created_at.desc(), Score.id.desc()).limit(limit)
    )
    return list(result.scalars().all())


def aggregate_score_rows(
    rows: Iterable[Score],
    *,
    self_deal_damping: float = _DEFAULT_SELF_DEAL_DAMPING,
) -> dict[str, HotkeyAggregate]:
    """Aggregate windowed score rows into per-hotkey mass (single pool).

    Dual-role (demand + supply same hotkey) is allowed and sums finite mass
    (VAL-SCORE-027). Self-deal flags reduce contributing mass (VAL-SCORE-012).
    """

    role_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    self_deal_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        hotkey = str(row.hotkey or "").strip()
        if not hotkey:
            continue
        role = normalize_role(row.role)
        raw = finite_non_negative(row.composite)
        # Only positive composites contribute to raw weight mass.
        if raw <= 0.0:
            counts[hotkey] += 1
            # Still record zero role key so dual-role identity is visible.
            role_sums[hotkey][role] += 0.0
            continue
        flagged = is_self_deal_score(row)
        mass = apply_self_deal_damping(raw, self_deal=flagged, damping=self_deal_damping)
        role_sums[hotkey][role] += mass
        counts[hotkey] += 1
        if flagged:
            self_deal_counts[hotkey] += 1

    out: dict[str, HotkeyAggregate] = {}
    for hotkey, roles in role_sums.items():
        total = finite_non_negative(sum(roles.values()))
        out[hotkey] = HotkeyAggregate(
            hotkey=hotkey,
            aggregate=total,
            roles={r: finite_non_negative(v) for r, v in roles.items()},
            score_count=int(counts.get(hotkey, 0)),
            self_deal_count=int(self_deal_counts.get(hotkey, 0)),
        )
    return out


def aggregates_to_weights(aggregates: Mapping[str, HotkeyAggregate]) -> dict[str, float]:
    """Hotkey → finite ≥0 absolute mass map; drop zero-mass keys.

    This is pre-normalize population mass (leaderboard / raw_mass snapshot).
    Emission surfaces call :func:`finalize_incentives_with_settings` afterwards.
    """

    weights: dict[str, float] = {}
    for hotkey, agg in aggregates.items():
        mass = finite_non_negative(agg.aggregate)
        if mass > 0.0:
            weights[hotkey] = mass
    return sanitize_weights_map(weights)


async def compute_hotkey_aggregates(
    session: AsyncSession,
    *,
    hyper: HyperSettings | None = None,
    window: int | None = None,
) -> dict[str, HotkeyAggregate]:
    """Load windowed scores and aggregate (source of leaderboard + weights)."""

    settings = hyper if hyper is not None else get_hyper_settings()
    win = int(window) if window is not None else int(settings.score_window_attempts)
    damping = float(getattr(settings, "self_deal_damping", _DEFAULT_SELF_DEAL_DAMPING))
    rows = await list_scores_in_window(session, window=win)
    return aggregate_score_rows(rows, self_deal_damping=damping)


async def compute_mass_map(
    session: AsyncSession,
    *,
    hyper: HyperSettings | None = None,
    window: int | None = None,
) -> dict[str, float]:
    """Pre-normalize absolute mass map (VAL-WGT-013 raw_mass source)."""

    aggregates = await compute_hotkey_aggregates(session, hyper=hyper, window=window)
    return aggregates_to_weights(aggregates)


async def compute_raw_weights(
    session: AsyncSession,
    *,
    hyper: HyperSettings | None = None,
    window: int | None = None,
) -> dict[str, float]:
    """Incentive weights for get_weights / weight-preview / push.

    M10 default: unit-sum map when mass > 0 (VAL-WGT-011/014); empty → ``{}``
    (VAL-WGT-012). Absolute mass is available via :func:`compute_mass_map`.
    """

    # Local import avoids circular import with incentive → aggregation helpers.
    from hypercluster.domain.incentive import finalize_incentives_with_settings

    settings = hyper if hyper is not None else get_hyper_settings()
    mass = await compute_mass_map(session, hyper=settings, window=window)
    return finalize_incentives_with_settings(mass, hyper=settings)


async def build_leaderboard(
    session: AsyncSession,
    *,
    hyper: HyperSettings | None = None,
    window: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregated rows keyed by hotkey, ordered by mass desc (VAL-SCORE-018/029).

    Vacant participation returns ``[]`` — never invents ranks from the provider
    registry or non-scored hotkeys.
    """

    aggregates = await compute_hotkey_aggregates(session, hyper=hyper, window=window)
    ordered = sorted(
        aggregates.values(),
        key=lambda a: (-a.aggregate, a.hotkey),
    )
    if limit is not None:
        ordered = ordered[: max(0, int(limit))]
    return [agg.to_public(rank=idx + 1) for idx, agg in enumerate(ordered)]


__all__ = [
    "SCORE_ROLES",
    "HotkeyAggregate",
    "aggregate_score_rows",
    "aggregates_to_weights",
    "apply_self_deal_damping",
    "build_leaderboard",
    "compute_hotkey_aggregates",
    "compute_mass_map",
    "compute_raw_weights",
    "finite_non_negative",
    "is_self_deal_score",
    "list_scores_in_window",
    "normalize_role",
    "sanitize_weights_map",
    "score_rows_bind_hotkey_role",
]
