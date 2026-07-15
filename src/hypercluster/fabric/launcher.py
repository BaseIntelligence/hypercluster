"""Multi-node sim launcher contract + honesty layers L1/L2 (architecture §8.3–8.4).

Fulfills:
  VAL-FAB-013  synthetic NCCL allreduce metrics under sim (L1)
  VAL-FAB-014  LaunchResult status enumerations (succeeded|failed|timeout)
  VAL-FAB-015  cross-rank progress digests when L2 enabled
  VAL-FAB-025  inventory spoof path zeros fabric_gate via honesty check
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from hypercluster.fabric.discovery import DIGEST_PREFIX, FabricReport, canonical_json
from hypercluster.fabric.gates import (
    FabricGateResult,
    evaluate_fabric_gate,
    has_active_ib_devices,
    summarize_gate_for_score,
)
from hypercluster.fabric.planner import PlacementResult, RankBinding

LaunchStatus = Literal["succeeded", "failed", "timeout"]
HonestyLevel = Literal["l0", "l1", "l2"]

# L1 allreduce golden bands (GB/s per world-size relative metric).
# Fixed-size allreduce bitrate ± tolerance for honesty pass path.
ALLREDUCE_BASE_GBPS = 18.0
ALLREDUCE_TOLERANCE_FRAC = 0.15  # ±15%
ALLREDUCE_BYTES = 64 * 1024 * 1024  # 64 MiB fixed synthetic payload

LAUNCHER_VERSION = "fabric-launcher.v1"


class EfficiencyMetrics(BaseModel):
    """Efficiency / microbench metrics attached to a LaunchResult (L1)."""

    allreduce_gbps: float = Field(..., ge=0.0)
    allreduce_bytes: int = Field(default=ALLREDUCE_BYTES, ge=0)
    allreduce_iters: int = Field(default=8, ge=1)
    expected_allreduce_gbps: float = Field(..., ge=0.0)
    within_band: bool = True
    efficiency: float = Field(..., ge=0.0)
    wall_time_s: float = Field(default=0.05, ge=0.0)
    world_size: int = Field(..., ge=1)
    source: str = "sim_launcher"
    honesty_level: str = "l1"
    noise: float = Field(default=0.0)

    def to_public(self) -> dict[str, Any]:
        return self.model_dump()


class LaunchRequest(BaseModel):
    """Inputs for sim (or real) multi-node launch."""

    placement: PlacementResult
    image_digest: str = Field(..., min_length=1)
    entrypoint: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: int = Field(default=300, ge=0)
    fabric_mode: str = "auto"
    honesty_level: HonestyLevel = "l1"
    # Honesty injects
    inject_status: LaunchStatus | None = None  # force status when set
    inject_sleep_s: float = Field(default=0.0, ge=0.0)
    eth_fallback_injected: bool = False
    inventory_spoof: bool = False  # claimed IB absent at launch honesty check
    spoofed_node_ids: list[str] = Field(default_factory=list)
    # Optional node reports for honesty relation to claimed inventory
    node_reports: list[FabricReport] = Field(default_factory=list)
    seed: int = Field(default=0, ge=0)

    @field_validator("honesty_level")
    @classmethod
    def _honest_level(cls, value: str) -> str:
        if value not in {"l0", "l1", "l2"}:
            raise ValueError("honesty_level must be l0|l1|l2")
        return value


class LaunchResult(BaseModel):
    """Outcome of multi-node launch (architecture LaunchResult)."""

    status: LaunchStatus
    metrics: EfficiencyMetrics | None = None
    fabric_artifact_digest: str = Field(default="")
    rank_progress_digests: list[str] = Field(default_factory=list)
    nccl_debug_excerpt_digest: str = Field(default="")
    failure_code: str | None = None
    reason: str = ""
    fabric_gate: float = 1.0
    composite: float = 1.0
    score_factors: dict[str, Any] = Field(default_factory=dict)
    world_size: int = 0
    honesty_level: str = "l1"
    launcher_version: str = LAUNCHER_VERSION
    integrity_fail: bool = False

    def to_public(self) -> dict[str, Any]:
        body = self.model_dump(exclude={"metrics"})
        body["metrics"] = self.metrics.to_public() if self.metrics is not None else None
        return body

    def metrics_json(self) -> dict[str, Any]:
        """Shape suitable for job_attempts.metrics_json."""

        base: dict[str, Any] = {
            "source": "sim_launcher",
            "honesty_level": self.honesty_level,
            "fabric_gate": self.fabric_gate,
            "composite": self.composite,
            "integrity_fail": self.integrity_fail,
            "failure_code": self.failure_code,
            "fabric_artifact_digest": self.fabric_artifact_digest,
            "nccl_debug_excerpt_digest": self.nccl_debug_excerpt_digest,
            "rank_progress_digests": list(self.rank_progress_digests),
            "score_factors": dict(self.score_factors),
            "launcher_version": self.launcher_version,
        }
        if self.metrics is not None:
            base.update(self.metrics.to_public())
        return base


def expected_allreduce_gbps(*, world_size: int, fabric_mode: str = "auto") -> float:
    """Deterministic expected allreduce bitrate (GB/s) for honesty golden band."""

    base = ALLREDUCE_BASE_GBPS * max(1, world_size) / 4.0
    mode = (fabric_mode or "auto").strip().lower()
    if mode == "ib":
        base *= 1.05
    elif mode in {"eth", "nvlink_only"}:
        base *= 0.95
    return round(base, 6)


def allreduce_within_band(measured: float, expected: float) -> bool:
    if expected <= 0:
        return measured >= 0
    lo = expected * (1.0 - ALLREDUCE_TOLERANCE_FRAC)
    hi = expected * (1.0 + ALLREDUCE_TOLERANCE_FRAC)
    return lo <= measured <= hi


def _digest_payload(label: str, payload: Any) -> str:
    raw = canonical_json({"label": label, "payload": payload})
    return DIGEST_PREFIX + hashlib.sha256(raw.encode()).hexdigest()


def build_rank_progress_digests(
    *,
    world_size: int,
    job_id: str,
    seed: int = 0,
    rankmap: list[RankBinding] | None = None,
) -> list[str]:
    """L2: one progress digest per rank (length == world_size)."""

    digests: list[str] = []
    rank_lookup = {b.rank: b for b in (rankmap or [])}
    for rank in range(world_size):
        binding = rank_lookup.get(rank)
        digests.append(
            _digest_payload(
                "rank_progress",
                {
                    "job_id": job_id,
                    "rank": rank,
                    "seed": seed,
                    "node_id": binding.node_id if binding else None,
                    "local_rank": binding.local_rank if binding else None,
                    "tick": 1,
                },
            )
        )
    return digests


def build_fabric_artifact_digest(
    *,
    placement: PlacementResult,
    metrics: EfficiencyMetrics | None,
    status: LaunchStatus,
) -> str:
    return _digest_payload(
        "fabric_artifact",
        {
            "planner_version": placement.planner_version,
            "graph_digest": placement.graph_digest,
            "status": status,
            "metrics": metrics.to_public() if metrics else None,
            "rank_count": len(placement.rankmap),
        },
    )


def build_nccl_debug_excerpt_digest(
    *,
    world_size: int,
    fabric_mode: str,
    status: LaunchStatus,
    seed: int,
) -> str:
    excerpt = (
        f"NCCL INFO Bootstrap : Using lo:127.0.0.1\n"
        f"NCCL INFO world_size={world_size} fabric={fabric_mode} status={status} seed={seed}\n"
        f"NCCL INFO Channel 00/0 : {world_size} ranks\n"
    )
    return DIGEST_PREFIX + hashlib.sha256(excerpt.encode()).hexdigest()


def detect_inventory_spoof(
    *,
    fabric_mode: str,
    reports: list[FabricReport],
    inventory_spoof: bool,
    spoofed_node_ids: list[str] | None = None,
) -> list[str]:
    """Return reason codes when claimed IB inventory is absent at launch."""

    codes: list[str] = []
    mode = (fabric_mode or "auto").strip().lower()
    if inventory_spoof:
        codes.append("inventory_spoof")
        if mode == "ib" or mode == "auto":
            codes.append("claimed_ib_absent_at_launch")
        return codes

    # Implicit: reports claim IB devices the honesty check treats as absent when
    # spoofed_node_ids lists them (inventory claimed; probe fails).
    spoofed = set(spoofed_node_ids or [])
    if spoofed:
        for report in reports:
            if report.node_id in spoofed and has_active_ib_devices(report):
                codes.append("inventory_spoof")
                codes.append(f"spoofed_node:{report.node_id}")
                break
    return codes


def synthetic_allreduce_metrics(
    *,
    world_size: int,
    fabric_mode: str = "auto",
    noise: float = 0.0,
    seed: int = 0,
) -> EfficiencyMetrics:
    """L1: fixed-size allreduce bitrate with optional controllable noise.

    Controllable noise still within golden band for honesty pass path when
    |noise| ≤ tolerance.
    """

    expected = expected_allreduce_gbps(world_size=world_size, fabric_mode=fabric_mode)
    # Deterministic noise from seed if noise==0? Keep noise method explicit.
    # Clamp noise so their can inject off-band for fail tests.
    noise_clamped = max(-0.5, min(0.5, float(noise)))
    # Tiny deterministic dither from seed stays well inside band (±1%).
    dither = ((seed % 100) - 50) / 10000.0  # ±0.5%
    measured = expected * (1.0 + noise_clamped + dither)
    within = allreduce_within_band(measured, expected)
    # efficiency: relative to expected (capped at 1.0 when slightly over).
    efficiency = 0.0 if expected <= 0 else min(1.0, measured / expected)
    # wall clock: optical only
    wall = max(0.01, (ALLREDUCE_BYTES / (measured * 1e9 / 8.0)) if measured > 0 else 1.0)
    return EfficiencyMetrics(
        allreduce_gbps=round(measured, 6),
        allreduce_bytes=ALLREDUCE_BYTES,
        allreduce_iters=8,
        expected_allreduce_gbps=expected,
        within_band=within,
        efficiency=round(efficiency, 6),
        wall_time_s=round(wall, 6),
        world_size=world_size,
        source="sim_launcher",
        honesty_level="l1",
        noise=noise_clamped,
    )


def sim_launch(request: LaunchRequest) -> LaunchResult:
    """Run local multi-node sim launch honesty path (no GPU / no real NCCL).

    Inject mapping (VAL-FAB-014):
      inject_status="succeeded" | None → succeeded (when honesty ok)
      inject_status="failed"           → failed
      inject_status="timeout"          → timeout
      timeout via inject_sleep_s > timeout_s → timeout
    """

    placement = request.placement
    if placement.rankmap:
        world_size = max(1, len(placement.rankmap))
    else:
        world_size = max(1, int(request.env.get("WORLD_SIZE", "1") or 1))
    honesty = request.honesty_level
    fabric_mode = request.fabric_mode or "auto"

    # Timeout inject: oversize sleep (VAL-FAB-014).
    forced: LaunchStatus | None = request.inject_status
    oversize = request.inject_sleep_s > float(request.timeout_s)
    if forced is None and oversize and request.timeout_s >= 0:
        forced = "timeout"

    spoof_codes = detect_inventory_spoof(
        fabric_mode=fabric_mode,
        reports=list(request.node_reports),
        inventory_spoof=request.inventory_spoof,
        spoofed_node_ids=list(request.spoofed_node_ids),
    )

    gate = evaluate_fabric_gate(
        fabric_mode=fabric_mode,
        reports=list(request.node_reports),
        eth_fallback_injected=request.eth_fallback_injected,
        inventory_spoof=bool(spoof_codes) or request.inventory_spoof,
        correctness_present=True,
    )
    if spoof_codes and "inventory_spoof" not in gate.reason_codes:
        # Extend reason codes if gate helper did not already tag spoof.
        gate = FabricGateResult(
            fabric_gate=0.0,
            composite_zeroed=True,
            reason_codes=list(dict.fromkeys([*gate.reason_codes, *spoof_codes])),
            required_transport=gate.required_transport,
            actual_transport=gate.actual_transport or "spoofed",
        )

    # Explicit failure inject.
    if forced == "failed":
        return LaunchResult(
            status="failed",
            metrics=None,
            fabric_artifact_digest=build_fabric_artifact_digest(
                placement=placement, metrics=None, status="failed"
            ),
            rank_progress_digests=[],
            nccl_debug_excerpt_digest=build_nccl_debug_excerpt_digest(
                world_size=world_size,
                fabric_mode=fabric_mode,
                status="failed",
                seed=request.seed,
            ),
            failure_code="sim_launch_fail",
            reason="sim inject failed",
            fabric_gate=0.0,
            composite=0.0,
            score_factors=summarize_gate_for_score(
                FabricGateResult(
                    fabric_gate=0.0,
                    composite_zeroed=True,
                    reason_codes=["sim_launch_fail"],
                ),
                correctness=0.0,
                efficiency=0.0,
            ),
            world_size=world_size,
            honesty_level=honesty,
            integrity_fail=False,
        )

    if forced == "timeout":
        return LaunchResult(
            status="timeout",
            metrics=None,
            fabric_artifact_digest=build_fabric_artifact_digest(
                placement=placement, metrics=None, status="timeout"
            ),
            rank_progress_digests=[],
            nccl_debug_excerpt_digest=build_nccl_debug_excerpt_digest(
                world_size=world_size,
                fabric_mode=fabric_mode,
                status="timeout",
                seed=request.seed,
            ),
            failure_code="timeout",
            reason="sim inject timeout / oversize sleep",
            fabric_gate=gate.fabric_gate if not gate.composite_zeroed else 0.0,
            composite=0.0,
            score_factors=summarize_gate_for_score(
                gate, correctness=0.0, efficiency=0.0
            ),
            world_size=world_size,
            honesty_level=honesty,
            integrity_fail=False,
        )

    # Success path — still subject to honesty / spoof zero composite.
    metrics = synthetic_allreduce_metrics(
        world_size=world_size,
        fabric_mode=fabric_mode,
        noise=0.0,
        seed=request.seed,
    )
    # L1 honesty: off-band metrics after success are still recorded but gate may zero.
    if not metrics.within_band:
        gate = FabricGateResult(
            fabric_gate=0.0,
            composite_zeroed=True,
            reason_codes=list(dict.fromkeys([*gate.reason_codes, "allreduce_out_of_band"])),
            required_transport=gate.required_transport,
            actual_transport=gate.actual_transport,
        )

    integrity_fail = bool(gate.composite_zeroed or spoof_codes)
    factors = summarize_gate_for_score(
        gate,
        correctness=0.0 if integrity_fail and request.inventory_spoof else 1.0,
        efficiency=metrics.efficiency if metrics.within_band else 0.0,
        tee_bonus=1.0,
    )
    # VAL-FAB-025: spoof → fabric_gate 0 and composite 0 always.
    if integrity_fail:
        factors["fabric_gate"] = 0.0
        factors["composite"] = 0.0

    rank_digests: list[str] = []
    if honesty == "l2":
        job_label = "sim"
        if placement.rankmap and placement.rankmap[0].job_id:
            job_label = str(placement.rankmap[0].job_id)
        rank_digests = build_rank_progress_digests(
            world_size=world_size,
            job_id=job_label,
            seed=request.seed,
            rankmap=list(placement.rankmap),
        )
        if len(rank_digests) != world_size:
            # Missing ranks fail honesty (VAL-FAB-015).
            integrity_fail = True
            factors["fabric_gate"] = 0.0
            factors["composite"] = 0.0
            factors.setdefault("reason_codes", []).append("l2_rank_digest_length_mismatch")

    artifact = build_fabric_artifact_digest(
        placement=placement, metrics=metrics, status="succeeded"
    )
    nccl_dbg = build_nccl_debug_excerpt_digest(
        world_size=world_size,
        fabric_mode=fabric_mode,
        status="succeeded",
        seed=request.seed,
    )

    status: LaunchStatus = "succeeded"
    failure_code: str | None = None
    reason = "sim multi-node launch ok"
    if integrity_fail and request.inventory_spoof:
        # Spoof still "runs" but integrity fails for scoring (status can be
        # succeeded operationally with zero composite, or failed integrity).
        # Prefer succeeded + zero gate so metrics remain, matching VAL-FAB-025
        # "cheat/integrity fail → fabric_gate 0 and composite 0".
        failure_code = "inventory_spoof"
        reason = "inventory spoof detected; fabric honesty fail-closed"
        status = "succeeded"  # operational complete; honesty via score factors

    return LaunchResult(
        status=status,
        metrics=metrics,
        fabric_artifact_digest=artifact,
        rank_progress_digests=rank_digests,
        nccl_debug_excerpt_digest=nccl_dbg,
        failure_code=failure_code,
        reason=reason,
        fabric_gate=float(factors["fabric_gate"]),
        composite=float(factors["composite"]),
        score_factors=factors,
        world_size=world_size,
        honesty_level=honesty,
        integrity_fail=integrity_fail,
    )


def placement_result_from_dicts(
    *,
    rankmap: list[dict[str, Any]],
    nccl_env: dict[str, str],
    planner_version: str = "fabric-planner.v1",
    graph_digest: str = "",
    job_id: str = "sim",
    ok: bool = True,
) -> PlacementResult:
    """Build a PlacementResult from persisted job placement rows (lifecycle wire)."""

    from hypercluster.fabric.planner import PlacementResult as PR
    from hypercluster.fabric.planner import RankBinding

    bindings = [
        RankBinding(
            rank=int(r["rank"]),
            node_id=str(r["node_id"]),
            local_rank=int(r["local_rank"]),
            gpu_index=int(r.get("gpu_index", r["local_rank"])),
            job_id=job_id,
        )
        for r in rankmap
    ]
    if not graph_digest:
        graph_digest = _digest_payload("graph", {"rankmap": rankmap, "job_id": job_id})
    return PR(
        ok=ok,
        rankmap=bindings,
        nccl_env=dict(nccl_env),
        planner_version=planner_version,
        graph_digest=graph_digest,
        reason="from_dicts",
    )


__all__ = [
    "ALLREDUCE_BASE_GBPS",
    "ALLREDUCE_BYTES",
    "ALLREDUCE_TOLERANCE_FRAC",
    "LAUNCHER_VERSION",
    "EfficiencyMetrics",
    "LaunchRequest",
    "LaunchResult",
    "allreduce_within_band",
    "build_fabric_artifact_digest",
    "build_nccl_debug_excerpt_digest",
    "build_rank_progress_digests",
    "detect_inventory_spoof",
    "expected_allreduce_gbps",
    "placement_result_from_dicts",
    "sim_launch",
    "synthetic_allreduce_metrics",
]
