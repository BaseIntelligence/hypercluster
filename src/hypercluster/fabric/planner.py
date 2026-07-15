"""Topology-aware pack/spread planner and NCCL env matrix (architecture §8.2).

Fulfills:
  VAL-FAB-004  pack fills fewest nodes
  VAL-FAB-005  spread distributes across nodes
  VAL-FAB-006  rankmap covers 0..world_size-1 exactly once
  VAL-FAB-007  local ranks consecutive per node
  VAL-FAB-008  multi-node NCCL env keys
  VAL-FAB-009  planner_version + graph_digest stability
  VAL-FAB-020  nccl_env.v1 fixture parity
  VAL-FAB-022  nvlink_only prefers intra-node dense GPUs
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from hypercluster.fabric.discovery import DIGEST_PREFIX, FabricReport, canonical_json
from hypercluster.fabric.gates import (
    PLANNER_NCCL_ENV_VERSION,
    build_nccl_env_for_mode,
    evaluate_fabric_mode,
    has_active_ib_devices,
)

PLANNER_VERSION = "fabric-planner.v1"

PlacementPolicy = Literal["pack", "spread"]
FabricMode = Literal["auto", "ib", "eth", "nvlink_only"]


class RankBinding(BaseModel):
    """Single global rank → node/local GPU binding."""

    rank: int = Field(..., ge=0)
    node_id: str = Field(..., min_length=1)
    local_rank: int = Field(..., ge=0)
    gpu_index: int = Field(..., ge=0)
    job_id: str | None = None

    def to_public(self) -> dict[str, Any]:
        data = {
            "rank": self.rank,
            "node_id": self.node_id,
            "local_rank": self.local_rank,
            "gpu_index": self.gpu_index,
        }
        if self.job_id is not None:
            data["job_id"] = self.job_id
        return data


class PlacementRequest(BaseModel):
    """Inputs for topology-aware placement (architecture PlacementRequest)."""

    job_id: str = Field(..., min_length=1)
    world_size: int = Field(..., ge=1)
    nnodes: int = Field(..., ge=1)
    nproc_per_node: int = Field(..., ge=1)
    policy: PlacementPolicy = "pack"
    fabric: FabricMode = "auto"
    node_reports: list[FabricReport] = Field(default_factory=list)
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    socket_ifname: str = "lo"
    backend: str = "nccl"
    allow_partial_nodes: bool = True  # may use fewer than nnodes when pack fits

    @field_validator("policy")
    @classmethod
    def _policy_ok(cls, value: str) -> str:
        if value not in {"pack", "spread"}:
            raise ValueError("policy must be 'pack' or 'spread'")
        return value

    @field_validator("fabric")
    @classmethod
    def _fabric_ok(cls, value: str) -> str:
        if value not in {"auto", "ib", "eth", "nvlink_only"}:
            raise ValueError("fabric must be auto|ib|eth|nvlink_only")
        return value


class PlacementResult(BaseModel):
    """Placement output: rankmap + nccl_env + stable digests."""

    ok: bool
    rankmap: list[RankBinding] = Field(default_factory=list)
    nccl_env: dict[str, str] = Field(default_factory=dict)
    planner_version: str = PLANNER_VERSION
    graph_digest: str = ""
    nnodes_used: int = 0
    policy: PlacementPolicy = "pack"
    fabric: FabricMode = "auto"
    reason: str = ""
    failure_code: str | None = None

    def rankmap_public(self) -> list[dict[str, Any]]:
        return [b.to_public() for b in self.rankmap]


def _gpu_capacity(report: FabricReport) -> int:
    if report.gpu_count and report.gpu_count > 0:
        return int(report.gpu_count)
    # Heuristic from topo matrix line count when gpu_count unset.
    lines = [ln for ln in (report.gpu_gpu_topo_matrix or "").splitlines() if ln.strip()]
    # Header + N rows
    if len(lines) > 1:
        return max(0, len(lines) - 1)
    return 0


def _nvlink_density(report: FabricReport) -> float:
    """Higher = denser intra-node NVLink claims (matrix tokens)."""

    matrix = report.gpu_gpu_topo_matrix or ""
    if not matrix:
        return 0.0
    tokens = matrix.upper().count("NV")
    # SYS / PIX etc. are lower affinity.
    sys_tokens = matrix.upper().count("SYS") + matrix.upper().count("PIX")
    return float(tokens) - 0.1 * float(sys_tokens)


def _node_prefer_key(report: FabricReport, *, fabric: str) -> tuple[Any, ...]:
    """Sort key for preferring dense/capable nodes first (stable by node_id)."""

    gpus = _gpu_capacity(report)
    density = _nvlink_density(report)
    has_ib = 1 if has_active_ib_devices(report) else 0
    rate = float(report.ib_rate_gbps or 0.0)
    # Prefer: more GPUs, denser NVLink, IB when relevant, higher rate, then id.
    return (-gpus, -density, -has_ib if fabric in {"ib", "auto"} else 0, -rate, report.node_id)


def _eligible_reports(
    reports: list[FabricReport],
    *,
    fabric: str,
    min_gpus: int = 1,
) -> list[FabricReport]:
    """Filter reports that can participate under fabric mode."""

    out: list[FabricReport] = []
    for r in reports:
        if _gpu_capacity(r) < min_gpus:
            continue
        if fabric == "ib" and not has_active_ib_devices(r):
            continue
        if fabric == "nvlink_only" and _gpu_capacity(r) < 1:
            continue
        out.append(r)
    return out


def _bind_ranks_on_nodes(
    *,
    job_id: str,
    world_size: int,
    selected: list[tuple[FabricReport, int]],
    # selected: (report, ranks_on_node) in assignment order
) -> list[RankBinding]:
    """Assign global ranks in node order with consecutive local ranks 0..n-1."""

    bindings: list[RankBinding] = []
    rank = 0
    for report, n_local in selected:
        if n_local <= 0:
            continue
        # Cap by actual GPU count.
        cap = min(n_local, _gpu_capacity(report))
        for local in range(cap):
            if rank >= world_size:
                break
            bindings.append(
                RankBinding(
                    rank=rank,
                    node_id=report.node_id,
                    local_rank=local,
                    gpu_index=local,
                    job_id=job_id,
                )
            )
            rank += 1
        if rank >= world_size:
            break
    return bindings


def _plan_pack(
    eligible: list[FabricReport],
    *,
    world_size: int,
    nproc_per_node: int,
    max_nodes: int,
) -> list[tuple[FabricReport, int]] | None:
    """Fill fewest nodes: greedily take densest nodes first until world_size filled.

    Each selected node contributes min(nproc_per_node, gpu_capacity) ranks,
    filling that node before opening the next (architecture pack rule).
    """

    ordered = sorted(eligible, key=lambda r: _node_prefer_key(r, fabric="auto"))
    selected: list[tuple[FabricReport, int]] = []
    remaining = world_size
    for report in ordered:
        if len(selected) >= max_nodes:
            break
        if remaining <= 0:
            break
        cap = min(nproc_per_node, _gpu_capacity(report))
        if cap < 1:
            continue
        take = min(cap, remaining)
        # For density: if this node can take take-rank needs, take take.
        # Prefer fully packing when any remaining will still fit later — pack
        # always fills current node up to min(cap, remaining).
        selected.append((report, take))
        remaining -= take
    if remaining > 0:
        return None
    return selected


def _plan_spread(
    eligible: list[FabricReport],
    *,
    world_size: int,
    nproc_per_node: int,
    max_nodes: int,
) -> list[tuple[FabricReport, int]] | None:
    """Distribute ranks across as many nodes as capacity allows.

    Prefer one rank per node first (round-robin), then fill additional local
    ranks while respecting nproc_per_node and GPU capacity.
    """

    ordered = sorted(eligible, key=lambda r: (r.node_id,))  # stable alphabetical for determinism
    # Cap pool to max_nodes (caller may pass a larger inventory than job nnodes).
    pool = ordered[:max_nodes] if max_nodes < len(ordered) else list(ordered)
    if not pool:
        return None

    caps = {r.node_id: min(nproc_per_node, _gpu_capacity(r)) for r in pool}
    assigned: dict[str, int] = {r.node_id: 0 for r in pool}
    remaining = world_size

    # Phase 1: spread first rank across distinct nodes.
    while remaining > 0:
        progressed = False
        for report in pool:
            if remaining <= 0:
                break
            nid = report.node_id
            if assigned[nid] >= caps[nid]:
                continue
            # Prefer nodes with fewer assigned so far (true spread).
            min_assigned = min(assigned.values()) if assigned else 0
            if assigned[nid] > min_assigned:
                continue
            assigned[nid] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    if remaining > 0:
        return None

    # Preserve node order for rank numbering: nodes that received ranks, in pool order.
    selected: list[tuple[FabricReport, int]] = []
    for report in pool:
        n = assigned[report.node_id]
        if n > 0:
            selected.append((report, n))
    return selected if selected else None


def compute_placement_graph_digest(
    *,
    policy: str,
    fabric: str,
    world_size: int,
    nnodes: int,
    nproc_per_node: int,
    rankmap: list[RankBinding],
    report_digests: list[str],
    nccl_env_version: str = PLANNER_NCCL_ENV_VERSION,
) -> str:
    """Stable graph_digest for identical placement inputs + rank outcomes."""

    body = {
        "planner_version": PLANNER_VERSION,
        "policy": policy,
        "fabric": fabric,
        "world_size": world_size,
        "nnodes": nnodes,
        "nproc_per_node": nproc_per_node,
        "rankmap": [b.to_public() for b in rankmap],
        "report_digests": sorted(report_digests),
        "nccl_env_version": nccl_env_version,
    }
    return DIGEST_PREFIX + hashlib.sha256(canonical_json(body).encode()).hexdigest()


def place_ranks(request: PlacementRequest) -> PlacementResult:
    """Topology-aware pack/spread placement + NCCL env matrix.

    Pure function (deterministic given fixed node_reports order semantics).
    Does not require GPUs or live IB hardware.
    """

    fabric = request.fabric
    policy = request.policy
    reports = list(request.node_reports)

    if request.world_size > request.nnodes * request.nproc_per_node:
        return PlacementResult(
            ok=False,
            planner_version=PLANNER_VERSION,
            graph_digest=DIGEST_PREFIX + hashlib.sha256(b"invalid-dims").hexdigest(),
            policy=policy,
            fabric=fabric,
            reason="world_size exceeds nnodes * nproc_per_node",
            failure_code="invalid_world_dimensions",
        )

    # Fabric mode admission (ib fail-closed when zero devices) before bind.
    if reports:
        mode_eval = evaluate_fabric_mode(fabric_mode=fabric, reports=reports)
        if fabric == "ib" and not mode_eval.may_succeed:
            return PlacementResult(
                ok=False,
                planner_version=PLANNER_VERSION,
                graph_digest=DIGEST_PREFIX
                + hashlib.sha256(b"fabric-mode-reject").hexdigest(),
                policy=policy,
                fabric=fabric,
                reason=mode_eval.reason or "fabric mode rejected",
                failure_code=mode_eval.failure_code or "fabric_mode_rejected",
            )

    # For nvlink_only, ranks still allocate GPUs; prefer densest multi-GPU nodes.
    min_gpus = 1
    eligible = _eligible_reports(reports, fabric=fabric, min_gpus=min_gpus)
    if not eligible:
        # Fall back to all reports with GPU capacity when eth/auto and filtered empty.
        eligible = [r for r in reports if _gpu_capacity(r) >= 1]

    if not eligible:
        return PlacementResult(
            ok=False,
            planner_version=PLANNER_VERSION,
            graph_digest=DIGEST_PREFIX + hashlib.sha256(b"no-eligible-nodes").hexdigest(),
            policy=policy,
            fabric=fabric,
            reason="no eligible nodes with GPU capacity for placement",
            failure_code="capacity_insufficient",
        )

    max_nodes = min(request.nnodes, len(eligible))
    if policy == "pack":
        planned = _plan_pack(
            eligible,
            world_size=request.world_size,
            nproc_per_node=request.nproc_per_node,
            max_nodes=max_nodes if not request.allow_partial_nodes else len(eligible),
        )
        # When allow_partial_nodes, pack may use as few as 1 up to full inventory,
        # capped only by actual capacity (and nnodes budget as soft max).
        # Respect nnodes as upper bound even with allow_partial.
        if planned is not None and len(planned) > request.nnodes:
            # Re-run with hard max.
            planned = _plan_pack(
                eligible,
                world_size=request.world_size,
                nproc_per_node=request.nproc_per_node,
                max_nodes=request.nnodes,
            )
        # Also prefer max_nodes from request when packing into a declared budget.
        if planned is None:
            planned = _plan_pack(
                eligible,
                world_size=request.world_size,
                nproc_per_node=request.nproc_per_node,
                max_nodes=request.nnodes,
            )
    else:
        # Spread: use up to nnodes (or capacity), prefer distribute.
        spread_budget = request.nnodes
        # If inventory has more nodes than nnodes, still allow spread across nnodes;
        # if caller set high nnodes (capacity free), use up to inventory len for
        # true distribution (VAL-FAB-005 on multi-node free capacity).
        spread_budget = min(max(request.nnodes, 1), len(eligible))
        planned = _plan_spread(
            eligible,
            world_size=request.world_size,
            nproc_per_node=request.nproc_per_node,
            max_nodes=spread_budget,
        )

    if planned is None:
        return PlacementResult(
            ok=False,
            planner_version=PLANNER_VERSION,
            graph_digest=DIGEST_PREFIX
            + hashlib.sha256(b"capacity-insufficient").hexdigest(),
            policy=policy,
            fabric=fabric,
            reason="insufficient GPU capacity for world_size under policy",
            failure_code="capacity_insufficient",
        )

    rankmap = _bind_ranks_on_nodes(
        job_id=request.job_id,
        world_size=request.world_size,
        selected=planned,
    )
    if len(rankmap) != request.world_size:
        return PlacementResult(
            ok=False,
            rankmap=rankmap,
            planner_version=PLANNER_VERSION,
            graph_digest=DIGEST_PREFIX + hashlib.sha256(b"rank-bind-short").hexdigest(),
            policy=policy,
            fabric=fabric,
            reason="unable to bind full world_size",
            failure_code="rankmap_incomplete",
        )

    # Validate consecutive local ranks (invariant).
    by_node: dict[str, list[int]] = {}
    for b in rankmap:
        by_node.setdefault(b.node_id, []).append(b.local_rank)
    for nid, locals_ in by_node.items():
        if sorted(locals_) != list(range(len(locals_))):
            return PlacementResult(
                ok=False,
                planner_version=PLANNER_VERSION,
                graph_digest=DIGEST_PREFIX
                + hashlib.sha256(b"local-rank-hole").hexdigest(),
                policy=policy,
                fabric=fabric,
                reason=f"local ranks not consecutive on {nid}",
                failure_code="local_rank_invalid",
            )

    used_nodes = {b.node_id for b in rankmap}
    used_reports = [r for r in reports if r.node_id in used_nodes]
    nccl_env = build_nccl_env_for_mode(
        fabric_mode=fabric,
        reports=used_reports or reports,
        backend=request.backend,
        master_addr=request.master_addr,
        master_port=request.master_port,
        socket_ifname=request.socket_ifname,
    )

    digest = compute_placement_graph_digest(
        policy=policy,
        fabric=fabric,
        world_size=request.world_size,
        nnodes=request.nnodes,
        nproc_per_node=request.nproc_per_node,
        rankmap=rankmap,
        report_digests=[r.report_digest for r in reports],
    )

    return PlacementResult(
        ok=True,
        rankmap=rankmap,
        nccl_env=nccl_env,
        planner_version=PLANNER_VERSION,
        graph_digest=digest,
        nnodes_used=len(used_nodes),
        policy=policy,
        fabric=fabric,
        reason="ready",
    )


def rankmap_as_dicts(result: PlacementResult) -> list[dict[str, Any]]:
    """Serialize rankmap for JobPlacement.rankmap_json / launch contracts."""

    return result.rankmap_public()


__all__ = [
    "PLANNER_VERSION",
    "PlacementPolicy",
    "PlacementRequest",
    "PlacementResult",
    "RankBinding",
    "compute_placement_graph_digest",
    "place_ranks",
    "rankmap_as_dicts",
]
