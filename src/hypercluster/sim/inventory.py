"""Synthetic multi-node IB/NVLink inventory for local sim (architecture §12.2).

Fulfills VAL-FAB-019: seeded inventory enables multi-node plan without hardware.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from hypercluster.fabric.discovery import (
    DIGEST_PREFIX,
    FabricReport,
    build_fabric_report,
    canonical_json,
    synthetic_ib_devices,
    synthetic_nvlink_topo_matrix,
)


@dataclass(slots=True)
class SimNode:
    """Virtual provider node with synthetic fabric."""

    node_id: str
    hostname: str
    gpu_model: str
    gpu_count: int
    fabric_report: FabricReport
    nvlink_pairs: list[tuple[int, int]] = field(default_factory=list)
    location_hint: str = "sim"

    def to_public(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "location_hint": self.location_hint,
            "nvlink_pairs": [list(p) for p in self.nvlink_pairs],
            "fabric_report": self.fabric_report.to_public(),
            "report_digest": self.fabric_report.report_digest,
        }


@dataclass(slots=True)
class SimInventory:
    """N virtual nodes + IB/NVLink edges (VAL-FAB-019)."""

    nodes: list[SimNode]
    ib_edges: list[dict[str, Any]]
    nvlink_edges: list[dict[str, Any]]
    graph_digest: str
    seed: int = 0

    def to_public(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "node_count": len(self.nodes),
            "nodes": [n.to_public() for n in self.nodes],
            "ib_edges": list(self.ib_edges),
            "nvlink_edges": list(self.nvlink_edges),
            "graph_digest": self.graph_digest,
        }

    def reports(self) -> list[FabricReport]:
        return [n.fabric_report for n in self.nodes]


@dataclass(slots=True)
class PlanReadiness:
    """Lightweight multi-node placement readiness from sim inventory."""

    ok: bool
    rankmap: list[dict[str, Any]]
    reason: str = ""
    nnodes_used: int = 0


def _nvlink_pairs(gpu_count: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(gpu_count):
        for j in range(i + 1, gpu_count):
            pairs.append((i, j))
    return pairs


def _build_sim_node(
    *,
    index: int,
    gpus_per_node: int,
    seed: int,
    rate_gbps: float = 200.0,
    collected_at: datetime | None = None,
    topo_variant: str = "pack",
) -> SimNode:
    node_id = f"sim-node-{index}"
    matrix = synthetic_nvlink_topo_matrix(gpus_per_node)
    if topo_variant != "pack":
        # Perturb matrix so digest changes without hardware.
        matrix = matrix + f"#variant={topo_variant}:seed={seed}:node={index}\n"

    when = collected_at or datetime(2026, 1, 1, 0, 0, tzinfo=UTC).replace(
        # Fixed epoch day + seed so digests stable unless topo_variant changes.
        day=1 + (seed % 27),
        hour=index % 24,
    )
    # Keep collected_at stable for identical seed rescans.
    when = datetime(2026, 7, 1, index % 24, seed % 60, 0, tzinfo=UTC)

    report = build_fabric_report(
        node_id=node_id,
        collected_at=when,
        ib_devices=synthetic_ib_devices(
            node_index=index, count=1, rate_gbps=rate_gbps
        ),
        gpu_gpu_topo_matrix=matrix,
        numa_map={f"gpu{g}": g % 2 for g in range(gpus_per_node)},
        nccl_version="sim-2.21.5",
        eth_ifaces=["eth0", "lo"],
        gpu_count=gpus_per_node,
        source="sim",
    )
    return SimNode(
        node_id=node_id,
        hostname=f"sim-host-{index}.local",
        gpu_model="H100-SXM",
        gpu_count=gpus_per_node,
        fabric_report=report,
        nvlink_pairs=_nvlink_pairs(gpus_per_node),
        location_hint="sim-dc",
    )


def seed_sim_inventory(
    *,
    seed: int = 0,
    node_count: int = 4,
    gpus_per_node: int = 2,
    rate_gbps: float = 200.0,
    topo_variant: str = "pack",
) -> SimInventory:
    """Deterministic synthetic IB/NVLink multi-node inventory."""

    n = max(0, int(node_count))
    gpus = max(1, int(gpus_per_node))
    nodes = [
        _build_sim_node(
            index=i,
            gpus_per_node=gpus,
            seed=seed,
            rate_gbps=rate_gbps,
            topo_variant=topo_variant,
        )
        for i in range(n)
    ]

    # Fully connected IB fabric among nodes (multi-node edges).
    ib_edges: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ib_edges.append(
                {
                    "src": nodes[i].node_id,
                    "dst": nodes[j].node_id,
                    "rate_gbps": rate_gbps,
                    "medium": "ib",
                }
            )

    nvlink_edges: list[dict[str, Any]] = []
    for node in nodes:
        for a, b in node.nvlink_pairs:
            nvlink_edges.append(
                {
                    "node_id": node.node_id,
                    "gpu_a": a,
                    "gpu_b": b,
                    "medium": "nvlink",
                }
            )

    graph_body = {
        "seed": seed,
        "node_ids": [x.node_id for x in nodes],
        "ib_edges": ib_edges,
        "nvlink_edges": nvlink_edges,
        "report_digests": [x.fabric_report.report_digest for x in nodes],
    }
    graph_digest = DIGEST_PREFIX + hashlib.sha256(
        canonical_json(graph_body).encode()
    ).hexdigest()

    return SimInventory(
        nodes=nodes,
        ib_edges=ib_edges,
        nvlink_edges=nvlink_edges,
        graph_digest=graph_digest,
        seed=seed,
    )


def default_sim_inventory(*, seed: int = 0) -> SimInventory:
    """Mission default: 4 nodes × 2 GPUs with IB + NVLink."""

    return seed_sim_inventory(seed=seed, node_count=4, gpus_per_node=2)


def plan_readiness(
    inventory: SimInventory,
    *,
    world_size: int,
    nnodes: int,
    nproc_per_node: int,
) -> PlanReadiness:
    """Build a simple multi-node rankmap when inventory capacity is sufficient.

    Pure function for VAL-FAB-019 (scenario/planner inputs feed later M4 slices).
    """

    if world_size < 1 or nnodes < 1 or nproc_per_node < 1:
        return PlanReadiness(ok=False, rankmap=[], reason="invalid world/nnodes/nproc")
    if not inventory.nodes:
        return PlanReadiness(
            ok=False,
            rankmap=[],
            reason="no topology: sim inventory is empty",
        )
    if world_size > nnodes * nproc_per_node:
        return PlanReadiness(
            ok=False,
            rankmap=[],
            reason="world_size exceeds nnodes * nproc_per_node",
        )

    # Select nodes that have enough GPUs for nproc_per_node.
    capable = [n for n in inventory.nodes if n.gpu_count >= nproc_per_node]
    if len(capable) < nnodes:
        return PlanReadiness(
            ok=False,
            rankmap=[],
            reason=(
                f"no topology: need {nnodes} nodes with >= {nproc_per_node} GPUs, "
                f"have {len(capable)}"
            ),
        )

    selected = capable[:nnodes]
    rankmap: list[dict[str, Any]] = []
    rank = 0
    for node in selected:
        for local in range(nproc_per_node):
            if rank >= world_size:
                break
            rankmap.append(
                {
                    "rank": rank,
                    "node_id": node.node_id,
                    "local_rank": local,
                    "gpu_index": local,
                }
            )
            rank += 1

    if len(rankmap) != world_size:
        return PlanReadiness(
            ok=False,
            rankmap=rankmap,
            reason="unable to bind full world_size on selected topology",
        )

    return PlanReadiness(
        ok=True,
        rankmap=rankmap,
        reason="ready",
        nnodes_used=len({b["node_id"] for b in rankmap}),
    )


__all__ = [
    "PlanReadiness",
    "SimInventory",
    "SimNode",
    "default_sim_inventory",
    "plan_readiness",
    "seed_sim_inventory",
]
