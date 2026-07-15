"""VAL-FAB-004/005/006/007/008/009/020/022: topology-aware pack/spread planner."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hypercluster.fabric.discovery import FabricReport, IbDevice, build_fabric_report
from hypercluster.fabric.gates import PLANNER_NCCL_ENV_VERSION
from hypercluster.fabric.planner import (
    PLANNER_VERSION,
    PlacementRequest,
    PlacementResult,
    RankBinding,
    place_ranks,
)
from hypercluster.sim.inventory import seed_sim_inventory


def _ib_multi_gpu(
    node_id: str,
    *,
    gpu_count: int,
    rate_gbps: float = 200.0,
) -> FabricReport:
    # Synthetic NVLink-dense matrix for dense multi-GPU claims.
    header = "GPU\t" + "\t".join(f"GPU{i}" for i in range(gpu_count))
    rows = [header]
    for i in range(gpu_count):
        cells = []
        for j in range(gpu_count):
            if i == j:
                cells.append("X")
            else:
                cells.append("NV12")
        rows.append(f"GPU{i}\t" + "\t".join(cells))
    matrix = "\n".join(rows) + "\n"
    return build_fabric_report(
        node_id=node_id,
        collected_at=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
        ib_devices=[IbDevice(name="mlx5_0", port=1, rate_gbps=rate_gbps, state="Active")],
        eth_ifaces=["eth0", "lo"],
        gpu_count=gpu_count,
        gpu_gpu_topo_matrix=matrix,
        numa_map={f"gpu{g}": g % 2 for g in range(gpu_count)},
        source="sim",
    )


def _four_node_reports(gpus: int = 2) -> list[FabricReport]:
    return [_ib_multi_gpu(f"node-{i}", gpu_count=gpus) for i in range(4)]


# ----- VAL-FAB-004 pack -----------------------------------------------------


def test_pack_fills_fewest_nodes_for_dense_world() -> None:
    """VAL-FAB-004: pack concentrates ranks on minimal node set."""

    # 2 GPUs on first node is enough for world_size=2 — pack must not use 4 nodes.
    reports = _four_node_reports(gpus=2)
    result = place_ranks(
        PlacementRequest(
            job_id="job-pack-dense",
            world_size=2,
            nnodes=4,
            nproc_per_node=2,
            policy="pack",
            fabric="auto",
            node_reports=reports,
        )
    )
    assert isinstance(result, PlacementResult)
    assert result.ok is True
    node_ids = {b.node_id for b in result.rankmap}
    assert len(node_ids) == 1, f"pack should use 1 node, got {node_ids}"


def test_pack_uses_two_nodes_when_one_insufficient() -> None:
    reports = [
        _ib_multi_gpu("small-a", gpu_count=2),
        _ib_multi_gpu("small-b", gpu_count=2),
        _ib_multi_gpu("small-c", gpu_count=2),
        _ib_multi_gpu("small-d", gpu_count=2),
    ]
    result = place_ranks(
        PlacementRequest(
            job_id="job-pack-2",
            world_size=4,
            nnodes=4,
            nproc_per_node=2,
            policy="pack",
            fabric="ib",
            node_reports=reports,
        )
    )
    assert result.ok is True
    node_ids = {b.node_id for b in result.rankmap}
    assert len(node_ids) == 2


# ----- VAL-FAB-005 spread ---------------------------------------------------


def test_spread_distributes_across_more_nodes_than_pack() -> None:
    """VAL-FAB-005: spread uses ≥ pack unique nodes on identical capacity."""

    # world_size=4, nproc_per_node=2, 4×2 GPU inventory:
    # pack fills 2 ranks/node → 2 nodes; spread round-robins → up to 4 nodes.
    reports = _four_node_reports(gpus=2)
    pack = place_ranks(
        PlacementRequest(
            job_id="job-pol",
            world_size=4,
            nnodes=4,
            nproc_per_node=2,
            policy="pack",
            fabric="auto",
            node_reports=reports,
        )
    )
    spread = place_ranks(
        PlacementRequest(
            job_id="job-pol",
            world_size=4,
            nnodes=4,
            nproc_per_node=2,
            policy="spread",
            fabric="auto",
            node_reports=reports,
        )
    )
    assert pack.ok and spread.ok
    pack_nodes = {b.node_id for b in pack.rankmap}
    spread_nodes = {b.node_id for b in spread.rankmap}
    assert len(spread_nodes) >= len(pack_nodes)
    assert len(pack_nodes) == 2
    assert len(spread_nodes) == 4
    assert len(pack_nodes) < len(spread_nodes)


# ----- VAL-FAB-006 / VAL-FAB-007 rankmap invariants -------------------------


def test_rankmap_covers_world_exactly_once() -> None:
    """VAL-FAB-006: ranks 0..world_size-1 exactly once with valid bindings."""

    inv = seed_sim_inventory(seed=7, node_count=4, gpus_per_node=2)
    world = 6
    result = place_ranks(
        PlacementRequest(
            job_id="job-ranks",
            world_size=world,
            nnodes=4,
            nproc_per_node=2,
            policy="pack",
            fabric="ib",
            node_reports=inv.reports(),
        )
    )
    assert result.ok is True
    ranks = [b.rank for b in result.rankmap]
    assert sorted(ranks) == list(range(world))
    assert len(set(ranks)) == world
    for b in result.rankmap:
        assert isinstance(b, RankBinding)
        assert b.node_id
        assert b.local_rank >= 0
        assert b.gpu_index >= 0


def test_local_ranks_consecutive_per_node() -> None:
    """VAL-FAB-007: per-node local_rank is 0..n-1 without holes."""

    reports = _four_node_reports(gpus=4)
    result = place_ranks(
        PlacementRequest(
            job_id="job-local",
            world_size=8,
            nnodes=4,
            nproc_per_node=4,
            policy="pack",
            fabric="auto",
            node_reports=reports,
        )
    )
    assert result.ok is True
    by_node: dict[str, list[int]] = {}
    for b in result.rankmap:
        by_node.setdefault(b.node_id, []).append(b.local_rank)
    for node_id, locals_ in by_node.items():
        sorted_locals = sorted(locals_)
        assert sorted_locals == list(range(len(sorted_locals))), (
            f"{node_id}: local ranks {sorted_locals} not consecutive"
        )


# ----- VAL-FAB-008 multi-node NCCL env --------------------------------------


def test_nccl_env_has_required_multi_node_keys_for_ib() -> None:
    """VAL-FAB-008: multi-node plan emits MASTER_* + IB/async keys."""

    reports = _four_node_reports(gpus=2)
    result = place_ranks(
        PlacementRequest(
            job_id="job-nccl",
            world_size=4,
            nnodes=2,
            nproc_per_node=2,
            policy="pack",
            fabric="ib",
            node_reports=reports,
            master_addr="10.0.0.1",
            master_port="29501",
        )
    )
    assert result.ok is True
    env = result.nccl_env
    assert env["MASTER_ADDR"] == "10.0.0.1"
    assert env["MASTER_PORT"] == "29501"
    assert env.get("TORCH_NCCL_ASYNC_ERROR_HANDLING") == "1"
    assert "NCCL_SOCKET_IFNAME" in env
    assert env.get("NCCL_NET") == "IB"
    assert env.get("NCCL_IB_HCA")
    assert result.nnodes_used >= 1


def test_nccl_env_eth_keys_without_ib() -> None:
    reports = [
        build_fabric_report(
            node_id="e0",
            collected_at=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
            ib_devices=[],
            eth_ifaces=["eth0"],
            gpu_count=2,
            source="sim",
        ),
        build_fabric_report(
            node_id="e1",
            collected_at=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
            ib_devices=[],
            eth_ifaces=["eth0"],
            gpu_count=2,
            source="sim",
        ),
    ]
    result = place_ranks(
        PlacementRequest(
            job_id="job-eth",
            world_size=2,
            nnodes=2,
            nproc_per_node=1,
            policy="spread",
            fabric="eth",
            node_reports=reports,
        )
    )
    assert result.ok is True
    assert result.nccl_env.get("NCCL_NET") != "IB"
    assert "NCCL_IB_HCA" not in result.nccl_env


# ----- VAL-FAB-009 planner_version / graph_digest stable --------------------


def test_planner_version_and_graph_digest_stable() -> None:
    """VAL-FAB-009: identical request → same planner_version + graph_digest."""

    reports = _four_node_reports(gpus=2)
    req = PlacementRequest(
        job_id="job-stable",
        world_size=4,
        nnodes=2,
        nproc_per_node=2,
        policy="pack",
        fabric="ib",
        node_reports=reports,
    )
    a = place_ranks(req)
    b = place_ranks(req)
    assert a.ok and b.ok
    assert a.planner_version == PLANNER_VERSION
    assert a.planner_version.startswith("fabric-planner.")
    assert a.graph_digest == b.graph_digest
    assert a.graph_digest.startswith("sha256:")
    # Rankmaps identical (deterministic).
    assert [x.model_dump() for x in a.rankmap] == [x.model_dump() for x in b.rankmap]


# ----- VAL-FAB-020 nccl_env.v1 fixture parity --------------------------------


def test_nccl_env_v1_fixture_parity() -> None:
    """VAL-FAB-020: golden nccl_env.v1 keys/critical values match fixture."""

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "fabric"
        / "nccl_env.v1.json"
    )
    assert fixture_path.is_file(), f"missing golden fixture {fixture_path}"
    golden = json.loads(fixture_path.read_text(encoding="utf-8"))

    reports = _four_node_reports(gpus=2)
    result = place_ranks(
        PlacementRequest(
            job_id=golden["job_id"],
            world_size=golden["world_size"],
            nnodes=golden["nnodes"],
            nproc_per_node=golden["nproc_per_node"],
            policy=golden["policy"],
            fabric=golden["fabric"],
            node_reports=reports,
            master_addr=golden["master_addr"],
            master_port=golden["master_port"],
            socket_ifname=golden.get("socket_ifname", "lo"),
            backend=golden.get("backend", "nccl"),
        )
    )
    assert result.ok is True
    assert PLANNER_NCCL_ENV_VERSION == "nccl_env.v1"
    assert result.nccl_env.get("HYPER_NCCL_ENV_VERSION") == "nccl_env.v1"

    expected = golden["nccl_env"]
    # Critical keys + values must match; ignore dynamic-only if documented.
    critical = golden.get(
        "critical_keys",
        [
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "NCCL_SOCKET_IFNAME",
            "NCCL_NET",
            "NCCL_IB_HCA",
            "NCCL_IB_GID_INDEX",
            "HYPER_NCCL_ENV_VERSION",
            "HYPER_FABRIC_MODE",
            "HYPER_BACKEND",
        ],
    )
    for key in critical:
        assert key in result.nccl_env, f"missing key {key}"
        assert result.nccl_env[key] == expected[key], (
            f"{key}: got {result.nccl_env[key]!r} expected {expected[key]!r}"
        )


# ----- VAL-FAB-022 nvlink_only dense ----------------------------------------


def test_nvlink_only_prefers_intra_node_dense_gpus() -> None:
    """VAL-FAB-022: nvlink_only pack places all ranks on one dense node."""

    dense = _ib_multi_gpu("dense-nvlink", gpu_count=8)
    sparse = [
        _ib_multi_gpu("s0", gpu_count=2),
        _ib_multi_gpu("s1", gpu_count=2),
        _ib_multi_gpu("s2", gpu_count=2),
        _ib_multi_gpu("s3", gpu_count=2),
    ]
    reports = [dense, *sparse]
    result = place_ranks(
        PlacementRequest(
            job_id="job-nvl",
            world_size=8,
            nnodes=4,
            nproc_per_node=8,
            policy="pack",
            fabric="nvlink_only",
            node_reports=reports,
        )
    )
    assert result.ok is True
    node_ids = {b.node_id for b in result.rankmap}
    assert node_ids == {"dense-nvlink"}
    assert result.nccl_env.get("NCCL_P2P_LEVEL") == "NVL" or result.nccl_env.get(
        "NCCL_NET"
    ) != "IB"


def test_place_ranks_fails_closed_when_capacity_insufficient() -> None:
    reports = [_ib_multi_gpu("only", gpu_count=1)]
    result = place_ranks(
        PlacementRequest(
            job_id="job-short",
            world_size=8,
            nnodes=2,
            nproc_per_node=4,
            policy="pack",
            fabric="auto",
            node_reports=reports,
        )
    )
    assert result.ok is False
    assert result.rankmap == []
    assert result.failure_code is not None


def test_placement_request_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError):
        PlacementRequest(
            job_id="x",
            world_size=1,
            nnodes=1,
            nproc_per_node=1,
            policy="round_robin",  # type: ignore[arg-type]
            fabric="auto",
            node_reports=[],
        )
