"""VAL-FAB-013/014/015/025: multi-node launcher results, L1 metrics, L2 digests, spoof zero."""

from __future__ import annotations

from datetime import UTC, datetime

from hypercluster.fabric.discovery import IbDevice, build_fabric_report
from hypercluster.fabric.gates import evaluate_fabric_gate, summarize_gate_for_score
from hypercluster.fabric.launcher import (
    ALLREDUCE_TOLERANCE_FRAC,
    LaunchRequest,
    LaunchResult,
    allreduce_within_band,
    build_rank_progress_digests,
    expected_allreduce_gbps,
    sim_launch,
    synthetic_allreduce_metrics,
)
from hypercluster.fabric.planner import PlacementRequest, place_ranks
from hypercluster.sim.inventory import seed_sim_inventory


def _placement(*, world_size: int = 4, nnodes: int = 2) -> object:
    inv = seed_sim_inventory(seed=7, node_count=4, gpus_per_node=2)
    result = place_ranks(
        PlacementRequest(
            job_id="job-launch-test",
            world_size=world_size,
            nnodes=nnodes,
            nproc_per_node=2,
            policy="pack",
            fabric="auto",
            node_reports=inv.reports(),
        )
    )
    assert result.ok
    return result


# ----- VAL-FAB-013 synthetic allreduce metrics (L1) -------------------------


def test_synthetic_allreduce_metrics_present_and_within_band() -> None:
    """VAL-FAB-013: metrics_json shape for success path within golden band."""

    placement = _placement(world_size=4)
    req = LaunchRequest(
        placement=placement,  # type: ignore[arg-type]
        image_digest="sha256:sim000000000000000000000000000000000000000000000000000000000001",
        entrypoint=["python", "-m", "train"],
        fabric_mode="auto",
        honesty_level="l1",
        seed=3,
    )
    result = sim_launch(req)
    assert result.status == "succeeded"
    assert result.metrics is not None
    mj = result.metrics_json()
    assert "allreduce_gbps" in mj
    assert mj["allreduce_gbps"] > 0
    assert mj["source"] == "sim_launcher"
    assert mj["within_band"] is True
    expected = expected_allreduce_gbps(world_size=4, fabric_mode="auto")
    assert allreduce_within_band(mj["allreduce_gbps"], expected)
    # Controllable small noise still in band.
    noisy = synthetic_allreduce_metrics(
        world_size=4, fabric_mode="auto", noise=ALLREDUCE_TOLERANCE_FRAC * 0.5, seed=0
    )
    assert noisy.within_band is True


def test_allreduce_metrics_fail_when_no_metrics_would_be_absent_on_success_path() -> None:
    """VAL-FAB-013: successful multi-node launch always attaches metrics."""

    placement = _placement(world_size=2)
    result = sim_launch(
        LaunchRequest(
            placement=placement,  # type: ignore[arg-type]
            image_digest="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            honesty_level="l1",
        )
    )
    assert result.status == "succeeded"
    assert result.metrics is not None
    assert result.fabric_artifact_digest.startswith("sha256:")


# ----- VAL-FAB-014 LaunchResult status enumerations -------------------------


def test_launch_result_success_failed_timeout_injects() -> None:
    """VAL-FAB-014: three inject paths map to succeeded|failed|timeout."""

    placement = _placement(world_size=2)
    image = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    ok = sim_launch(
        LaunchRequest(
            placement=placement,  # type: ignore[arg-type]
            image_digest=image,
            inject_status="succeeded",
        )
    )
    assert ok.status == "succeeded"
    assert ok.metrics is not None
    assert ok.fabric_artifact_digest
    assert isinstance(ok, LaunchResult)

    fail = sim_launch(
        LaunchRequest(
            placement=placement,  # type: ignore[arg-type]
            image_digest=image,
            inject_status="failed",
        )
    )
    assert fail.status == "failed"
    assert fail.failure_code == "sim_launch_fail"

    timed = sim_launch(
        LaunchRequest(
            placement=placement,  # type: ignore[arg-type]
            image_digest=image,
            inject_status="timeout",
        )
    )
    assert timed.status == "timeout"
    assert timed.failure_code == "timeout"

    # Oversize sleep vs timeout_s also → timeout (inject path).
    oversize = sim_launch(
        LaunchRequest(
            placement=placement,  # type: ignore[arg-type]
            image_digest=image,
            timeout_s=1,
            inject_sleep_s=5.0,
        )
    )
    assert oversize.status == "timeout"


# ----- VAL-FAB-015 L2 rank progress digests ---------------------------------


def test_l2_rank_progress_digests_match_world_size() -> None:
    """VAL-FAB-015: when L2 enabled, digests length == world_size."""

    world_size = 4
    placement = _placement(world_size=world_size)
    result = sim_launch(
        LaunchRequest(
            placement=placement,  # type: ignore[arg-type]
            image_digest="sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            honesty_level="l2",
            seed=11,
        )
    )
    assert result.status == "succeeded"
    assert len(result.rank_progress_digests) == world_size
    assert all(d.startswith("sha256:") for d in result.rank_progress_digests)
    # Distinct per rank.
    assert len(set(result.rank_progress_digests)) == world_size

    digests = build_rank_progress_digests(
        world_size=world_size,
        job_id="job-launch-test",
        seed=11,
        rankmap=list(placement.rankmap),  # type: ignore[union-attr]
    )
    assert len(digests) == world_size


def test_l1_path_does_not_require_rank_digests() -> None:
    """L1 honesty does not attach L2 digests (length may be 0)."""

    placement = _placement(world_size=2)
    result = sim_launch(
        LaunchRequest(
            placement=placement,  # type: ignore[arg-type]
            image_digest="sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            honesty_level="l1",
        )
    )
    assert result.rank_progress_digests == []


# ----- VAL-FAB-025 inventory spoof → composite 0 ----------------------------


def test_inventory_spoof_zeros_fabric_gate_and_composite() -> None:
    """VAL-FAB-025: spoofed IB inventory integrity fails with composite 0."""

    inv = seed_sim_inventory(seed=0, node_count=2, gpus_per_node=2)
    reports = inv.reports()
    # Claims of IB exist...
    assert all(r.ib_devices for r in reports)
    placement = place_ranks(
        PlacementRequest(
            job_id="job-spoof",
            world_size=2,
            nnodes=2,
            nproc_per_node=1,
            policy="pack",
            fabric="ib",
            node_reports=reports,
        )
    )
    assert placement.ok

    result = sim_launch(
        LaunchRequest(
            placement=placement,
            image_digest="sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            fabric_mode="ib",
            honesty_level="l1",
            inventory_spoof=True,
            node_reports=reports,
        )
    )
    # Operational may complete, but honesty zeros gate product.
    assert result.fabric_gate == 0.0
    assert result.composite == 0.0
    assert result.integrity_fail is True
    factors = result.score_factors
    assert factors["fabric_gate"] == 0.0
    assert factors["composite"] == 0.0
    assert "inventory_spoof" in (factors.get("reason_codes") or []) or result.failure_code == (
        "inventory_spoof"
    )

    # Direct gate helper also zeros (reuse by scoring).
    gate = evaluate_fabric_gate(
        fabric_mode="ib",
        reports=reports,
        inventory_spoof=True,
        correctness_present=True,
    )
    assert gate.fabric_gate == 0.0
    assert gate.composite_zeroed is True
    preview = summarize_gate_for_score(gate, correctness=1.0, efficiency=0.9, tee_bonus=1.1)
    assert preview["composite"] == 0.0


def test_honest_ib_launch_keeps_fabric_gate_one() -> None:
    """Control: honest multi-node IB path keeps fabric_gate 1 for success."""

    inv = seed_sim_inventory(seed=1, node_count=2, gpus_per_node=2)
    placement = place_ranks(
        PlacementRequest(
            job_id="job-honest",
            world_size=2,
            nnodes=2,
            nproc_per_node=1,
            policy="pack",
            fabric="ib",
            node_reports=inv.reports(),
        )
    )
    assert placement.ok
    result = sim_launch(
        LaunchRequest(
            placement=placement,
            image_digest="sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            fabric_mode="ib",
            honesty_level="l1",
            inventory_spoof=False,
            node_reports=inv.reports(),
        )
    )
    assert result.status == "succeeded"
    assert result.fabric_gate == 1.0
    assert result.composite > 0.0
    assert result.integrity_fail is False


def test_spoofed_node_ids_list_also_triggers_honesty_fail() -> None:
    """Per-node spoof list (claimed devices that fail probe) zeros composite."""

    report = build_fabric_report(
        node_id="node-a",
        collected_at=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
        ib_devices=[IbDevice(name="mlx5_0", port=1, rate_gbps=200.0, state="Active")],
        eth_ifaces=["eth0"],
        gpu_count=2,
        source="sim",
    )
    placement = place_ranks(
        PlacementRequest(
            job_id="job-spoof-list",
            world_size=1,
            nnodes=1,
            nproc_per_node=1,
            policy="pack",
            fabric="auto",
            node_reports=[report],
        )
    )
    result = sim_launch(
        LaunchRequest(
            placement=placement,
            image_digest="sha256:1111111111111111111111111111111111111111111111111111111111111111",
            fabric_mode="auto",
            spoofed_node_ids=["node-a"],
            node_reports=[report],
        )
    )
    assert result.composite == 0.0
    assert result.fabric_gate == 0.0
