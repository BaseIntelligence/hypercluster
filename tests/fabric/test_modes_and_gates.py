"""VAL-FAB-002/003/010/011/012/021/023: fabric modes, require_ib, eth fallback gates."""

from __future__ import annotations

from datetime import UTC, datetime

from hypercluster.fabric.discovery import (
    FabricReport,
    IbDevice,
    build_fabric_report,
)
from hypercluster.fabric.gates import (
    IB_RATE_MISMATCH_POLICY,
    ClusterFabricEvaluation,
    FabricGateResult,
    FabricModeEvaluation,
    RequireIbCheck,
    build_nccl_env_for_mode,
    evaluate_cluster_member_reports,
    evaluate_fabric_gate,
    evaluate_fabric_mode,
    evaluate_ib_rate_consistency,
    evaluate_require_ib_nodes,
    has_active_ib_devices,
    reports_by_node_id,
)
from hypercluster.sim.inventory import seed_sim_inventory


def _eth_report(node_id: str = "eth-node") -> FabricReport:
    return build_fabric_report(
        node_id=node_id,
        collected_at=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
        ib_devices=[],
        eth_ifaces=["eth0", "lo"],
        gpu_count=2,
        gpu_gpu_topo_matrix="GPU\tGPU0\tGPU1\nGPU0\tX\tSYS\nGPU1\tSYS\tX\n",
        source="sim",
    )


def _ib_report(
    node_id: str = "ib-node",
    *,
    rate_gbps: float = 200.0,
    devices: int = 1,
) -> FabricReport:
    return build_fabric_report(
        node_id=node_id,
        collected_at=datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC),
        ib_devices=[
            IbDevice(name=f"mlx5_{i}", port=1, rate_gbps=rate_gbps, state="Active")
            for i in range(devices)
        ],
        eth_ifaces=["eth0"],
        gpu_count=2,
        gpu_gpu_topo_matrix="GPU\tGPU0\tGPU1\nGPU0\tX\tNV12\nGPU1\tNV12\tX\n",
        source="sim",
    )


# ----- VAL-FAB-002 / VAL-FAB-003 --------------------------------------------


def test_has_active_ib_devices_detects_empty_and_nonempty() -> None:
    assert has_active_ib_devices(_eth_report()) is False
    assert has_active_ib_devices(_ib_report()) is True


def test_fabric_ib_zero_devices_fail_closed() -> None:
    """VAL-FAB-002: fabric=ib with empty ib_devices fails closed (placement/gate)."""

    result = evaluate_fabric_mode(
        fabric_mode="ib",
        reports=[_eth_report("n1"), _eth_report("n2")],
    )
    assert isinstance(result, FabricModeEvaluation)
    assert result.ok is False
    assert result.fabric_gate == 0.0
    assert result.may_succeed is False
    assert "missing IB" in result.reason or "ib" in result.reason.lower()
    assert result.failure_code in {
        "missing_ib",
        "fabric_ib_required",
        "ib_devices_missing",
    }


def test_fabric_ib_with_devices_ok() -> None:
    result = evaluate_fabric_mode(
        fabric_mode="ib",
        reports=[_ib_report("n1"), _ib_report("n2")],
    )
    assert result.ok is True
    assert result.fabric_gate == 1.0
    assert result.may_succeed is True
    assert result.resolved_transport == "ib"


def test_fabric_auto_ok_on_eth_only() -> None:
    """VAL-FAB-003: fabric=auto on eth/sim succeeds without requiring IB."""

    result = evaluate_fabric_mode(
        fabric_mode="auto",
        reports=[_eth_report("n1")],
    )
    assert result.ok is True
    assert result.may_succeed is True
    assert result.fabric_gate == 1.0
    assert result.resolved_transport in {"eth", "socket", "auto-eth"}


def test_fabric_auto_prefers_ib_when_present() -> None:
    result = evaluate_fabric_mode(
        fabric_mode="auto",
        reports=[_ib_report("n1"), _ib_report("n2")],
    )
    assert result.ok is True
    assert result.resolved_transport == "ib"
    assert result.fabric_gate == 1.0


def test_fabric_eth_ok_without_ib() -> None:
    result = evaluate_fabric_mode(fabric_mode="eth", reports=[_eth_report()])
    assert result.ok is True
    assert result.may_succeed is True
    assert result.resolved_transport == "eth"


# ----- VAL-FAB-021 ----------------------------------------------------------


def test_eth_mode_nccl_env_does_not_set_nccl_net_ib() -> None:
    """VAL-FAB-021: eth fabric must not force NCCL_NET=IB / IB_HCA as sole transport."""

    env = build_nccl_env_for_mode(
        fabric_mode="eth",
        reports=[_eth_report("n1")],
    )
    assert env.get("NCCL_NET") != "IB"
    assert "NCCL_IB_HCA" not in env
    assert env.get("HYPER_FABRIC_MODE") == "eth"
    # eth uses sockets
    assert env.get("NCCL_NET") in {"Socket", "socket", "SOCKET", None} or env.get(
        "NCCL_NET", ""
    ).lower() != "ib"


def test_ib_mode_nccl_env_sets_net_ib_when_devices_present() -> None:
    env = build_nccl_env_for_mode(
        fabric_mode="ib",
        reports=[_ib_report("n1", rate_gbps=400.0)],
    )
    assert env.get("NCCL_NET") == "IB"
    assert "NCCL_IB_HCA" in env
    assert env["NCCL_IB_HCA"]


def test_auto_mode_nccl_env_eth_path_no_ib_force() -> None:
    env = build_nccl_env_for_mode(fabric_mode="auto", reports=[_eth_report()])
    assert env.get("NCCL_NET") != "IB"
    assert "NCCL_IB_HCA" not in env


# ----- VAL-FAB-012 ----------------------------------------------------------


def test_forbidden_eth_fallback_under_ib_zeroes_fabric_gate() -> None:
    """VAL-FAB-012: eth fallback under fabric=ib zeroes fabric_gate even if correct."""

    result = evaluate_fabric_gate(
        fabric_mode="ib",
        required_transport="ib",
        actual_transport="eth",
        reports=[_ib_report()],  # inventory claimed IB but runtime fell back
        eth_fallback_injected=True,
        correctness_present=True,
    )
    assert isinstance(result, FabricGateResult)
    assert result.fabric_gate == 0.0
    assert result.composite_zeroed is True
    assert result.reason_codes
    assert any(
        "fallback" in c.lower() or "eth" in c.lower() for c in result.reason_codes
    )


def test_honest_ib_run_gate_one() -> None:
    result = evaluate_fabric_gate(
        fabric_mode="ib",
        required_transport="ib",
        actual_transport="ib",
        reports=[_ib_report()],
        eth_fallback_injected=False,
        correctness_present=True,
    )
    assert result.fabric_gate == 1.0
    assert result.composite_zeroed is False


def test_ib_required_with_zero_device_reports_zeroes_gate() -> None:
    result = evaluate_fabric_gate(
        fabric_mode="ib",
        required_transport="ib",
        actual_transport="ib",
        reports=[_eth_report()],
        eth_fallback_injected=False,
        correctness_present=True,
    )
    assert result.fabric_gate == 0.0


# ----- VAL-FAB-010 ----------------------------------------------------------


def test_require_ib_accepts_all_ib_nodes() -> None:
    """VAL-FAB-010: require_ib ok when all nodes report compatible IB."""

    check = evaluate_require_ib_nodes(
        require_ib=True,
        reports=[_ib_report("a"), _ib_report("b")],
        node_ids=["a", "b"],
    )
    assert isinstance(check, RequireIbCheck)
    assert check.ok is True
    assert check.may_rent is True


def test_require_ib_rejects_eth_only_inventory() -> None:
    """VAL-FAB-010 FAIL: require_ib must not rent eth-only inventory."""

    check = evaluate_require_ib_nodes(
        require_ib=True,
        reports=[_eth_report("a"), _ib_report("b")],
        node_ids=["a", "b"],
    )
    assert check.ok is False
    assert check.may_rent is False
    assert "a" in check.missing_ib_node_ids or any(
        "a" in x for x in check.missing_ib_node_ids
    )


def test_require_ib_false_allows_eth() -> None:
    check = evaluate_require_ib_nodes(
        require_ib=False,
        reports=[_eth_report("a")],
        node_ids=["a"],
    )
    assert check.ok is True
    assert check.may_rent is True


def test_require_ib_rejects_when_reports_stripped() -> None:
    """Updated zero-IB re-report for previously IB node blocks new rents."""

    check = evaluate_require_ib_nodes(
        require_ib=True,
        reports=[_eth_report("was-ib")],
        node_ids=["was-ib"],
    )
    assert check.ok is False
    assert check.may_rent is False


# ----- VAL-FAB-011 ----------------------------------------------------------


def test_cluster_requires_all_member_fabric_reports() -> None:
    """VAL-FAB-011: mode=cluster needs reports for all leased nodes."""

    reports = [_ib_report("n0"), _ib_report("n1")]
    ok_eval = evaluate_cluster_member_reports(
        mode="cluster",
        member_node_ids=["n0", "n1"],
        reports=reports,
    )
    assert isinstance(ok_eval, ClusterFabricEvaluation)
    assert ok_eval.ok is True
    assert ok_eval.may_launch is True

    partial = evaluate_cluster_member_reports(
        mode="cluster",
        member_node_ids=["n0", "n1", "n2"],
        reports=reports,
    )
    assert partial.ok is False
    assert partial.may_launch is False
    assert "n2" in partial.missing_node_ids


def test_cluster_partial_reports_block_success_path() -> None:
    inv = seed_sim_inventory(seed=1, node_count=3, gpus_per_node=2)
    all_ids = [n.node_id for n in inv.nodes]
    # Drop last report
    partial_reports = [n.fabric_report for n in inv.nodes[:-1]]
    result = evaluate_cluster_member_reports(
        mode="cluster",
        member_node_ids=all_ids,
        reports=partial_reports,
    )
    assert result.may_launch is False
    assert result.missing_node_ids


def test_single_mode_one_report_sufficient() -> None:
    result = evaluate_cluster_member_reports(
        mode="single",
        member_node_ids=["only"],
        reports=[_eth_report("only")],
    )
    assert result.ok is True


# ----- VAL-FAB-023 ----------------------------------------------------------


def test_ib_rate_mismatch_policy_is_documented() -> None:
    """VAL-FAB-023: policy constant is documented (strict or soft)."""

    assert IB_RATE_MISMATCH_POLICY in {"strict", "soft", "flag"}
    assert isinstance(IB_RATE_MISMATCH_POLICY, str)


def test_mixed_zero_and_nonzero_ib_fails_ib_fabric() -> None:
    """VAL-FAB-023: zero-IB mixed with IB-domain fails under ib fabric."""

    result = evaluate_fabric_mode(
        fabric_mode="ib",
        reports=[_ib_report("ib1"), _eth_report("eth1")],
    )
    assert result.ok is False
    assert result.fabric_gate == 0.0


def test_heterogeneous_ib_rates_all_ib_still_place_with_digest() -> None:
    """All-IB heterogeneous rates still place with explicit graph digest."""

    reports = [
        _ib_report("fast", rate_gbps=400.0),
        _ib_report("slow", rate_gbps=100.0),
    ]
    consistency = evaluate_ib_rate_consistency(reports)
    assert consistency.all_have_ib is True
    assert consistency.rates_uniform is False
    # Policy either flags or accepts — never pretends eth node is IB peer.
    assert consistency.may_place_ib is True
    assert consistency.graph_digest.startswith("sha256:")
    if IB_RATE_MISMATCH_POLICY == "strict":
        # Strict could still allow place with flag; max/min spread policy documented.
        assert consistency.warning or consistency.ok
    else:
        assert consistency.ok is True or consistency.warning


def test_zero_ib_with_nonzero_peer_not_uniform_fabric() -> None:
    consistency = evaluate_ib_rate_consistency([_ib_report("a"), _eth_report("b")])
    assert consistency.all_have_ib is False
    assert consistency.may_place_ib is False
    assert consistency.ok is False


def test_reports_by_node_id_helper() -> None:
    r1 = _ib_report("x")
    r2 = _eth_report("y")
    mapping = reports_by_node_id([r1, r2])
    assert mapping["x"].node_id == "x"
    assert mapping["y"].node_id == "y"
