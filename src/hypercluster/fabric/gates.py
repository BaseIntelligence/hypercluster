"""Fabric mode gates, require_ib consistency, eth-fallback scoring inputs.

Architecture §8.1 / §10 fabric_gate rules.

Fulfills:
  VAL-FAB-002  fabric=ib zero IB devices fail-closed
  VAL-FAB-003  fabric=auto eth/sim allowed
  VAL-FAB-010  require_ib authenticates node fabric consistency
  VAL-FAB-011  cluster mode demands all member FabricReports
  VAL-FAB-012  forbidden eth fallback under IB zeroes fabric_gate
  VAL-FAB-021  eth mode does not set NCCL_NET=IB
  VAL-FAB-023  mismatched IB rates policy documented and enforced
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from hypercluster.fabric.discovery import DIGEST_PREFIX, FabricReport, canonical_json

FabricMode = Literal["auto", "ib", "eth", "nvlink_only"]
Transport = Literal["ib", "eth", "nvlink", "socket", "auto-eth", "auto-ib"]

# Policy for heterogeneous IB rates when ALL nodes still have IB:
#   soft  — place allowed, warning + graph digest (default)
#   flag  — place allowed, fabric_gate stays 1 but warning codes emitted
#   strict — reject place when max/min rate ratio exceeds STRICT_RATE_RATIO
#
# Mixed zero-IB + IB is ALWAYS fail-closed for fabric=ib / require_ib (never silent).
IB_RATE_MISMATCH_POLICY: Literal["strict", "soft", "flag"] = "soft"
STRICT_RATE_RATIO = 4.0  # e.g. 400 vs 100 still under soft; 800 vs 100 strict if enabled

PLANNER_NCCL_ENV_VERSION = "nccl_env.v1"


@dataclass(slots=True, frozen=True)
class FabricModeEvaluation:
    """Outcome of fabric mode admission against node FabricReports."""

    ok: bool
    may_succeed: bool
    fabric_gate: float
    resolved_transport: str
    reason: str
    failure_code: str | None = None
    missing_ib_node_ids: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class FabricGateResult:
    """Scoring fabric_gate for an attempt (architecture §10)."""

    fabric_gate: float
    composite_zeroed: bool
    reason_codes: list[str] = field(default_factory=list)
    required_transport: str = ""
    actual_transport: str = ""


@dataclass(slots=True, frozen=True)
class RequireIbCheck:
    """require_ib offer / rent fabric consistency (VAL-FAB-010)."""

    ok: bool
    may_rent: bool
    missing_ib_node_ids: list[str] = field(default_factory=list)
    reason: str = ""
    failure_code: str | None = None


@dataclass(slots=True, frozen=True)
class ClusterFabricEvaluation:
    """Cluster multi-node fabric domain readiness (VAL-FAB-011)."""

    ok: bool
    may_launch: bool
    missing_node_ids: list[str] = field(default_factory=list)
    reason: str = ""
    failure_code: str | None = None


@dataclass(slots=True, frozen=True)
class IbRateConsistency:
    """IB rate policy across a cluster (VAL-FAB-023)."""

    ok: bool
    may_place_ib: bool
    all_have_ib: bool
    rates_uniform: bool
    graph_digest: str
    min_rate: float | None = None
    max_rate: float | None = None
    warning: str | None = None
    reason: str = ""
    policy: str = IB_RATE_MISMATCH_POLICY


def reports_by_node_id(reports: list[FabricReport]) -> dict[str, FabricReport]:
    """Map node_id → latest-looking report (last wins if duplicate)."""

    out: dict[str, FabricReport] = {}
    for r in reports:
        out[r.node_id] = r
    return out


def has_active_ib_devices(report: FabricReport | None) -> bool:
    """True when report has at least one IB device that is not Down/disabled."""

    if report is None:
        return False
    for dev in report.ib_devices:
        if not dev.name:
            continue
        state = (dev.state or "Active").strip().lower()
        if state in {"down", "disabled", "inactive", "error"}:
            continue
        if dev.rate_gbps is not None and float(dev.rate_gbps) <= 0:
            continue
        return True
    # Fallback: aggregated rate alone is not enough without devices for zero-device rule.
    return False


def _node_rate(report: FabricReport) -> float | None:
    if report.ib_rate_gbps is not None and report.ib_rate_gbps > 0:
        return float(report.ib_rate_gbps)
    rates = [float(d.rate_gbps) for d in report.ib_devices if d.rate_gbps and d.rate_gbps > 0]
    return max(rates) if rates else None


def list_ib_hca_names(reports: list[FabricReport]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for report in reports:
        for dev in report.ib_devices:
            if dev.name and dev.name not in seen:
                seen.add(dev.name)
                names.append(dev.name)
    return names


def evaluate_ib_rate_consistency(reports: list[FabricReport]) -> IbRateConsistency:
    """Evaluate IB rate agreement across nodes (VAL-FAB-023).

    Never pretends a zero-IB node is a peer of an IB node for ``fabric=ib``.
    All-IB heterogeneous rates: place allowed under soft/flag policy with graph digest.
    """

    if not reports:
        digest = DIGEST_PREFIX + hashlib.sha256(b"empty-ib-rate").hexdigest()
        return IbRateConsistency(
            ok=False,
            may_place_ib=False,
            all_have_ib=False,
            rates_uniform=True,
            graph_digest=digest,
            reason="no fabric reports",
        )

    have_ib = [has_active_ib_devices(r) for r in reports]
    all_ib = all(have_ib)
    any_ib = any(have_ib)
    rates = [_node_rate(r) for r in reports if has_active_ib_devices(r)]
    rates_clean = [r for r in rates if r is not None]

    graph_body = {
        "node_ids": [r.node_id for r in reports],
        "rates": {
            r.node_id: _node_rate(r) if has_active_ib_devices(r) else 0.0 for r in reports
        },
        "has_ib": {r.node_id: has_active_ib_devices(r) for r in reports},
        "policy": IB_RATE_MISMATCH_POLICY,
    }
    graph_digest = DIGEST_PREFIX + hashlib.sha256(canonical_json(graph_body).encode()).hexdigest()

    if not all_ib:
        # Mixed zero / non-zero IB — never uniform, never place as IB domain.
        return IbRateConsistency(
            ok=False,
            may_place_ib=False,
            all_have_ib=False,
            rates_uniform=False,
            graph_digest=graph_digest,
            min_rate=min(rates_clean) if rates_clean else None,
            max_rate=max(rates_clean) if rates_clean else None,
            reason=(
                "mixed zero-IB and IB nodes; fabric does not pretend eth node "
                "as IB peer"
            ),
            warning="zero_vs_nonzero_ib",
        )

    if not rates_clean:
        return IbRateConsistency(
            ok=False,
            may_place_ib=False,
            all_have_ib=any_ib,
            rates_uniform=True,
            graph_digest=graph_digest,
            reason="all nodes claim IB without positive rates",
        )

    min_r = min(rates_clean)
    max_r = max(rates_clean)
    uniform = max_r == min_r
    warning: str | None = None
    may_place = True
    ok = True

    if not uniform:
        ratio = max_r / min_r if min_r > 0 else float("inf")
        warning = f"heterogeneous_ib_rates min={min_r} max={max_r} ratio={ratio:.3g}"
        if IB_RATE_MISMATCH_POLICY == "strict" and ratio > STRICT_RATE_RATIO:
            ok = False
            may_place = False
            warning = f"strict_rate_mismatch {warning}"
        else:
            # soft / flag: place allowed with digest + warning
            ok = True
            may_place = True

    return IbRateConsistency(
        ok=ok,
        may_place_ib=may_place,
        all_have_ib=True,
        rates_uniform=uniform,
        graph_digest=graph_digest,
        min_rate=min_r,
        max_rate=max_r,
        warning=warning,
        reason="ok" if ok else "strict rate policy reject",
        policy=IB_RATE_MISMATCH_POLICY,
    )


def evaluate_fabric_mode(
    *,
    fabric_mode: str,
    reports: list[FabricReport],
) -> FabricModeEvaluation:
    """Admit fabric mode against reports (VAL-FAB-002/003/023).

    - ``ib``: every bound node must have active IB devices; mixed eth fails closed.
    - ``auto``: may run eth-only (sockets); upgrade to IB when all nodes have IB.
    - ``eth``: always ok without IB; never requires devices.
    - ``nvlink_only``: ok if reports exist (GPU topology used by planner later).
    """

    mode = (fabric_mode or "auto").strip().lower()
    if mode not in {"auto", "ib", "eth", "nvlink_only"}:
        return FabricModeEvaluation(
            ok=False,
            may_succeed=False,
            fabric_gate=0.0,
            resolved_transport="unknown",
            reason=f"unsupported fabric mode {fabric_mode!r}",
            failure_code="invalid_fabric_mode",
        )

    if not reports:
        if mode in {"eth", "auto", "nvlink_only"}:
            # Empty inventory is a placement issue later; mode itself is not require_ib.
            return FabricModeEvaluation(
                ok=True,
                may_succeed=True,
                fabric_gate=1.0,
                resolved_transport="eth" if mode == "eth" else "auto-eth",
                reason="no reports yet; mode does not mandate IB",
            )
        return FabricModeEvaluation(
            ok=False,
            may_succeed=False,
            fabric_gate=0.0,
            resolved_transport="ib",
            reason="missing IB: fabric=ib requires fabric reports with devices",
            failure_code="missing_ib",
        )

    missing = [r.node_id for r in reports if not has_active_ib_devices(r)]
    all_ib = len(missing) == 0
    consistency = evaluate_ib_rate_consistency(reports)

    if mode == "ib":
        if not all_ib:
            return FabricModeEvaluation(
                ok=False,
                may_succeed=False,
                fabric_gate=0.0,
                resolved_transport="ib",
                reason=(
                    "missing IB devices on node(s): "
                    + (", ".join(missing) if missing else "unknown")
                ),
                failure_code="missing_ib",
                missing_ib_node_ids=list(missing),
            )
        if not consistency.may_place_ib:
            return FabricModeEvaluation(
                ok=False,
                may_succeed=False,
                fabric_gate=0.0,
                resolved_transport="ib",
                reason=consistency.reason or "IB rate policy reject",
                failure_code="ib_rate_mismatch",
            )
        return FabricModeEvaluation(
            ok=True,
            may_succeed=True,
            fabric_gate=1.0,
            resolved_transport="ib",
            reason=consistency.warning or "IB fabric ready",
        )

    if mode == "eth":
        return FabricModeEvaluation(
            ok=True,
            may_succeed=True,
            fabric_gate=1.0,
            resolved_transport="eth",
            reason="eth fabric does not require IB",
        )

    if mode == "nvlink_only":
        return FabricModeEvaluation(
            ok=True,
            may_succeed=True,
            fabric_gate=1.0,
            resolved_transport="nvlink",
            reason="nvlink_only fabric mode",
        )

    # auto
    if all_ib and consistency.may_place_ib:
        return FabricModeEvaluation(
            ok=True,
            may_succeed=True,
            fabric_gate=1.0,
            resolved_transport="ib",
            reason="auto selected IB (all members have active devices)",
        )
    return FabricModeEvaluation(
        ok=True,
        may_succeed=True,
        fabric_gate=1.0,
        resolved_transport="auto-eth",
        reason="auto falling back to eth/sim sockets without requiring IB",
    )


def evaluate_fabric_gate(
    *,
    fabric_mode: str,
    required_transport: str | None = None,
    actual_transport: str | None = None,
    reports: list[FabricReport] | None = None,
    eth_fallback_injected: bool = False,
    correctness_present: bool = True,
) -> FabricGateResult:
    """Compute fabric_gate ∈ {0, 1} for scoring (VAL-FAB-002/012).

    Correctness alone does not restore fabric_gate when the fabric was lied about.
    """

    _ = correctness_present  # may be non-zero correctness; fabric gate is independent
    mode = (fabric_mode or "auto").strip().lower()
    reports = reports or []
    required = (required_transport or ("ib" if mode == "ib" else "")).strip().lower()
    actual = (actual_transport or "").strip().lower()
    codes: list[str] = []

    # Forbidden eth fallback under IB demand.
    if mode == "ib" or required in {"ib", "infiniband", "rdma"}:
        if eth_fallback_injected:
            codes.append("forbidden_eth_fallback")
        if actual in {"eth", "socket", "auto-eth", "tcp"}:
            codes.append("actual_transport_eth_under_ib")
        mode_eval = evaluate_fabric_mode(fabric_mode="ib", reports=reports)
        if not mode_eval.ok:
            codes.append(mode_eval.failure_code or "missing_ib")
            if mode_eval.missing_ib_node_ids:
                codes.append("missing_ib_devices")

    if codes:
        return FabricGateResult(
            fabric_gate=0.0,
            composite_zeroed=True,
            reason_codes=codes,
            required_transport=required or mode,
            actual_transport=actual or "unknown",
        )

    # Non-IB modes: gate 1 when mode evaluation says ok.
    mode_eval = evaluate_fabric_mode(fabric_mode=mode, reports=reports)
    if not mode_eval.ok:
        return FabricGateResult(
            fabric_gate=0.0,
            composite_zeroed=True,
            reason_codes=[mode_eval.failure_code or "fabric_mode_fail"],
            required_transport=required or mode,
            actual_transport=actual or mode_eval.resolved_transport,
        )

    return FabricGateResult(
        fabric_gate=1.0,
        composite_zeroed=False,
        reason_codes=[],
        required_transport=required or mode,
        actual_transport=actual or mode_eval.resolved_transport,
    )


def evaluate_require_ib_nodes(
    *,
    require_ib: bool,
    reports: list[FabricReport],
    node_ids: list[str],
) -> RequireIbCheck:
    """Authenticate node fabric consistency for require_ib offers (VAL-FAB-010).

    Rent/list path should re-check latest reports so stripped IB re-report
    prevents new rents even if the offer was created earlier.
    """

    if not require_ib:
        return RequireIbCheck(ok=True, may_rent=True, reason="require_ib not set")

    by_id = reports_by_node_id(reports)
    missing: list[str] = []
    for nid in node_ids:
        report = by_id.get(nid)
        if report is None or not has_active_ib_devices(report):
            missing.append(nid)

    if missing:
        return RequireIbCheck(
            ok=False,
            may_rent=False,
            missing_ib_node_ids=missing,
            reason=(
                "require_ib offer blocked: node(s) lack active IB fabric: "
                + ", ".join(missing)
            ),
            failure_code="require_ib_not_satisfied",
        )

    # Also reject if rates claim zero-vs-nonzero mixed (should not happen if has_ib).
    consistency = evaluate_ib_rate_consistency(
        [by_id[n] for n in node_ids if n in by_id]
    )
    if not consistency.may_place_ib:
        return RequireIbCheck(
            ok=False,
            may_rent=False,
            missing_ib_node_ids=list(missing),
            reason=consistency.reason,
            failure_code="require_ib_fabric_inconsistent",
        )

    return RequireIbCheck(
        ok=True,
        may_rent=True,
        reason="all nodes have compatible IB reports",
    )


def evaluate_cluster_member_reports(
    *,
    mode: str,
    member_node_ids: list[str],
    reports: list[FabricReport],
) -> ClusterFabricEvaluation:
    """Ensure cluster fabric domain has reports for all leased nodes (VAL-FAB-011)."""

    mode_norm = (mode or "single").strip().lower()
    by_id = reports_by_node_id(reports)
    missing = [nid for nid in member_node_ids if nid not in by_id]

    if mode_norm == "cluster":
        if missing:
            return ClusterFabricEvaluation(
                ok=False,
                may_launch=False,
                missing_node_ids=missing,
                reason=(
                    "cluster mode requires FabricReports for all member nodes; "
                    f"missing: {', '.join(missing)}"
                ),
                failure_code="cluster_fabric_reports_incomplete",
            )
        if not member_node_ids:
            return ClusterFabricEvaluation(
                ok=False,
                may_launch=False,
                missing_node_ids=[],
                reason="cluster mode with empty member set",
                failure_code="cluster_empty",
            )
        return ClusterFabricEvaluation(
            ok=True,
            may_launch=True,
            reason="all cluster member FabricReports present",
        )

    # single: at least the one member if declared
    if member_node_ids and missing:
        return ClusterFabricEvaluation(
            ok=False,
            may_launch=False,
            missing_node_ids=missing,
            reason="missing fabric report for single-node member",
            failure_code="fabric_report_missing",
        )
    return ClusterFabricEvaluation(
        ok=True,
        may_launch=True,
        reason="single-mode fabric report set sufficient",
    )


def build_nccl_env_for_mode(
    *,
    fabric_mode: str,
    reports: list[FabricReport] | None = None,
    backend: str = "nccl",
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
    socket_ifname: str = "lo",
) -> dict[str, str]:
    """Build NCCL env matrix for a fabric mode (VAL-FAB-021 + stubs for later).

    eth / auto-on-eth never forces ``NCCL_NET=IB`` or IB_HCA.
    ib (when devices present) sets IB transport keys.
    """

    reports = reports or []
    mode_eval = evaluate_fabric_mode(fabric_mode=fabric_mode, reports=reports)
    mode = (fabric_mode or "auto").strip().lower()
    transport = mode_eval.resolved_transport

    env: dict[str, str] = {
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": master_port,
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_SOCKET_IFNAME": socket_ifname,
        "HYPER_BACKEND": backend,
        "HYPER_FABRIC_MODE": mode,
        "HYPER_NCCL_ENV_VERSION": PLANNER_NCCL_ENV_VERSION,
        "HYPER_RESOLVED_TRANSPORT": transport,
    }

    use_ib = False
    if mode == "ib" and mode_eval.ok and transport == "ib":
        use_ib = True
    elif mode == "auto" and transport == "ib":
        use_ib = True

    if use_ib:
        hcas = list_ib_hca_names(reports)
        env["NCCL_NET"] = "IB"
        env["NCCL_IB_HCA"] = ",".join(hcas) if hcas else "mlx5_0"
        env["NCCL_IB_GID_INDEX"] = "3"
    elif mode == "eth" or transport in {"eth", "auto-eth", "socket"}:
        # Explicitly Socket — never IB force (VAL-FAB-021).
        env["NCCL_NET"] = "Socket"
        # Do not set NCCL_IB_HCA
    elif mode == "nvlink_only":
        # Prefer sockets + leave IB unset so nccl stays intra-node.
        env["NCCL_NET"] = "Socket"
        env["NCCL_P2P_LEVEL"] = "NVL"
    else:
        env["NCCL_NET"] = "Socket"

    return env


def fabric_mode_blocks_success(mode_eval: FabricModeEvaluation) -> bool:
    return not mode_eval.may_succeed


def summarize_gate_for_score(
    gate: FabricGateResult,
    *,
    correctness: float = 1.0,
    efficiency: float = 1.0,
    tee_bonus: float = 1.0,
) -> dict[str, Any]:
    """Four-factor product preview using fabric_gate (library/scoring.md)."""

    composite = float(correctness) * float(efficiency) * float(gate.fabric_gate) * float(
        tee_bonus
    )
    if gate.composite_zeroed:
        composite = 0.0
    return {
        "correctness": float(correctness),
        "efficiency": float(efficiency),
        "fabric_gate": float(gate.fabric_gate),
        "tee_bonus": float(tee_bonus),
        "composite": composite,
        "reason_codes": list(gate.reason_codes),
    }


# Re-export for IbDevice type checks by callers that only import gates.
__all__ = [
    "IB_RATE_MISMATCH_POLICY",
    "PLANNER_NCCL_ENV_VERSION",
    "STRICT_RATE_RATIO",
    "ClusterFabricEvaluation",
    "FabricGateResult",
    "FabricModeEvaluation",
    "IbRateConsistency",
    "RequireIbCheck",
    "build_nccl_env_for_mode",
    "evaluate_cluster_member_reports",
    "evaluate_fabric_gate",
    "evaluate_fabric_mode",
    "evaluate_ib_rate_consistency",
    "evaluate_require_ib_nodes",
    "fabric_mode_blocks_success",
    "has_active_ib_devices",
    "list_ib_hca_names",
    "reports_by_node_id",
    "summarize_gate_for_score",
]
