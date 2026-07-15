"""Fabric / InfiniBand / NCCL modules — owned largely by fabric-worker milestones."""

from __future__ import annotations

from hypercluster.fabric.discovery import (
    DIGEST_PREFIX,
    FabricReport,
    IbDevice,
    build_fabric_report,
    canonical_report_payload,
    compute_report_digest,
    synthetic_ib_devices,
    synthetic_nvlink_topo_matrix,
    validate_accepted_report,
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
    evaluate_require_ib_nodes,
    has_active_ib_devices,
)

__all__ = [
    "DIGEST_PREFIX",
    "IB_RATE_MISMATCH_POLICY",
    "ClusterFabricEvaluation",
    "FabricGateResult",
    "FabricModeEvaluation",
    "FabricReport",
    "IbDevice",
    "RequireIbCheck",
    "build_fabric_report",
    "build_nccl_env_for_mode",
    "canonical_report_payload",
    "compute_report_digest",
    "evaluate_cluster_member_reports",
    "evaluate_fabric_gate",
    "evaluate_fabric_mode",
    "evaluate_require_ib_nodes",
    "has_active_ib_devices",
    "synthetic_ib_devices",
    "synthetic_nvlink_topo_matrix",
    "validate_accepted_report",
]
