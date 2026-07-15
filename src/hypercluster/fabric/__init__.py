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

__all__ = [
    "DIGEST_PREFIX",
    "FabricReport",
    "IbDevice",
    "build_fabric_report",
    "canonical_report_payload",
    "compute_report_digest",
    "synthetic_ib_devices",
    "synthetic_nvlink_topo_matrix",
    "validate_accepted_report",
]
