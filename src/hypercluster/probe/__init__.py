"""M9 non-TEE GPU host probe (ordered checks + evidence model).

Public surface for the domain pipeline. Real SSH executor and API routes
land in later M9 features; this package stays transport-protocol pure so
CI can drive every gate with FakeSsh (no real network).
"""

from __future__ import annotations

from hypercluster.probe.model_table import (
    GpuFamilySpec,
    VramWindow,
    family_for_name,
    lookup_vram_window,
    models_match,
    normalize_gpu_model,
)
from hypercluster.probe.pipeline import (
    ADVISORY_CHECK_IDS,
    CHECK_ORDER,
    FATAL_CHECK_IDS,
    GpuProbeConfig,
    GpuProbeContext,
    GpuProbeService,
    run_gpu_probe,
)
from hypercluster.probe.transport import (
    COMMAND_ALLOWLIST,
    FakeSshTransport,
    SshCommandResult,
    SshTransport,
    TransportError,
)
from hypercluster.probe.types import (
    CheckResult,
    ClaimedInventory,
    GpuHostEvidence,
    MeasuredGpu,
    MeasuredInventory,
    ProbeDigests,
    ProbeStatus,
    new_evidence_id,
)

__all__ = [
    "ADVISORY_CHECK_IDS",
    "CHECK_ORDER",
    "COMMAND_ALLOWLIST",
    "FATAL_CHECK_IDS",
    "CheckResult",
    "ClaimedInventory",
    "FakeSshTransport",
    "GpuFamilySpec",
    "GpuHostEvidence",
    "GpuProbeConfig",
    "GpuProbeContext",
    "GpuProbeService",
    "MeasuredGpu",
    "MeasuredInventory",
    "ProbeDigests",
    "ProbeStatus",
    "SshCommandResult",
    "SshTransport",
    "TransportError",
    "VramWindow",
    "family_for_name",
    "lookup_vram_window",
    "models_match",
    "new_evidence_id",
    "normalize_gpu_model",
    "run_gpu_probe",
]
