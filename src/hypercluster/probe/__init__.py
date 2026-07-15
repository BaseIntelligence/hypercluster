"""M9 non-TEE GPU host probe (ordered checks + evidence model).

Public surface for the domain pipeline. Real SSH executor and API routes
land in later M9 features; this package stays transport-protocol pure so
CI can drive every gate with FakeSsh (no real network).
"""

from __future__ import annotations

from hypercluster.probe.allowlist import (
    COMMAND_REGISTRY,
    CommandSpec,
    argv_for_command,
    command_timeout_s,
    is_allowlisted,
)
from hypercluster.probe.fixtures import (
    KNOWN_FIXTURE_NAMES,
    FakeSshFixture,
    get_fixture,
    list_fixtures,
    load_fixture_json,
    load_named_fixture,
)
from hypercluster.probe.inventory_merge import (
    apply_probe_to_node_fields,
    merge_probe_into_inventory,
)
from hypercluster.probe.keys import (
    KeyMaterialError,
    KeyRef,
    compute_key_fingerprint,
    public_key_meta_for_evidence,
    reject_body_private_key_fields,
    resolve_key_ref,
)
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
from hypercluster.probe.redact import (
    contains_private_key_material,
    redact_mapping,
    redact_secrets,
    redact_text,
    truncate_output,
)
from hypercluster.probe.resolve import (
    FAKE_SSH_NOT_ALLOWED,
    SSH_TRANSPORT_UNAVAILABLE,
    TransportConfigError,
    resolve_ssh_transport,
)
from hypercluster.probe.ssh_exec import (
    DEFAULT_NODE_PROBE_LOCK,
    NodeProbeLock,
    RealSshExecutor,
    RealSshTarget,
    build_real_ssh_transport,
    parse_ssh_endpoint,
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
    "COMMAND_REGISTRY",
    "DEFAULT_NODE_PROBE_LOCK",
    "FAKE_SSH_NOT_ALLOWED",
    "FATAL_CHECK_IDS",
    "KNOWN_FIXTURE_NAMES",
    "CheckResult",
    "ClaimedInventory",
    "CommandSpec",
    "FakeSshFixture",
    "FakeSshTransport",
    "GpuFamilySpec",
    "GpuHostEvidence",
    "GpuProbeConfig",
    "GpuProbeContext",
    "GpuProbeService",
    "KeyMaterialError",
    "KeyRef",
    "MeasuredGpu",
    "MeasuredInventory",
    "NodeProbeLock",
    "ProbeDigests",
    "ProbeStatus",
    "RealSshExecutor",
    "RealSshTarget",
    "SSH_TRANSPORT_UNAVAILABLE",
    "SshCommandResult",
    "SshTransport",
    "TransportConfigError",
    "TransportError",
    "VramWindow",
    "apply_probe_to_node_fields",
    "argv_for_command",
    "build_real_ssh_transport",
    "command_timeout_s",
    "compute_key_fingerprint",
    "contains_private_key_material",
    "family_for_name",
    "get_fixture",
    "is_allowlisted",
    "list_fixtures",
    "load_fixture_json",
    "load_named_fixture",
    "lookup_vram_window",
    "merge_probe_into_inventory",
    "models_match",
    "new_evidence_id",
    "normalize_gpu_model",
    "parse_ssh_endpoint",
    "public_key_meta_for_evidence",
    "redact_mapping",
    "redact_secrets",
    "redact_text",
    "reject_body_private_key_fields",
    "resolve_key_ref",
    "resolve_ssh_transport",
    "run_gpu_probe",
    "truncate_output",
]
