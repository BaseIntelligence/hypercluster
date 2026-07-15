"""dstack TEE offline verify path (architecture §9).

Offline fixtures are CI-mandatory; live path is optional later.
"""

from __future__ import annotations

from hypercluster.attest.gpu_evidence import (
    GPU_EVIDENCE_SCHEME,
    GpuEvidence,
    GpuEvidenceSchemaError,
    mock_gpu_evidence,
    parse_gpu_evidence,
    validate_gpu_evidence,
)
from hypercluster.attest.models import TeeVerifyRequest, TeeVerifyResult
from hypercluster.attest.offline_fixtures import (
    OFFLINE_FIXTURE_KIND,
    OfflineQuoteEnvelope,
    load_quote_fixture,
    make_offline_envelope,
    package_quote_b64,
    unpack_quote_b64,
)
from hypercluster.attest.policy import (
    DEFAULT_COMPOSE_HASH_GOLDEN,
    TeeVerifyPolicy,
    default_policy_from_settings,
)
from hypercluster.attest.report_data import (
    REPORT_DATA_SIZE,
    ParsedReportData,
    ReportDataLayoutError,
    build_job_digest,
    build_report_data,
    parse_report_data,
    report_data_hex,
)
from hypercluster.attest.verify import verify_offline_fixture_file, verify_tee

__all__ = [
    "DEFAULT_COMPOSE_HASH_GOLDEN",
    "GPU_EVIDENCE_SCHEME",
    "OFFLINE_FIXTURE_KIND",
    "REPORT_DATA_SIZE",
    "GpuEvidence",
    "GpuEvidenceSchemaError",
    "OfflineQuoteEnvelope",
    "ParsedReportData",
    "ReportDataLayoutError",
    "TeeVerifyPolicy",
    "TeeVerifyRequest",
    "TeeVerifyResult",
    "build_job_digest",
    "build_report_data",
    "default_policy_from_settings",
    "load_quote_fixture",
    "make_offline_envelope",
    "mock_gpu_evidence",
    "package_quote_b64",
    "parse_gpu_evidence",
    "parse_report_data",
    "report_data_hex",
    "unpack_quote_b64",
    "validate_gpu_evidence",
    "verify_offline_fixture_file",
    "verify_tee",
]
