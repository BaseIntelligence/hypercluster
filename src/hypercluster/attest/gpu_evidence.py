"""Mock GPU NRAS evidence schema for tdx+gpu_cc (VAL-TEE-012).

Live NVIDIA Remote Attestation Service integration is out of scope; CI uses a
challenge-owned mock envelope so tests can assert nonce-echo bind without
network or silicon.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

GPU_EVIDENCE_SCHEME = "nras_mock_v1"


class GpuEvidenceSchemaError(ValueError):
    """Raised when mock GPU evidence fails schema / nonce bind."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class GpuEvidence(BaseModel):
    """Mock NRAS GPU evidence (challenge-owned offline schema)."""

    model_config = ConfigDict(extra="forbid")

    scheme: Literal["nras_mock_v1"] = "nras_mock_v1"
    nonce: str = Field(..., min_length=1)
    # Must equal ``nonce`` when supplied (nonce echo bind).
    nonce_echo: str | None = None
    architecture: str | None = None
    measurement: str | None = None
    attester: str | None = "sim-nras"
    raw: dict[str, Any] | None = None

    @field_validator("nonce")
    @classmethod
    def _nonce_nonempty(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("nonce is required")
        return v


def parse_gpu_evidence(raw: dict[str, Any] | None) -> GpuEvidence:
    """Parse GPU evidence dict; raise GpuEvidenceSchemaError on bad shape."""

    if raw is None:
        raise GpuEvidenceSchemaError("gpu_evidence_missing", "gpu_evidence is required")
    if not isinstance(raw, dict):
        raise GpuEvidenceSchemaError("gpu_evidence_invalid", "gpu_evidence must be an object")
    try:
        return GpuEvidence.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 — surface as schema fail
        raise GpuEvidenceSchemaError(
            "gpu_evidence_schema",
            f"invalid gpu evidence schema: {exc}",
        ) from exc


def validate_gpu_evidence(
    raw: dict[str, Any] | None,
    *,
    expected_nonce: str | None = None,
    require: bool = True,
) -> tuple[bool, list[str], GpuEvidence | None]:
    """Validate mock NRAS evidence.

    Returns ``(ok, reason_codes, evidence_or_none)``.

    * Good nonce echo with matching expected_nonce → ok.
    * Mismatched nonce / nonce_echo → fail with ``gpu_evidence_nonce_mismatch``.
    * Missing when ``require`` → ``gpu_evidence_missing``.
    """

    reasons: list[str] = []
    if raw is None:
        if require:
            reasons.append("gpu_evidence_missing")
            return False, reasons, None
        return True, reasons, None

    try:
        evidence = parse_gpu_evidence(raw)
    except GpuEvidenceSchemaError as exc:
        reasons.append(exc.code)
        reasons.append("gpu_evidence_invalid")
        return False, reasons, None

    echo = evidence.nonce_echo if evidence.nonce_echo is not None else evidence.nonce
    if echo != evidence.nonce:
        reasons.append("gpu_evidence_nonce_mismatch")
        reasons.append("gpu_nonce_echo_fail")
        return False, reasons, evidence

    if expected_nonce is not None and evidence.nonce != expected_nonce:
        reasons.append("gpu_evidence_nonce_mismatch")
        reasons.append("gpu_nonce_bind_fail")
        return False, reasons, evidence

    return True, reasons, evidence


def mock_gpu_evidence(
    *,
    nonce: str,
    nonce_echo: str | None = None,
    architecture: str = "hopper",
    measurement: str = "sha256:gpu-mock-meas-0001",
) -> dict[str, Any]:
    """Build a green-path mock NRAS blob for fixtures / tests."""

    body = GpuEvidence(
        scheme="nras_mock_v1",
        nonce=nonce,
        nonce_echo=nonce_echo if nonce_echo is not None else nonce,
        architecture=architecture,
        measurement=measurement,
        attester="sim-nras",
    )
    return body.model_dump(mode="json")


__all__ = [
    "GPU_EVIDENCE_SCHEME",
    "GpuEvidence",
    "GpuEvidenceSchemaError",
    "mock_gpu_evidence",
    "parse_gpu_evidence",
    "validate_gpu_evidence",
]
