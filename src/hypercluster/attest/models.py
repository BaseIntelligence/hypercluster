"""TEE verify request / result models (architecture §9.1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TeeVerifyRequest(BaseModel):
    """Challenge-owned offline/live/sim verify input."""

    model_config = ConfigDict(extra="forbid")

    quote_b64: str = Field(..., min_length=1)
    event_log: str | None = None
    vm_config: dict[str, Any] | None = None
    report_data_expected: bytes
    gpu_evidence: dict[str, Any] | None = None
    mode: Literal["offline_fixture", "live", "sim"] = "offline_fixture"


class TeeVerifyResult(BaseModel):
    """Verification verdict (is_valid + machine-readable reason_codes)."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    quote_verified: bool
    event_log_verified: bool = False
    os_image_hash_verified: bool = False
    tcb_status: str = "unknown"
    advisory_ids: list[str] = Field(default_factory=list)
    compose_hash: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    verify_mode: Literal["offline_fixture", "live", "sim"] | None = None

    def to_public(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


__all__ = ["TeeVerifyRequest", "TeeVerifyResult"]
