"""GpuHostEvidence schema and check result DTOs (M9 design §3.4).

Never stores private key material. At most ``key_fingerprint`` / key_ref
kind+name appear on evidence documents (VAL-GPU-031).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ProbeStatus = Literal["passed", "failed", "error"]
ProbeMode = Literal["full", "quick"]
ProbeTransport = Literal["real", "fake"]


def _iso_z(dt: datetime | None = None) -> str:
    value = dt if dt is not None else datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def new_evidence_id() -> str:
    return str(uuid.uuid4())


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class CheckResult(BaseModel):
    """Single ordered probe check outcome."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    fatal: bool
    passed: bool
    halt: bool = False
    message: str = "ok"
    duration_ms: int = Field(default=0, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)

    def to_public(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ClaimedInventory(BaseModel):
    """Node-row / request-claimed inventory used for match gates."""

    model_config = ConfigDict(extra="forbid")

    gpu_model: str = Field(..., min_length=1)
    gpu_count: int = Field(..., ge=0)


class MeasuredGpu(BaseModel):
    """One GPU row measured from nvidia-smi query."""

    model_config = ConfigDict(extra="forbid")

    name: str
    uuid: str
    memory_total_mb: int | None = None
    driver_version: str | None = None
    power_limit_w: float | None = None
    power_default_w: float | None = None
    util_gpu: float | None = None
    util_mem: float | None = None
    clocks_sm_mhz: float | None = None


class MeasuredInventory(BaseModel):
    """Parsed host GPU inventory."""

    model_config = ConfigDict(extra="forbid")

    gpu_count: int = Field(default=0, ge=0)
    gpus: list[MeasuredGpu] = Field(default_factory=list)
    cuda_runtime_hint: str | None = None
    docker: dict[str, Any] = Field(default_factory=dict)

    def uuid_set(self) -> list[str]:
        return sorted({g.uuid for g in self.gpus if g.uuid})


class ProbeDigests(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inventory_sha256: str | None = None
    microbench_digest: str | None = None
    evidence_sha256: str | None = None


class GpuHostEvidence(BaseModel):
    """Full evidence document for one probe run (design §3.4)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_evidence_id)
    node_id: str | None = None
    provider_hotkey: str | None = None
    ssh_endpoint: str | None = None
    started_at: str = Field(default_factory=_iso_z)
    finished_at: str | None = None
    status: ProbeStatus = "failed"
    mode: ProbeMode = "full"
    transport: ProbeTransport = "fake"
    claimed: ClaimedInventory
    measured: MeasuredInventory = Field(default_factory=MeasuredInventory)
    checks: list[CheckResult] = Field(default_factory=list)
    advisories: list[CheckResult] = Field(default_factory=list)
    digests: ProbeDigests = Field(default_factory=ProbeDigests)
    failure_code: str | None = None
    key_fingerprint: str | None = None
    raw_redacted: dict[str, Any] = Field(default_factory=dict)

    @field_validator("status")
    @classmethod
    def _status_ok(cls, value: str) -> str:
        if value not in {"passed", "failed", "error"}:
            raise ValueError("status must be passed|failed|error")
        return value

    def checks_by_id(self) -> dict[str, CheckResult]:
        return {c.id: c for c in self.checks}

    def fatal_failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.fatal and not c.passed]

    def seal(self) -> GpuHostEvidence:
        """Fill finished_at + digests after pipeline completion."""

        if self.finished_at is None:
            self.finished_at = _iso_z()
        inv_payload = {
            "claimed": self.claimed.model_dump(mode="json"),
            "measured": self.measured.model_dump(mode="json"),
        }
        inv_hash = hashlib.sha256(canonical_json(inv_payload).encode()).hexdigest()
        body = {
            "id": self.id,
            "node_id": self.node_id,
            "status": self.status,
            "claimed": self.claimed.model_dump(mode="json"),
            "measured": self.measured.model_dump(mode="json"),
            "checks": [c.model_dump(mode="json") for c in self.checks],
            "failure_code": self.failure_code,
        }
        evidence_hash = hashlib.sha256(canonical_json(body).encode()).hexdigest()
        self.digests = ProbeDigests(
            inventory_sha256=f"sha256:{inv_hash}",
            microbench_digest=self.digests.microbench_digest,
            evidence_sha256=f"sha256:{evidence_hash}",
        )
        return self

    def to_public(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        # Never expose private-key-shaped keys / PEM values (VAL-GPU-031).
        from hypercluster.probe.redact import redact_mapping

        raw = data.get("raw_redacted") or {}
        if isinstance(raw, dict):
            data["raw_redacted"] = redact_mapping(raw)
        # Belt-and-suspenders: never allow PEM strings on any evidence field.
        return redact_mapping(data)


__all__ = [
    "CheckResult",
    "ClaimedInventory",
    "GpuHostEvidence",
    "MeasuredGpu",
    "MeasuredInventory",
    "ProbeDigests",
    "ProbeMode",
    "ProbeStatus",
    "ProbeTransport",
    "canonical_json",
    "new_evidence_id",
]
