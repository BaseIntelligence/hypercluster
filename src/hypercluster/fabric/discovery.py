"""FabricReport discovery model and canonical digests (architecture §8.1).

Fulfills VAL-FAB-001 (schema + report_digest + topology fields).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DIGEST_PREFIX = "sha256:"


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON for digests (sorted keys, compact separators)."""

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_report_digest(payload: dict[str, Any]) -> str:
    """sha256: hex digest of permanent report fields (report_digest excluded)."""

    body = {k: v for k, v in payload.items() if k != "report_digest"}
    return DIGEST_PREFIX + hashlib.sha256(canonical_json(body).encode()).hexdigest()


def gpu_topo_sha256_from_matrix(matrix: str) -> str:
    return hashlib.sha256(matrix.encode()).hexdigest()


class IbDevice(BaseModel):
    """InfiniBand HCA / port entry."""

    name: str = Field(..., min_length=1)
    port: int = Field(default=1, ge=1)
    rate_gbps: float = Field(..., ge=0)
    state: str = Field(default="Active")
    guid: str | None = None
    lid: int | None = None

    def to_public(self) -> dict[str, Any]:
        data = self.model_dump()
        return {k: v for k, v in data.items() if v is not None}


class FabricReport(BaseModel):
    """Node fabric self-report (architecture FabricReport schema).

    Required permanent fields for acceptance (VAL-FAB-001):
    ``node_id``, ``collected_at``, ``ib_devices``, ``gpu_topo_sha256``,
    ``report_digest``.
    """

    node_id: str = Field(..., min_length=1)
    collected_at: datetime
    ib_devices: list[IbDevice] = Field(default_factory=list)
    ib_rate_gbps: float | None = None
    gpu_gpu_topo_matrix: str = ""
    gpu_topo_sha256: str = Field(..., min_length=1)
    numa_map: dict[str, int] = Field(default_factory=dict)
    nccl_version: str | None = None
    eth_ifaces: list[str] = Field(default_factory=list)
    report_digest: str = Field(..., min_length=len(DIGEST_PREFIX) + 8)
    gpu_count: int = Field(default=0, ge=0)
    source: Literal["sim", "scan", "inject", "manual"] = "sim"

    @field_validator("report_digest")
    @classmethod
    def _digest_shape(cls, value: str) -> str:
        if not value.startswith(DIGEST_PREFIX):
            raise ValueError("report_digest must start with 'sha256:'")
        hexpart = value[len(DIGEST_PREFIX) :]
        if len(hexpart) != 64 or any(c not in "0123456789abcdef" for c in hexpart.lower()):
            raise ValueError("report_digest must be sha256:<64 hex chars>")
        return value

    @field_validator("gpu_topo_sha256")
    @classmethod
    def _topo_sha_nonempty(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("gpu_topo_sha256 is required")
        return value

    @model_validator(mode="after")
    def _fill_ib_rate(self) -> FabricReport:
        if self.ib_rate_gbps is None and self.ib_devices:
            rates = [d.rate_gbps for d in self.ib_devices if d.rate_gbps is not None]
            if rates:
                object.__setattr__(self, "ib_rate_gbps", max(rates))
        return self

    def to_public(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "collected_at": _iso_z(self.collected_at),
            "ib_devices": [d.to_public() for d in self.ib_devices],
            "ib_rate_gbps": self.ib_rate_gbps,
            "gpu_gpu_topo_matrix": self.gpu_gpu_topo_matrix,
            "gpu_topo_sha256": self.gpu_topo_sha256,
            "numa_map": self.numa_map,
            "nccl_version": self.nccl_version,
            "eth_ifaces": list(self.eth_ifaces),
            "report_digest": self.report_digest,
            "gpu_count": self.gpu_count,
            "source": self.source,
        }


def canonical_report_payload(report: FabricReport) -> dict[str, Any]:
    """Canonical field set used for report_digest (excludes digest itself)."""

    return {
        "node_id": report.node_id,
        "collected_at": _iso_z(report.collected_at),
        "ib_devices": [d.to_public() for d in report.ib_devices],
        "ib_rate_gbps": report.ib_rate_gbps,
        "gpu_gpu_topo_matrix": report.gpu_gpu_topo_matrix,
        "gpu_topo_sha256": report.gpu_topo_sha256,
        "numa_map": report.numa_map,
        "nccl_version": report.nccl_version,
        "eth_ifaces": list(report.eth_ifaces),
        "gpu_count": report.gpu_count,
        "source": report.source,
    }


def build_fabric_report(
    *,
    node_id: str,
    collected_at: datetime | None = None,
    ib_devices: list[IbDevice] | list[dict[str, Any]] | None = None,
    ib_rate_gbps: float | None = None,
    gpu_gpu_topo_matrix: str = "",
    numa_map: dict[str, int] | None = None,
    nccl_version: str | None = "sim-2.21.5",
    eth_ifaces: list[str] | None = None,
    gpu_count: int = 0,
    source: Literal["sim", "scan", "inject", "manual"] = "sim",
    force_empty_topo_sha: bool = False,
    report_digest: str | None = None,
) -> FabricReport:
    """Build a FabricReport, computing gpu_topo_sha256 + report_digest when needed."""

    when = collected_at or datetime.now(UTC)
    devices: list[IbDevice] = []
    for item in ib_devices or []:
        if isinstance(item, IbDevice):
            devices.append(item)
        else:
            devices.append(IbDevice.model_validate(item))

    matrix = gpu_gpu_topo_matrix or ""
    if force_empty_topo_sha:
        topo_sha = ""
    else:
        topo_sha = gpu_topo_sha256_from_matrix(matrix) if matrix else (
            # Single-GPU / empty matrix still needs a nonempty digest input for schema.
            hashlib.sha256(f"empty-topo:{node_id}:{gpu_count}".encode()).hexdigest()
        )

    rate = ib_rate_gbps
    if rate is None and devices:
        rate = max(d.rate_gbps for d in devices)

    interim = {
        "node_id": node_id,
        "collected_at": _iso_z(when),
        "ib_devices": [d.to_public() for d in devices],
        "ib_rate_gbps": rate,
        "gpu_gpu_topo_matrix": matrix,
        "gpu_topo_sha256": topo_sha,
        "numa_map": numa_map or {},
        "nccl_version": nccl_version,
        "eth_ifaces": list(eth_ifaces or []),
        "gpu_count": gpu_count,
        "source": source,
    }
    digest = report_digest or compute_report_digest(interim)

    # Bypass empty topo validation when testing reject path.
    if force_empty_topo_sha:
        return FabricReport.model_construct(
            node_id=node_id,
            collected_at=when,
            ib_devices=devices,
            ib_rate_gbps=rate,
            gpu_gpu_topo_matrix=matrix,
            gpu_topo_sha256=topo_sha,
            numa_map=numa_map or {},
            nccl_version=nccl_version,
            eth_ifaces=list(eth_ifaces or []),
            report_digest=digest if digest.startswith(DIGEST_PREFIX) else DIGEST_PREFIX + digest,
            gpu_count=gpu_count,
            source=source,
        )

    return FabricReport(
        node_id=node_id,
        collected_at=when,
        ib_devices=devices,
        ib_rate_gbps=rate,
        gpu_gpu_topo_matrix=matrix,
        gpu_topo_sha256=topo_sha,
        numa_map=numa_map or {},
        nccl_version=nccl_version,
        eth_ifaces=list(eth_ifaces or []),
        report_digest=digest,
        gpu_count=gpu_count,
        source=source,
    )


def validate_accepted_report(report: FabricReport, *, gpu_count: int | None = None) -> FabricReport:
    """Accept-path guards: digest matches payload; multi-GPU requires topo sha.

    Raises ValueError on policy fail (VAL-FAB-001).
    """

    gpus = gpu_count if gpu_count is not None else report.gpu_count
    if not report.report_digest:
        raise ValueError("report_digest is required")
    if not report.gpu_topo_sha256 or not str(report.gpu_topo_sha256).strip():
        raise ValueError("gpu_topo_sha256 is required")
    if gpus is not None and gpus > 1 and not report.gpu_topo_sha256.strip():
        raise ValueError("gpu_topo_sha256 required for multi-GPU claims")

    payload = canonical_report_payload(report)
    expected = compute_report_digest(payload)
    if report.report_digest != expected:
        # Allow caller-supplied digests when payload was sealed earlier with same
        # logical content under alternate datetime serialization — re-check strict.
        raise ValueError(
            f"report_digest mismatch: got {report.report_digest}, expected {expected}"
        )
    return report


def synthetic_nvlink_topo_matrix(gpu_count: int) -> str:
    """nvidia-smi topo -m style synthetic matrix with NVLink among GPUs."""

    if gpu_count <= 0:
        return ""
    lines: list[str] = []
    header = "\t".join(["GPU"] + [f"GPU{i}" for i in range(gpu_count)])
    lines.append(header)
    for i in range(gpu_count):
        cells: list[str] = [f"GPU{i}"]
        for j in range(gpu_count):
            if i == j:
                cells.append("X")
            else:
                # Dense NVLink within a node.
                cells.append("NV12")
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def synthetic_ib_devices(
    *,
    node_index: int = 0,
    count: int = 1,
    rate_gbps: float = 200.0,
) -> list[IbDevice]:
    devices: list[IbDevice] = []
    for i in range(max(0, count)):
        devices.append(
            IbDevice(
                name=f"mlx5_{i}",
                port=1,
                rate_gbps=rate_gbps,
                state="Active",
                guid=f"0x{node_index:04x}{i:04x}sim",
                lid=100 + node_index * 10 + i,
            )
        )
    return devices


__all__ = [
    "DIGEST_PREFIX",
    "FabricReport",
    "IbDevice",
    "build_fabric_report",
    "canonical_json",
    "canonical_report_payload",
    "compute_report_digest",
    "gpu_topo_sha256_from_matrix",
    "synthetic_ib_devices",
    "synthetic_nvlink_topo_matrix",
    "validate_accepted_report",
]
