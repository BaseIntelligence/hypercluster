"""SQLAlchemy models for marketplace identity and rentals.

Schema aligns with architecture.md §3.1 (providers, nodes, offers, leases, pods, nonces).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hypercluster.db.database import Base


def utc_now() -> datetime:
    """Timezone-aware UTC now for model defaults."""

    return datetime.now(UTC)


def isoformat_utc(value: datetime | None) -> str | None:
    """Serialize datetimes as ISO-8601 UTC strings for API responses."""

    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


class Provider(Base):
    """Miner hotkey onboarded as a capacity supplier."""

    __tablename__ = "providers"
    __table_args__ = (UniqueConstraint("hotkey", name="uq_providers_hotkey"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    hotkey: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    # Liveness clock advanced by heartbeat; identity fields must not change.
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    nodes: Mapped[list[Node]] = relationship(
        "Node",
        back_populates="provider",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hotkey": self.hotkey,
            "display_name": self.display_name,
            "status": self.status,
            "last_seen_at": isoformat_utc(self.last_seen_at),
            "created_at": isoformat_utc(self.created_at),
            "updated_at": isoformat_utc(self.updated_at),
        }


class Node(Base):
    """GPU host offered by a provider (home-grown inventory)."""

    __tablename__ = "nodes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    provider_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("providers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    hostname: Mapped[str | None] = mapped_column(String(256), nullable=True)
    ssh_endpoint: Mapped[str | None] = mapped_column(String(512), nullable=True)
    gpu_model: Mapped[str] = mapped_column(String(128), nullable=False, default="unknown")
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cpu_cores: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mem_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    location_hint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tee_capability: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    # registered|healthy|draining|offline|rented
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="registered")
    last_heartbeat: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    # Raw discovery blob (IB devices, rates, topo flags) as JSON text.
    inventory_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Denormalized IB capability flags for offer/require_ib guards (M2+).
    has_ib: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ib_rate_gbps: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    provider: Mapped[Provider] = relationship("Provider", back_populates="nodes")

    def to_dict(self) -> dict[str, Any]:
        inventory: Any = None
        if self.inventory_json:
            import json

            try:
                inventory = json.loads(self.inventory_json)
            except (TypeError, ValueError):
                inventory = self.inventory_json
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "hostname": self.hostname,
            "ssh_endpoint": self.ssh_endpoint,
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "cpu_cores": self.cpu_cores,
            "mem_gb": self.mem_gb,
            "location_hint": self.location_hint,
            "tee_capability": self.tee_capability,
            "status": self.status,
            "last_heartbeat": isoformat_utc(self.last_heartbeat),
            "has_ib": bool(self.has_ib),
            "ib_rate_gbps": self.ib_rate_gbps,
            "inventory": inventory,
            "created_at": isoformat_utc(self.created_at),
            "updated_at": isoformat_utc(self.updated_at),
        }


class Offer(Base):
    """Home-grown marketplace listing (Lium-shaped capacity snapshot).

    Status lifecycle (M2): listed → withdrawn | leased | expired.
    Price/lifetime hard guards enforced in domain layer (VAL-MKT-008..011).
    """

    __tablename__ = "offers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    provider_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("providers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # JSON list of node id strings (one node or multi-node cluster set).
    node_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="single")
    gpu_model: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False)
    node_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    require_ib: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tee: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    price_per_hour: Mapped[float] = mapped_column(Float, nullable=False)
    max_lifetime_hours: Mapped[float] = mapped_column(Float, nullable=False)
    location_hint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # listed|leased|expired|withdrawn
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="listed", index=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    def node_ids(self) -> list[str]:
        try:
            raw = json.loads(self.node_ids_json)
        except (TypeError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        return [str(x) for x in raw]

    def to_dict(self) -> dict[str, Any]:
        meta: Any = None
        if self.metadata_json:
            try:
                meta = json.loads(self.metadata_json)
            except (TypeError, ValueError):
                meta = self.metadata_json
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "node_ids": self.node_ids(),
            "mode": self.mode,
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "node_count": self.node_count,
            "require_ib": bool(self.require_ib),
            "tee": self.tee,
            "price_per_hour": float(self.price_per_hour),
            "max_lifetime_hours": float(self.max_lifetime_hours),
            "location_hint": self.location_hint,
            "status": self.status,
            "metadata": meta,
            "created_at": isoformat_utc(self.created_at),
            "updated_at": isoformat_utc(self.updated_at),
        }


class Lease(Base):
    """Time-bounded rental of an offer (architecture §3.1 / §7.3).

    Status lifecycle: requested → active → terminated | expired | failed.
    Active rentals are protected from idle-only reclaim (VAL-MKT-020).
    """

    __tablename__ = "leases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    offer_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("offers.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    renter_hotkey: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    provider_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("providers.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    # requested|active|expired|terminated|failed
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="requested", index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    price_per_hour: Mapped[float] = mapped_column(Float, nullable=False)
    lifetime_hours: Mapped[float] = mapped_column(Float, nullable=False)
    termination_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "offer_id": self.offer_id,
            "renter_hotkey": self.renter_hotkey,
            "provider_id": self.provider_id,
            "status": self.status,
            "started_at": isoformat_utc(self.started_at),
            "ends_at": isoformat_utc(self.ends_at),
            "price_per_hour": float(self.price_per_hour),
            "lifetime_hours": float(self.lifetime_hours),
            "termination_reason": self.termination_reason,
            "created_at": isoformat_utc(self.created_at),
            "updated_at": isoformat_utc(self.updated_at),
        }


class Pod(Base):
    """Runtime binding for a lease (single node or multi-node cluster).

    Status lifecycle: provisioning → running → stopping → stopped | error.
    Local sim promotes provisioning → running on rent (VAL-MKT-017).
    """

    __tablename__ = "pods"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    lease_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("leases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,
    )
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="single")
    # provisioning|running|stopping|stopped|error
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="provisioning")
    node_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    image_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ssh_authorized_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    endpoints_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    def node_ids(self) -> list[str]:
        try:
            raw = json.loads(self.node_ids_json)
        except (TypeError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        return [str(x) for x in raw]

    def endpoints(self) -> Any:
        if not self.endpoints_json:
            return None
        try:
            return json.loads(self.endpoints_json)
        except (TypeError, ValueError):
            return self.endpoints_json

    def to_dict(self) -> dict[str, Any]:
        auth: Any = None
        if self.ssh_authorized_json:
            try:
                auth = json.loads(self.ssh_authorized_json)
            except (TypeError, ValueError):
                auth = self.ssh_authorized_json
        return {
            "id": self.id,
            "lease_id": self.lease_id,
            "mode": self.mode,
            "status": self.status,
            "node_ids": self.node_ids(),
            "image_digest": self.image_digest,
            "ssh_authorized": auth,
            "endpoints": self.endpoints(),
            "created_at": isoformat_utc(self.created_at),
            "updated_at": isoformat_utc(self.updated_at),
        }


class Job(Base):
    """Modal-like HyperJob lifecycle row (architecture §3.1 / §6).

    Admit-phase statuses: submitted → admitted (static gates).
    Later M3 features advance: placing → provisioning → running → collecting
    → scoring → terminal (succeeded|failed|cancelled|timeout).
    Idempotency: unique (submitter_hotkey, client_request_id) when key present.
    """

    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint(
            "submitter_hotkey",
            "client_request_id",
            name="uq_jobs_hotkey_client_request_id",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    submitter_hotkey: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    # Idempotency key from client; NULL when not supplied (SQLite UNIQUE
    # treats NULLs as distinct so non-idempotent creates stay independent).
    client_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True, default=None)
    # admitted|placing|provisioning|running|collecting|scoring|
    # succeeded|failed|cancelled|timeout (submitted may appear pre-admit in flight)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="admitted", index=True)
    image_digest: Mapped[str] = mapped_column(String(256), nullable=False)
    entrypoint_json: Mapped[str] = mapped_column(Text, nullable=False)
    world_size: Mapped[int] = mapped_column(Integer, nullable=False)
    nnodes: Mapped[int] = mapped_column(Integer, nullable=False)
    nproc_per_node: Mapped[int] = mapped_column(Integer, nullable=False)
    backend: Mapped[str] = mapped_column(String(32), nullable=False, default="nccl")
    fabric_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="auto")
    tee_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    env_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    resource_json: Mapped[str] = mapped_column(Text, nullable=False)
    timeout_s: Mapped[int] = mapped_column(Integer, nullable=False)
    placement_policy: Mapped[str] = mapped_column(String(16), nullable=False, default="pack")
    lease_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("leases.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    pod_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("pods.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    admitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    def entrypoint(self) -> list[str]:
        try:
            raw = json.loads(self.entrypoint_json)
        except (TypeError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        return [str(x) for x in raw]

    def env(self) -> dict[str, str] | None:
        if not self.env_json:
            return None
        try:
            raw = json.loads(self.env_json)
        except (TypeError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        return {str(k): str(v) for k, v in raw.items()}

    def resource(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.resource_json)
        except (TypeError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.id,
            "submitter_hotkey": self.submitter_hotkey,
            "client_request_id": self.client_request_id or None,
            "status": self.status,
            "image_digest": self.image_digest,
            "entrypoint": self.entrypoint(),
            "world_size": int(self.world_size),
            "nnodes": int(self.nnodes),
            "nproc_per_node": int(self.nproc_per_node),
            "backend": self.backend,
            "fabric": self.fabric_mode,
            "fabric_mode": self.fabric_mode,
            "tee": self.tee_mode,
            "tee_mode": self.tee_mode,
            "env": self.env(),
            "resource": self.resource(),
            "timeout_s": int(self.timeout_s),
            "placement_policy": self.placement_policy,
            "lease_id": self.lease_id,
            "pod_id": self.pod_id,
            "admitted_at": isoformat_utc(self.admitted_at),
            "started_at": isoformat_utc(self.started_at),
            "finished_at": isoformat_utc(self.finished_at),
            "failure_code": self.failure_code,
            "created_at": isoformat_utc(self.created_at),
            "updated_at": isoformat_utc(self.updated_at),
        }


class JobPlacement(Base):
    """Topology-aware rankmap + NCCL env for a job (architecture §3.1 / §8.2)."""

    __tablename__ = "job_placements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rankmap_json: Mapped[str] = mapped_column(Text, nullable=False)
    placement_policy: Mapped[str] = mapped_column(String(16), nullable=False, default="pack")
    nccl_env_json: Mapped[str] = mapped_column(Text, nullable=False)
    planner_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="fabric-planner.v1",
    )
    # Launch contract merge (entrypoint + env + NCCL + image) for observability.
    launch_contract_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    graph_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    def rankmap(self) -> list[Any]:
        try:
            raw = json.loads(self.rankmap_json)
        except (TypeError, ValueError):
            return []
        return raw if isinstance(raw, list) else []

    def nccl_env(self) -> dict[str, str]:
        try:
            raw = json.loads(self.nccl_env_json)
        except (TypeError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items()}

    def launch_contract(self) -> dict[str, Any] | None:
        if not self.launch_contract_json:
            return None
        try:
            raw = json.loads(self.launch_contract_json)
        except (TypeError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "rankmap": self.rankmap(),
            "placement_policy": self.placement_policy,
            "nccl_env": self.nccl_env(),
            "planner_version": self.planner_version,
            "launch_contract": self.launch_contract(),
            "graph_digest": self.graph_digest,
            "created_at": isoformat_utc(self.created_at),
        }


class JobAttempt(Base):
    """Single launch attempt for a job (unique job_id + attempt_no)."""

    __tablename__ = "job_attempts"
    __table_args__ = (
        UniqueConstraint("job_id", "attempt_no", name="uq_job_attempts_job_attempt_no"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    launcher_log_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)
    fabric_report_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Fingerprint of for provider result envelope (idempotent re-post).
    result_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    launch_contract_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def metrics(self) -> dict[str, Any] | None:
        if not self.metrics_json:
            return None
        try:
            raw = json.loads(self.metrics_json)
        except (TypeError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    def launch_contract(self) -> dict[str, Any] | None:
        if not self.launch_contract_json:
            return None
        try:
            raw = json.loads(self.launch_contract_json)
        except (TypeError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "attempt_no": int(self.attempt_no),
            "status": self.status,
            "launcher_log_uri": self.launcher_log_uri,
            "fabric_report_digest": self.fabric_report_digest,
            "metrics": self.metrics(),
            "output_digest": self.output_digest,
            "result_digest": self.result_digest,
            "failure_code": self.failure_code,
            "launch_contract": self.launch_contract(),
            "started_at": isoformat_utc(self.started_at),
            "finished_at": isoformat_utc(self.finished_at),
        }


class JobProof(Base):
    """Execution / TEE proof attached to an attempt (secrets never stored here)."""

    __tablename__ = "job_proofs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    attempt_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("job_attempts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    proof_tier: Mapped[str] = mapped_column(String(32), nullable=False, default="sim")
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    tdx_quote_b64: Mapped[str | None] = mapped_column(Text, nullable=True)
    gpu_evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    dstack_verdict_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    verified: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    verify_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="sim")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    def to_public_summary(self) -> dict[str, Any]:
        """Summary shape without raw quotes or secret material."""

        return {
            "id": self.id,
            "attempt_id": self.attempt_id,
            "proof_tier": self.proof_tier,
            "verified": bool(self.verified),
            "verify_mode": self.verify_mode,
            "created_at": isoformat_utc(self.created_at),
        }


class JobFabricReport(Base):
    """Per-job fabric report view (sim multi-node digests; architecture §8.1)."""

    __tablename__ = "job_fabric_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,
    )
    attempt_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    ib_devices_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    ib_rate_gbps: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_topo_sha256: Mapped[str | None] = mapped_column(String(128), nullable=True)
    numa_map_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    nccl_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    report_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    raw_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    def to_dict(self) -> dict[str, Any]:
        try:
            ib_devices = json.loads(self.ib_devices_json)
        except (TypeError, ValueError):
            ib_devices = []
        numa_map: Any = None
        if self.numa_map_json:
            try:
                numa_map = json.loads(self.numa_map_json)
            except (TypeError, ValueError):
                numa_map = self.numa_map_json
        return {
            "id": self.id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "collected_at": isoformat_utc(self.collected_at),
            "ib_devices": ib_devices if isinstance(ib_devices, list) else [],
            "ib_rate_gbps": self.ib_rate_gbps,
            "gpu_topo_sha256": self.gpu_topo_sha256,
            "numa_map": numa_map,
            "nccl_version": self.nccl_version,
            "report_digest": self.report_digest,
            "fabric_report_digest": self.report_digest,
            "created_at": isoformat_utc(self.created_at),
        }


class FabricReportRow(Base):
    """Per-node fabric self-report (architecture §3.1 fabric_reports).

    Distinct from JobFabricReport (per-job collected digests). This table is
    the durable inventory scan store used by fabric-scan (VAL-FAB-001/018).
    """

    __tablename__ = "fabric_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    ib_devices_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    ib_rate_gbps: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_topo_sha256: Mapped[str | None] = mapped_column(String(128), nullable=True)
    numa_map_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    nccl_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    report_digest: Mapped[str] = mapped_column(String(128), nullable=False)
    raw_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    def to_dict(self) -> dict[str, Any]:
        try:
            ib_devices = json.loads(self.ib_devices_json)
        except (TypeError, ValueError):
            ib_devices = []
        numa_map: Any = None
        if self.numa_map_json:
            try:
                numa_map = json.loads(self.numa_map_json)
            except (TypeError, ValueError):
                numa_map = self.numa_map_json
        raw: Any = {}
        if self.raw_json:
            try:
                raw = json.loads(self.raw_json)
            except (TypeError, ValueError):
                raw = {}
        return {
            "id": self.id,
            "node_id": self.node_id,
            "collected_at": isoformat_utc(self.collected_at),
            "ib_devices": ib_devices if isinstance(ib_devices, list) else [],
            "ib_rate_gbps": self.ib_rate_gbps,
            "gpu_topo_sha256": self.gpu_topo_sha256,
            "numa_map": numa_map,
            "nccl_version": self.nccl_version,
            "report_digest": self.report_digest,
            "raw": raw if isinstance(raw, dict) else {},
            "created_at": isoformat_utc(self.created_at),
        }


class RequestNonce(Base):
    """Replay protection for signed miner requests."""

    __tablename__ = "nonces"
    __table_args__ = (UniqueConstraint("nonce", "hotkey", name="uq_nonces_hotkey_nonce"),)

    nonce: Mapped[str] = mapped_column(String(128), primary_key=True)
    hotkey: Mapped[str] = mapped_column(String(128), primary_key=True)
    purpose: Mapped[str] = mapped_column(String(64), nullable=False, default="request")
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


__all__ = [
    "FabricReportRow",
    "Job",
    "JobAttempt",
    "JobFabricReport",
    "JobPlacement",
    "JobProof",
    "Lease",
    "Node",
    "Offer",
    "Pod",
    "Provider",
    "RequestNonce",
    "isoformat_utc",
    "utc_now",
]
