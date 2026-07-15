"""SQLAlchemy models for marketplace identity (providers / nodes / offers).

Schema aligns with architecture.md §3.1 (providers, nodes, offers, nonces).
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
    "Node",
    "Offer",
    "Provider",
    "RequestNonce",
    "isoformat_utc",
    "utc_now",
]
