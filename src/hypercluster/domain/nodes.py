"""Node register / inventory / heartbeat domain service."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import Node, utc_now
from hypercluster.domain.providers import get_provider_by_hotkey


class NodeError(Exception):
    """Domain error for node operations."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


# Default heart-beat miss window when status should go offline (seconds).
DEFAULT_NODE_LIVENESS_SECONDS = 120


def _extract_ib_flags(inventory: dict[str, Any] | None) -> tuple[bool, float | None]:
    """Derive has_ib + ib_rate_gbps from inventory payload.

    Accept several self-report shapes so later fabric-scan and knowledge of
    require_ib offers (VAL-MKT-005) can reject non-IB nodes at offer create.
    """

    if not inventory:
        return False, None

    has_ib = False
    rate: float | None = None

    if inventory.get("has_ib") is True or inventory.get("ib") is True:
        has_ib = True
    if "ib_rate_gbps" in inventory and inventory["ib_rate_gbps"] is not None:
        try:
            rate = float(inventory["ib_rate_gbps"])
            if rate > 0:
                has_ib = True
        except (TypeError, ValueError):
            rate = None

    devices = inventory.get("ib_devices")
    if isinstance(devices, list) and len(devices) > 0:
        has_ib = True
        for dev in devices:
            if isinstance(dev, dict) and dev.get("rate_gbps") is not None:
                try:
                    state = str(dev.get("state", "ACTIVE")).upper()
                    if state in {"", "ACTIVE", "UP"}:
                        rate = max(rate or 0.0, float(dev["rate_gbps"]))
                except (TypeError, ValueError):
                    continue

    return has_ib, rate


async def register_node(
    session: AsyncSession,
    *,
    hotkey: str,
    gpu_model: str,
    gpu_count: int,
    hostname: str | None = None,
    ssh_endpoint: str | None = None,
    cpu_cores: int | None = None,
    mem_gb: float | None = None,
    location_hint: str | None = None,
    tee_capability: str = "none",
    inventory: dict[str, Any] | None = None,
    node_id: str | None = None,
) -> tuple[Node, bool]:
    """Create or update a node owned by the provider hotkey.

    ``gpu_count`` must be a positive integer (VAL-MKT-004). IB capability is
    derived from ``inventory`` (VAL-MKT-005).
    """

    if not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count < 1:
        raise NodeError(
            "invalid_gpu_count",
            "gpu_count must be a positive integer",
            status_code=422,
        )
    if not gpu_model or not str(gpu_model).strip():
        raise NodeError("invalid_gpu_model", "gpu_model is required", status_code=422)

    provider = await get_provider_by_hotkey(session, hotkey)
    if provider is None:
        raise NodeError(
            "provider_not_found",
            "register provider before registering nodes",
            status_code=404,
        )
    if provider.status in {"suspended", "banned"}:
        raise NodeError(
            "provider_not_active",
            f"provider status is {provider.status}",
            status_code=403,
        )

    has_ib, ib_rate = _extract_ib_flags(inventory)
    inventory_json = json.dumps(inventory) if inventory is not None else None
    tee = (tee_capability or "none").strip() or "none"
    now = utc_now()

    existing: Node | None = None
    if node_id:
        result = await session.execute(select(Node).where(Node.id == node_id))
        existing = result.scalar_one_or_none()
        if existing is not None and existing.provider_id != provider.id:
            raise NodeError(
                "node_not_owned",
                "node belongs to another provider",
                status_code=403,
            )

    if existing is None and ssh_endpoint:
        # Soft upsert by (provider, ssh_endpoint) so register is durable.
        result = await session.execute(
            select(Node).where(
                Node.provider_id == provider.id,
                Node.ssh_endpoint == ssh_endpoint,
            )
        )
        existing = result.scalar_one_or_none()

    if existing is not None:
        existing.hostname = hostname if hostname is not None else existing.hostname
        existing.ssh_endpoint = (
            ssh_endpoint if ssh_endpoint is not None else existing.ssh_endpoint
        )
        existing.gpu_model = str(gpu_model).strip()
        existing.gpu_count = gpu_count
        existing.cpu_cores = cpu_cores if cpu_cores is not None else existing.cpu_cores
        existing.mem_gb = mem_gb if mem_gb is not None else existing.mem_gb
        existing.location_hint = (
            location_hint if location_hint is not None else existing.location_hint
        )
        existing.tee_capability = tee
        if inventory is not None:
            existing.inventory_json = inventory_json
            existing.has_ib = 1 if has_ib else 0
            existing.ib_rate_gbps = ib_rate
        existing.last_heartbeat = now
        existing.updated_at = now
        if existing.status in {"offline", "draining", "registered"}:
            existing.status = "healthy"
        await session.commit()
        await session.refresh(existing)
        return existing, False

    node = Node(
        id=node_id or str(uuid.uuid4()),
        provider_id=provider.id,
        hostname=hostname,
        ssh_endpoint=ssh_endpoint,
        gpu_model=str(gpu_model).strip(),
        gpu_count=gpu_count,
        cpu_cores=cpu_cores,
        mem_gb=mem_gb,
        location_hint=location_hint,
        tee_capability=tee,
        status="healthy",
        last_heartbeat=now,
        inventory_json=inventory_json,
        has_ib=1 if has_ib else 0,
        ib_rate_gbps=ib_rate,
        created_at=now,
        updated_at=now,
    )
    session.add(node)
    await session.commit()
    await session.refresh(node)
    return node, True


async def list_nodes(
    session: AsyncSession,
    *,
    hotkey: str | None = None,
    provider_id: str | None = None,
) -> list[Node]:
    """List nodes, optionally scoped to a provider hotkey or id."""

    stmt = select(Node).order_by(Node.created_at.asc())
    if provider_id:
        stmt = stmt.where(Node.provider_id == provider_id)
    elif hotkey:
        provider = await get_provider_by_hotkey(session, hotkey)
        if provider is None:
            return []
        stmt = stmt.where(Node.provider_id == provider.id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_node(session: AsyncSession, node_id: str) -> Node | None:
    result = await session.execute(select(Node).where(Node.id == node_id))
    return result.scalar_one_or_none()


async def node_heartbeat(
    session: AsyncSession,
    *,
    hotkey: str,
    node_id: str | None = None,
    liveness_seconds: int = DEFAULT_NODE_LIVENESS_SECONDS,
) -> list[Node]:
    """Refresh last_heartbeat for owned node(s).

    When ``node_id`` is omitted, all nodes for the provider are heartbeated.
    Nodes inside the liveness window become/remain ``healthy`` unless already
    ``rented`` (VAL-MKT-006).
    """

    provider = await get_provider_by_hotkey(session, hotkey)
    if provider is None:
        raise NodeError(
            "provider_not_found",
            "provider not registered for hotkey",
            status_code=404,
        )

    if node_id:
        result = await session.execute(select(Node).where(Node.id == node_id))
        node = result.scalar_one_or_none()
        if node is None:
            raise NodeError("node_not_found", "node not found", status_code=404)
        if node.provider_id != provider.id:
            raise NodeError(
                "node_not_owned",
                "node belongs to another provider",
                status_code=403,
            )
        nodes = [node]
    else:
        result = await session.execute(
            select(Node).where(Node.provider_id == provider.id)
        )
        nodes = list(result.scalars().all())
        if not nodes:
            raise NodeError("no_nodes", "provider has no registered nodes", status_code=404)

    now = utc_now()
    for node in nodes:
        node.last_heartbeat = now
        node.updated_at = now
        # Keep rented status; otherwise promote to healthy while heartbeating.
        if node.status not in {"rented", "banned"}:
            node.status = "healthy"

    # Also refresh provider last_seen as part of node heartbeats.
    provider.last_seen_at = now
    provider.updated_at = now
    if provider.status not in {"suspended", "banned"}:
        provider.status = "active"

    await session.commit()
    for node in nodes:
        await session.refresh(node)
    return nodes


async def mark_stale_nodes_offline(
    session: AsyncSession,
    *,
    liveness_seconds: int = DEFAULT_NODE_LIVENESS_SECONDS,
) -> int:
    """Optional sweep: nodes past miss threshold without rental → offline.

    Does not touch ``rented`` nodes (tenant short-circuit ownership for later
    VAL-MKT-014/020); only unaffected inventory.
    """

    now = utc_now()
    cutoff = now - timedelta(seconds=liveness_seconds)
    result = await session.execute(
        select(Node).where(
            Node.status.in_(("healthy", "registered", "draining")),
            Node.last_heartbeat.is_not(None),
            Node.last_heartbeat < cutoff,
        )
    )
    stale = list(result.scalars().all())
    for node in stale:
        node.status = "offline"
        node.updated_at = now
    if stale:
        await session.commit()
    return len(stale)


def node_to_public(node: Node) -> dict[str, Any]:
    return node.to_dict()


def node_has_ib(node: Node) -> bool:
    """Whether node inventory declares InfiniBand capability."""

    return bool(node.has_ib) or (node.ib_rate_gbps is not None and node.ib_rate_gbps > 0)


__all__ = [
    "DEFAULT_NODE_LIVENESS_SECONDS",
    "NodeError",
    "get_node",
    "list_nodes",
    "mark_stale_nodes_offline",
    "node_has_ib",
    "node_heartbeat",
    "node_to_public",
    "register_node",
]
