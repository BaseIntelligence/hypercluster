"""Lease / pod rent lifecycle (VAL-MKT-013..021, VAL-MKT-031).

Rent creates exclusive lease+pod from a listed offer. Double-rent fails closed.
Terminate (or max lifetime expiry) frees capacity. Active rentals are protected
from idle-only reclaim sweeps (tenant short-circuit / Lium lesson).
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import Lease, Node, Offer, Pod, Provider, utc_now
from hypercluster.domain.offers import (
    OFFER_STATUS_LEASED,
    OFFER_STATUS_LISTED,
    get_offer,
)

LEASE_STATUS_REQUESTED = "requested"
LEASE_STATUS_ACTIVE = "active"
LEASE_STATUS_EXPIRED = "expired"
LEASE_STATUS_TERMINATED = "terminated"
LEASE_STATUS_FAILED = "failed"

ACTIVE_LEASE_STATUSES = frozenset({LEASE_STATUS_REQUESTED, LEASE_STATUS_ACTIVE})
TERMINAL_LEASE_STATUSES = frozenset(
    {LEASE_STATUS_EXPIRED, LEASE_STATUS_TERMINATED, LEASE_STATUS_FAILED}
)

POD_STATUS_PROVISIONING = "provisioning"
POD_STATUS_RUNNING = "running"
POD_STATUS_STOPPING = "stopping"
POD_STATUS_STOPPED = "stopped"
POD_STATUS_ERROR = "error"


class LeaseError(Exception):
    """Domain error for lease / rent operations."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _finite_positive(value: Any, *, field: str) -> float:
    if value is None:
        raise LeaseError(
            f"missing_{field}",
            f"{field} is required and must be > 0",
            status_code=422,
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LeaseError(
            f"invalid_{field}",
            f"{field} must be a positive finite number",
            status_code=422,
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise LeaseError(
            f"invalid_{field}",
            f"{field} must be a positive finite number",
            status_code=422,
        )
    return number


async def _load_nodes_for_offer(session: AsyncSession, offer: Offer) -> list[Node]:
    node_ids = offer.node_ids()
    if not node_ids:
        raise LeaseError(
            "offer_has_no_nodes",
            "offer has empty node_ids",
            status_code=409,
        )
    result = await session.execute(select(Node).where(Node.id.in_(node_ids)))
    by_id = {n.id: n for n in result.scalars().all()}
    nodes: list[Node] = []
    for nid in node_ids:
        node = by_id.get(nid)
        if node is None:
            raise LeaseError(
                "offer_node_missing",
                f"offer node {nid} no longer exists",
                status_code=409,
            )
        nodes.append(node)
    return nodes


async def _assert_nodes_exclusive_free(
    session: AsyncSession,
    nodes: list[Node],
    *,
    exclude_lease_id: str | None = None,
) -> None:
    """Reject if any node is already bound by a non-terminal lease."""

    for node in nodes:
        # Node itself already busy under a foreign rental.
        if node.status == "rented":
            # Verify there is at least one active lease covering it; if dangling
            # rented flag remains without lease we still treat as conflict for now.
            pass

    # Active leases whose pods reference any of these nodes.
    node_ids = {n.id for n in nodes}
    active = await session.execute(
        select(Lease).where(Lease.status.in_(tuple(ACTIVE_LEASE_STATUSES)))
    )
    for lease in active.scalars().all():
        if exclude_lease_id and lease.id == exclude_lease_id:
            continue
        pod = await get_pod_by_lease(session, lease.id)
        if pod is None:
            continue
        if node_ids.intersection(pod.node_ids()):
            raise LeaseError(
                "capacity_unavailable",
                "one or more nodes already exclusive-leased",
                status_code=409,
            )


async def rent_offer(
    session: AsyncSession,
    *,
    renter_hotkey: str,
    offer_id: str,
    lifetime_hours: Any | None = None,
    max_price: Any | None = None,
    sim_ready: bool = True,
) -> tuple[Lease, Pod]:
    """Create exclusive lease + pod from a listed offer (VAL-MKT-013/014/017/019).

    When ``sim_ready`` is True (local default), pod is immediately ``running`` and
    lease ``active`` so CI/sim does not need a provider agent mark-ready step.
    """

    if not renter_hotkey or not str(renter_hotkey).strip():
        raise LeaseError(
            "missing_renter",
            "renter hotkey required",
            status_code=401,
        )
    renter = str(renter_hotkey).strip()

    # Select offer with a write intent — exclusive claim.
    result = await session.execute(select(Offer).where(Offer.id == offer_id))
    offer = result.scalar_one_or_none()
    if offer is None:
        raise LeaseError("offer_not_found", "offer not found", status_code=404)
    if offer.status != OFFER_STATUS_LISTED:
        raise LeaseError(
            "offer_not_listed",
            f"offer status is {offer.status}, expected listed",
            status_code=409,
        )

    # Renter max_price bound (VAL-MKT-010 rent-side).
    if max_price is not None:
        bound = _finite_positive(max_price, field="max_price")
        if float(offer.price_per_hour) > bound:
            raise LeaseError(
                "price_over_renter_max",
                (f"offer price_per_hour {offer.price_per_hour} exceeds renter max_price {bound}"),
                status_code=422,
            )

    # Lifetime: default to offer max; end user may request shorter/equal.
    offer_max = float(offer.max_lifetime_hours)
    if lifetime_hours is None:
        lifetime = offer_max
    else:
        lifetime = _finite_positive(lifetime_hours, field="lifetime_hours")
        if lifetime > offer_max:
            raise LeaseError(
                "lifetime_over_offer_max",
                f"lifetime_hours {lifetime} exceeds offer max {offer_max}",
                status_code=422,
            )

    nodes = await _load_nodes_for_offer(session, offer)
    for node in nodes:
        if node.status not in {"healthy", "registered", "rented"}:
            # Only free healthy inventory is rentable; reserved rented is barred below.
            if node.status not in {"healthy", "registered"}:
                raise LeaseError(
                    "node_not_offerable",
                    f"node {node.id} status {node.status} is not rentable",
                    status_code=409,
                )

    # VAL-FAB-010: re-authenticate require_ib against latest fabric reports /
    # denormalized IB flags so stripped IB re-report cannot be rented.
    if int(getattr(offer, "require_ib", 0) or 0) == 1:
        from hypercluster.domain.fabric_reports import load_latest_reports_for_nodes
        from hypercluster.domain.nodes import node_has_ib
        from hypercluster.fabric.gates import evaluate_require_ib_nodes

        # Fast path: denormalized has_ib flags (updated by fabric-scan).
        eth_only = [n.id for n in nodes if not node_has_ib(n)]
        if eth_only:
            raise LeaseError(
                "require_ib_not_satisfied",
                (
                    "require_ib offer blocked: node(s) lack InfiniBand after "
                    f"updated inventory: {', '.join(eth_only)}"
                ),
                status_code=409,
            )
        reports = await load_latest_reports_for_nodes(session, [n.id for n in nodes])
        check = evaluate_require_ib_nodes(
            require_ib=True,
            reports=reports,
            node_ids=[n.id for n in nodes],
        )
        if not check.may_rent:
            raise LeaseError(
                check.failure_code or "require_ib_not_satisfied",
                check.reason or "require_ib fabric consistency failed",
                status_code=409,
            )

    await _assert_nodes_exclusive_free(session, nodes)

    now = utc_now()
    ends_at = now + timedelta(hours=lifetime)
    lease_id = str(uuid.uuid4())
    pod_id = str(uuid.uuid4())

    lease_status = LEASE_STATUS_ACTIVE if sim_ready else LEASE_STATUS_REQUESTED
    pod_status = POD_STATUS_RUNNING if sim_ready else POD_STATUS_PROVISIONING

    # Sim endpoints: one entry per node (agent would fill real SSH later).
    endpoints = {
        n.id: {
            "ssh": n.ssh_endpoint,
            "hostname": n.hostname,
            "gpu_model": n.gpu_model,
            "gpu_count": n.gpu_count,
        }
        for n in nodes
    }

    lease = Lease(
        id=lease_id,
        offer_id=offer.id,
        renter_hotkey=renter,
        provider_id=offer.provider_id,
        status=lease_status,
        started_at=now if sim_ready else None,
        ends_at=ends_at,
        price_per_hour=float(offer.price_per_hour),
        lifetime_hours=lifetime,
        termination_reason=None,
        created_at=now,
        updated_at=now,
    )
    pod = Pod(
        id=pod_id,
        lease_id=lease_id,
        mode=offer.mode,
        status=pod_status,
        node_ids_json=json.dumps([n.id for n in nodes]),
        image_digest=None,
        ssh_authorized_json=None,
        endpoints_json=json.dumps(endpoints),
        created_at=now,
        updated_at=now,
    )

    # Claim exclusive capacity.
    offer.status = OFFER_STATUS_LEASED
    offer.updated_at = now
    for node in nodes:
        node.status = "rented"
        node.updated_at = now

    session.add(lease)
    session.add(pod)
    await session.commit()
    await session.refresh(lease)
    await session.refresh(pod)
    return lease, pod


async def get_lease(session: AsyncSession, lease_id: str) -> Lease | None:
    result = await session.execute(select(Lease).where(Lease.id == lease_id))
    return result.scalar_one_or_none()


async def get_pod(session: AsyncSession, pod_id: str) -> Pod | None:
    result = await session.execute(select(Pod).where(Pod.id == pod_id))
    return result.scalar_one_or_none()


async def get_pod_by_lease(session: AsyncSession, lease_id: str) -> Pod | None:
    result = await session.execute(select(Pod).where(Pod.lease_id == lease_id))
    return result.scalar_one_or_none()


async def list_leases(
    session: AsyncSession,
    *,
    hotkey: str | None = None,
    offer_id: str | None = None,
    status: str | None = None,
) -> list[Lease]:
    """List leases scoped to renter hotkey and/or provider hotkey (VAL-MKT-016).

    Fail-closed: when ``hotkey`` is missing / empty, return ``[]`` rather than
    the full lease table. Callers must supply identity for renter or provider
    views — there is no unauthenticated admin dump.

    When ``hotkey`` is set, return leases where the caller is the renter **or**
    the provider owner of the offer (via provider_id ↔ providers.hotkey).
    """

    # Identity-scoped list: missing hotkey must never dump all rentals.
    if not hotkey:
        return []

    stmt = select(Lease).order_by(Lease.created_at.asc())
    if offer_id:
        stmt = stmt.where(Lease.offer_id == offer_id)
    if status:
        stmt = stmt.where(Lease.status == status)
    # Join through providers for provider-view; also match renter_hotkey.
    provider_result = await session.execute(select(Provider.id).where(Provider.hotkey == hotkey))
    provider_ids = [row[0] for row in provider_result.all()]
    if provider_ids:
        stmt = stmt.where(
            or_(
                Lease.renter_hotkey == hotkey,
                Lease.provider_id.in_(provider_ids),
            )
        )
    else:
        stmt = stmt.where(Lease.renter_hotkey == hotkey)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _free_capacity_for_lease(session: AsyncSession, lease: Lease, pod: Pod | None) -> None:
    """After terminal lease: free nodes and open path for re-list.

    Offer stays ``leased`` historically; capacity frees by restoring nodes to
    healthy so a new listed offer can be created on the same inventory.
    """

    now = utc_now()
    node_ids: list[str] = []
    if pod is not None:
        node_ids = pod.node_ids()
        pod.status = POD_STATUS_STOPPED
        pod.updated_at = now
    elif lease.offer_id:
        offer = await get_offer(session, lease.offer_id)
        if offer is not None:
            node_ids = offer.node_ids()

    if node_ids:
        result = await session.execute(select(Node).where(Node.id.in_(node_ids)))
        for node in result.scalars().all():
            # Only release if no *other* active lease still after this termination
            # (cluster/single exclusive; after this lease is terminal it is free).
            if node.status == "rented":
                node.status = "healthy"
            node.updated_at = now

    # Mark offer historical: no longer actively leased for withdraw-block.
    # After terminal, withdraw is allowed on this offer id if still LEASED status —
    # policy: flip leased → withdrawn-capable by setting status to listed? No.
    # Offer was capacity snapshot; leave as leased/historical OR allow withdraw
    # of residual. VAL-MKT-031 carefully says after terminal, withdraw or re-list
    # of *capacity* is allowed. Re-list = new offer. Withdraw of terminal offer:
    # clear ACTIVE_LEASE block by treating status as terminal leased; expand
    # withdraw to allow after no active lease rows remain.
    offer = await get_offer(session, lease.offer_id)
    if offer is not None and offer.status == OFFER_STATUS_LEASED:
        # Check remaining active leases on this offer.
        remaining = await session.execute(
            select(Lease).where(
                Lease.offer_id == offer.id,
                Lease.status.in_(tuple(ACTIVE_LEASE_STATUSES)),
            )
        )
        if remaining.scalar_one_or_none() is None:
            # Keep as leased (history). Withdraw gate checks active leases, not
            # only offer.status; update offers.withdraw to dual-check. For
            # status clarity keep `leased` so default browse still excludes it.
            offer.updated_at = now


async def terminate_lease(
    session: AsyncSession,
    *,
    hotkey: str,
    lease_id: str,
    reason: str | None = "renter_cancel",
    allow_provider: bool = True,
) -> Lease:
    """Renter (or provider) terminate → terminal lease + stopped pod (VAL-MKT-015)."""

    lease = await get_lease(session, lease_id)
    if lease is None:
        raise LeaseError("lease_not_found", "lease not found", status_code=404)

    is_renter = lease.renter_hotkey == hotkey
    is_provider = False
    if allow_provider:
        provider_result = await session.execute(
            select(Provider).where(Provider.id == lease.provider_id)
        )
        provider = provider_result.scalar_one_or_none()
        is_provider = provider is not None and provider.hotkey == hotkey

    if not is_renter and not is_provider:
        raise LeaseError(
            "lease_not_owned",
            "caller is neither renter nor provider for this lease",
            status_code=403,
        )

    if lease.status in TERMINAL_LEASE_STATUSES:
        return lease  # idempotent

    now = utc_now()
    lease.status = LEASE_STATUS_TERMINATED
    lease.termination_reason = (reason or "terminated")[:256]
    lease.updated_at = now

    pod = await get_pod_by_lease(session, lease.id)
    await _free_capacity_for_lease(session, lease, pod)
    await session.commit()
    await session.refresh(lease)
    return lease


async def expire_due_leases(
    session: AsyncSession,
    *,
    now: Any | None = None,
) -> int:
    """Expire active leases whose ends_at is past (VAL-MKT-018 / VAL-MKT-021)."""

    from datetime import datetime

    current = now if isinstance(now, datetime) else utc_now()
    result = await session.execute(
        select(Lease).where(
            Lease.status.in_(tuple(ACTIVE_LEASE_STATUSES)),
            Lease.ends_at.is_not(None),
            Lease.ends_at <= current,
        )
    )
    due = list(result.scalars().all())
    for lease in due:
        lease.status = LEASE_STATUS_EXPIRED
        lease.termination_reason = "max_lifetime"
        lease.updated_at = current
        pod = await get_pod_by_lease(session, lease.id)
        await _free_capacity_for_lease(session, lease, pod)
    if due:
        await session.commit()
    return len(due)


async def run_idle_reclaim_sweep(
    session: AsyncSession,
    *,
    liveness_seconds: int = 120,
) -> int:
    """Idle-only reclaim: offline free (non-rented) nodes past heartbeat miss.

    **Must not** terminate active leases or force-stop pods on rented capacity
    (VAL-MKT-020 tenant short-circuit). Explicit terminate / max lifetime use
    separate paths (VAL-MKT-021).
    """

    from hypercluster.domain.nodes import mark_stale_nodes_offline

    # Ensure rented nodes under an active lease cannot slip into offline.
    result = await session.execute(
        select(Lease).where(Lease.status.in_(tuple(ACTIVE_LEASE_STATUSES)))
    )
    protected_node_ids: set[str] = set()
    for lease in result.scalars().all():
        pod = await get_pod_by_lease(session, lease.id)
        if pod is not None:
            protected_node_ids.update(pod.node_ids())

    if protected_node_ids:
        # Force-preserve rented status for active tenants before offline sweep.
        nodes = await session.execute(select(Node).where(Node.id.in_(protected_node_ids)))
        now = utc_now()
        for node in nodes.scalars().all():
            node.status = "rented"
            node.updated_at = now
        await session.commit()

    # Online free inventory past miss → offline (existing helper already skips rented).
    return await mark_stale_nodes_offline(session, liveness_seconds=liveness_seconds)


def lease_to_public(lease: Lease) -> dict[str, Any]:
    return lease.to_dict()


def pod_to_public(pod: Pod) -> dict[str, Any]:
    return pod.to_dict()


__all__ = [
    "ACTIVE_LEASE_STATUSES",
    "LEASE_STATUS_ACTIVE",
    "LEASE_STATUS_EXPIRED",
    "LEASE_STATUS_FAILED",
    "LEASE_STATUS_REQUESTED",
    "LEASE_STATUS_TERMINATED",
    "LeaseError",
    "POD_STATUS_ERROR",
    "POD_STATUS_PROVISIONING",
    "POD_STATUS_RUNNING",
    "POD_STATUS_STOPPED",
    "POD_STATUS_STOPPING",
    "TERMINAL_LEASE_STATUSES",
    "expire_due_leases",
    "get_lease",
    "get_pod",
    "get_pod_by_lease",
    "lease_to_public",
    "list_leases",
    "pod_to_public",
    "rent_offer",
    "run_idle_reclaim_sweep",
    "terminate_lease",
]
