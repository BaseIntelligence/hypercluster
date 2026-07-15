"""Public challenge routes (proxied by Base master when registered).

Identity routes (`/health`, `/ready`, `/version`) are installed by
`create_challenge_app` and are not registered here. Internal routes under
`/internal/*` are owned by the SDK factory and must never carry `@public_route`.

Marketplace providers/nodes/offers/leases/pods (VAL-MKT-001..021, 025..029, 031).
"""

from __future__ import annotations

from typing import Any

from base.challenge_sdk import public_route
from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from hypercluster.api.auth import DbSession, RequireMiner
from hypercluster.domain.leases import (
    LeaseError,
    get_lease,
    get_pod,
    get_pod_by_lease,
    lease_to_public,
    list_leases,
    pod_to_public,
    rent_offer,
    terminate_lease,
)
from hypercluster.domain.nodes import (
    NodeError,
    get_node,
    list_nodes,
    node_heartbeat,
    node_to_public,
    register_node,
)
from hypercluster.domain.offers import (
    DEFAULT_MAX_OFFER_LIFETIME_HOURS,
    DEFAULT_MAX_OFFER_PRICE_PER_HOUR,
    OFFER_STATUS_LISTED,
    OfferError,
    create_offer,
    get_offer,
    list_offers,
    offer_to_public,
    parse_require_ib_query,
    withdraw_offer,
)
from hypercluster.domain.providers import (
    ProviderError,
    get_provider_by_hotkey,
    list_providers,
    provider_heartbeat,
    provider_to_public,
    register_provider,
)

router = APIRouter()


class ProviderRegisterRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=256)


class NodeRegisterRequest(BaseModel):
    gpu_model: str = Field(..., min_length=1, max_length=128)
    gpu_count: int = Field(..., ge=1)
    hostname: str | None = Field(default=None, max_length=256)
    ssh_endpoint: str | None = Field(default=None, max_length=512)
    cpu_cores: int | None = Field(default=None, ge=1)
    mem_gb: float | None = Field(default=None, ge=0)
    location_hint: str | None = Field(default=None, max_length=128)
    tee_capability: str = Field(default="none", max_length=32)
    inventory: dict[str, Any] | None = None
    node_id: str | None = Field(default=None, max_length=36)


class NodeHeartbeatRequest(BaseModel):
    node_id: str | None = Field(default=None, max_length=36)


class OfferCreateRequest(BaseModel):
    """Offer publish body; price/lifetime hard guards also enforced in domain."""

    node_ids: list[str] = Field(..., min_length=1)
    # Optional on the wire so missing keys surface as domain 422 codes (not body-schema),
    # matching VAL-MKT-009 matrix of "missing price/lifetime".
    price_per_hour: float | None = Field(default=None)
    max_lifetime_hours: float | None = Field(default=None)
    mode: str = Field(default="single", max_length=16)
    require_ib: bool = False
    tee: str | None = Field(default=None, max_length=32)
    gpu_model: str | None = Field(default=None, max_length=128)
    gpu_count: int | None = Field(default=None, ge=1)
    location_hint: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None


class RentRequest(BaseModel):
    """Renter rent body; lifetime ≤ offer max; optional max_price renter bound."""

    lifetime_hours: float | None = Field(default=None, gt=0)
    max_price: float | None = Field(default=None, gt=0)


class TerminateLeaseRequest(BaseModel):
    reason: str | None = Field(default="renter_cancel", max_length=256)


def _header_hotkey(request: Request) -> str | None:
    return request.headers.get("x-hotkey") or request.headers.get("X-Hotkey")


def _offer_caps(request: Request) -> tuple[float, float]:
    """Read system offer caps from HyperSettings (env-tunable)."""

    hyper = getattr(request.app.state, "hyper_settings", None)
    price_cap = DEFAULT_MAX_OFFER_PRICE_PER_HOUR
    lifetime_cap = DEFAULT_MAX_OFFER_LIFETIME_HOURS
    if hyper is not None:
        price_cap = float(
            getattr(hyper, "max_offer_price_per_hour", price_cap) or price_cap
        )
        lifetime_cap = float(
            getattr(hyper, "max_offer_lifetime_hours", lifetime_cap) or lifetime_cap
        )
    return price_cap, lifetime_cap


@public_route(tags=["marketplace"])
@router.post("/v1/providers/register", status_code=status.HTTP_200_OK)
async def providers_register(
    body: ProviderRegisterRequest,
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Provider hotkey onboarding (VAL-MKT-001). Idempotent per hotkey."""

    try:
        provider, created = await register_provider(
            session,
            hotkey=identity.hotkey,
            display_name=body.display_name,
        )
    except ProviderError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    payload = provider_to_public(provider)
    payload["created"] = created
    return payload


@public_route(tags=["marketplace"])
@router.get("/v1/providers")
async def providers_list(
    session: DbSession,
    request: Request,
) -> dict[str, Any]:
    """List providers. Optional X-Hotkey scopes to owner (VAL-MKT-002)."""

    providers = await list_providers(session, hotkey=_header_hotkey(request))
    return {"items": [provider_to_public(p) for p in providers]}


@public_route(tags=["marketplace"])
@router.get("/v1/providers/me")
async def providers_me(
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Return the caller's own provider (requires signature)."""

    provider = await get_provider_by_hotkey(session, identity.hotkey)
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "provider_not_found", "message": "not registered"},
        )
    return provider_to_public(provider)


@public_route(tags=["marketplace"])
@router.post("/v1/providers/heartbeat")
async def providers_heartbeat(
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Advance provider liveness without mutating identity (VAL-MKT-003)."""

    try:
        provider = await provider_heartbeat(session, hotkey=identity.hotkey)
    except ProviderError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return provider_to_public(provider)


@public_route(tags=["marketplace"])
@router.post("/v1/nodes", status_code=status.HTTP_200_OK)
async def nodes_register(
    body: NodeRegisterRequest,
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Register / update node inventory with GPU + IB fields (VAL-MKT-004/005)."""

    try:
        node, created = await register_node(
            session,
            hotkey=identity.hotkey,
            gpu_model=body.gpu_model,
            gpu_count=body.gpu_count,
            hostname=body.hostname,
            ssh_endpoint=body.ssh_endpoint,
            cpu_cores=body.cpu_cores,
            mem_gb=body.mem_gb,
            location_hint=body.location_hint,
            tee_capability=body.tee_capability,
            inventory=body.inventory,
            node_id=body.node_id,
        )
    except NodeError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    payload = node_to_public(node)
    payload["created"] = created
    return payload


@public_route(tags=["marketplace"])
@router.get("/v1/nodes")
async def nodes_list(
    session: DbSession,
    request: Request,
) -> dict[str, Any]:
    """List nodes with capability fields (VAL-MKT-007). X-Hotkey scopes to owner."""

    nodes = await list_nodes(session, hotkey=_header_hotkey(request))
    return {"items": [node_to_public(n) for n in nodes]}


@public_route(tags=["marketplace"])
@router.post("/v1/nodes/heartbeat")
async def nodes_heartbeat(
    identity: RequireMiner,
    session: DbSession,
    request: Request,
    body: NodeHeartbeatRequest | None = None,
) -> dict[str, Any]:
    """Refresh last_heartbeat for owned node(s) (VAL-MKT-006)."""

    req = body if body is not None else NodeHeartbeatRequest()
    hyper = getattr(request.app.state, "hyper_settings", None)
    liveness = 120
    if hyper is not None:
        liveness = int(getattr(hyper, "node_liveness_seconds", 120))
    try:
        nodes = await node_heartbeat(
            session,
            hotkey=identity.hotkey,
            node_id=req.node_id,
            liveness_seconds=liveness,
        )
    except NodeError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return {"items": [node_to_public(n) for n in nodes]}


@public_route(tags=["marketplace"])
@router.get("/v1/nodes/{node_id}")
async def nodes_get(
    node_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Get a single node with capability fields (VAL-MKT-007)."""

    node = await get_node(session, node_id)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "node_not_found", "message": "node not found"},
        )
    return node_to_public(node)


@public_route(tags=["marketplace"])
@router.post("/v1/offers", status_code=status.HTTP_200_OK)
async def offers_create(
    body: OfferCreateRequest,
    identity: RequireMiner,
    session: DbSession,
    request: Request,
) -> dict[str, Any]:
    """Publish capacity offer with hard price/lifetime guards (VAL-MKT-008..011)."""

    price_cap, lifetime_cap = _offer_caps(request)
    try:
        offer = await create_offer(
            session,
            hotkey=identity.hotkey,
            node_ids=body.node_ids,
            price_per_hour=body.price_per_hour,
            max_lifetime_hours=body.max_lifetime_hours,
            mode=body.mode,
            require_ib=body.require_ib,
            tee=body.tee,
            gpu_model=body.gpu_model,
            gpu_count=body.gpu_count,
            location_hint=body.location_hint,
            metadata=body.metadata,
            max_price_cap=price_cap,
            max_lifetime_cap=lifetime_cap,
        )
    except OfferError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return offer_to_public(offer)


@public_route(tags=["marketplace"])
@router.get("/v1/offers")
async def offers_list(
    session: DbSession,
    gpu_model: str | None = Query(default=None),
    require_ib: str | None = Query(default=None),
    tee: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    mode: str | None = Query(default=None),
) -> dict[str, Any]:
    """Browse marketplace offers with composable filters (VAL-MKT-025..029).

    Default status is ``listed`` (rentable catalog). Pass ``status`` to override
    (e.g. ``withdrawn``). Capability filters compose AND with status.
    """

    # Default browse: listed only so withdrawn/leased never reappear as rentable.
    status_value: str | None
    if status_filter is None:
        status_value = OFFER_STATUS_LISTED
    elif status_filter.strip().lower() in {"", "all", "*"}:
        status_value = None
    else:
        status_value = status_filter.strip().lower()

    try:
        require_ib_flag = parse_require_ib_query(require_ib)
        items = await list_offers(
            session,
            status=status_value,
            gpu_model=gpu_model,
            require_ib=require_ib_flag,
            tee=tee,
            mode=mode,
        )
    except OfferError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return {"items": [offer_to_public(o) for o in items]}


@public_route(tags=["marketplace"])
@router.get("/v1/offers/{offer_id}")
async def offers_get(
    offer_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Get a single offer by id (any status)."""

    offer = await get_offer(session, offer_id)
    if offer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "offer_not_found", "message": "offer not found"},
        )
    return offer_to_public(offer)


@public_route(tags=["marketplace"])
@router.delete("/v1/offers/{offer_id}")
async def offers_withdraw(
    offer_id: str,
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Withdraw listing (VAL-MKT-012); owner-only; fail-closed under active lease."""

    try:
        offer = await withdraw_offer(
            session,
            hotkey=identity.hotkey,
            offer_id=offer_id,
        )
    except OfferError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return offer_to_public(offer)


@public_route(tags=["marketplace"])
@router.post("/v1/offers/{offer_id}/rent", status_code=status.HTTP_200_OK)
async def offers_rent(
    offer_id: str,
    identity: RequireMiner,
    session: DbSession,
    body: RentRequest | None = None,
) -> dict[str, Any]:
    """Rent listed offer → exclusive lease + pod (VAL-MKT-013/014/017/019)."""

    req = body if body is not None else RentRequest()
    try:
        lease, pod = await rent_offer(
            session,
            renter_hotkey=identity.hotkey,
            offer_id=offer_id,
            lifetime_hours=req.lifetime_hours,
            max_price=req.max_price,
            sim_ready=True,
        )
    except LeaseError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return {
        "lease": lease_to_public(lease),
        "pod": pod_to_public(pod),
    }


@public_route(tags=["marketplace"])
@router.get("/v1/leases")
async def leases_list(
    session: DbSession,
    request: Request,
    offer_id: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
) -> dict[str, Any]:
    """List leases for renter and/or provider hotkey (VAL-MKT-016).

    Scoped by optional X-Hotkey (no signature required for list view policy).
    Without X-Hotkey returns empty items (fail-closed identity scope).
    """

    hotkey = _header_hotkey(request)
    items = await list_leases(
        session,
        hotkey=hotkey,
        offer_id=offer_id,
        status=status_filter,
    )
    return {"items": [lease_to_public(x) for x in items]}


@public_route(tags=["marketplace"])
@router.get("/v1/leases/{lease_id}")
async def leases_get(
    lease_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Lease detail (status, offer_id, price, times) — VAL-MKT-016."""

    lease = await get_lease(session, lease_id)
    if lease is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "lease_not_found", "message": "lease not found"},
        )
    return lease_to_public(lease)


@public_route(tags=["marketplace"])
@router.post("/v1/leases/{lease_id}/terminate", status_code=status.HTTP_200_OK)
async def leases_terminate(
    lease_id: str,
    identity: RequireMiner,
    session: DbSession,
    body: TerminateLeaseRequest | None = None,
) -> dict[str, Any]:
    """Renter/provider terminate lease; pod stops; capacity free (VAL-MKT-015/021)."""

    req = body if body is not None else TerminateLeaseRequest()
    try:
        lease = await terminate_lease(
            session,
            hotkey=identity.hotkey,
            lease_id=lease_id,
            reason=req.reason,
            allow_provider=True,
        )
        pod = await get_pod_by_lease(session, lease.id)
    except LeaseError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    payload: dict[str, Any] = {"lease": lease_to_public(lease)}
    if pod is not None:
        payload["pod"] = pod_to_public(pod)
    return payload


@public_route(tags=["marketplace"])
@router.get("/v1/pods/{pod_id}")
async def pods_get(
    pod_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Pod detail with node binding and endpoints (VAL-MKT-017/019)."""

    pod = await get_pod(session, pod_id)
    if pod is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "pod_not_found", "message": "pod not found"},
        )
    return pod_to_public(pod)


@public_route(tags=["jobs"])
@router.get("/v1/jobs")
async def list_jobs() -> dict[str, list[object]]:
    """List submitter jobs (scaffold; domain logic lands in M3)."""

    return {"items": []}


@public_route(tags=["scoring"])
@router.get("/v1/leaderboard")
async def leaderboard() -> dict[str, list[object]]:
    """Aggregated composite scores (scaffold; domain logic lands in M6)."""

    return {"items": []}


__all__ = [
    "leaderboard",
    "leases_get",
    "leases_list",
    "leases_terminate",
    "list_jobs",
    "nodes_get",
    "nodes_heartbeat",
    "nodes_list",
    "nodes_register",
    "offers_create",
    "offers_get",
    "offers_list",
    "offers_rent",
    "offers_withdraw",
    "pods_get",
    "providers_heartbeat",
    "providers_list",
    "providers_me",
    "providers_register",
    "router",
]
