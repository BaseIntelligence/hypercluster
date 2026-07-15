"""Public challenge routes (proxied by Base master when registered).

Identity routes (`/health`, `/ready`, `/version`) are installed by
`create_challenge_app` and are not registered here. Internal routes under
`/internal/*` are owned by the SDK factory and must never carry `@public_route`.

Marketplace providers/nodes land here (VAL-MKT-001..007). Later features add
offers/leases/pods on the same router.
"""

from __future__ import annotations

from typing import Any

from base.challenge_sdk import public_route
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from hypercluster.api.auth import DbSession, RequireMiner
from hypercluster.domain.nodes import (
    NodeError,
    get_node,
    list_nodes,
    node_heartbeat,
    node_to_public,
    register_node,
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


def _header_hotkey(request: Request) -> str | None:
    return request.headers.get("x-hotkey") or request.headers.get("X-Hotkey")


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
@router.get("/v1/offers")
async def list_offers() -> dict[str, list[object]]:
    """Browse marketplace offers (scaffold; domain logic lands in later M2)."""

    return {"items": []}


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
    "list_jobs",
    "list_offers",
    "nodes_get",
    "nodes_heartbeat",
    "nodes_list",
    "nodes_register",
    "providers_heartbeat",
    "providers_list",
    "providers_me",
    "providers_register",
    "router",
]
