"""Public challenge routes (proxied by Base master when registered).

Identity routes (`/health`, `/ready`, `/version`) are installed by
`create_challenge_app` and are not registered here. Internal routes under
`/internal/*` are owned by the SDK factory and must never carry `@public_route`.
"""

from __future__ import annotations

from base.challenge_sdk import public_route
from fastapi import APIRouter

router = APIRouter()


@public_route(tags=["marketplace"])
@router.get("/v1/offers")
async def list_offers() -> dict[str, list[object]]:
    """Browse marketplace offers (scaffold; domain logic lands in M2)."""

    return {"items": []}


@public_route(tags=["marketplace"])
@router.get("/v1/nodes")
async def list_nodes() -> dict[str, list[object]]:
    """List capacity nodes (scaffold; domain logic lands in M2)."""

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
    "list_nodes",
    "list_offers",
    "router",
]
