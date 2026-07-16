"""Public + admin GPU price catalog HTTP routes (M11; VAL-PRICE-030..033).

Public (no auth):
  GET /v1/gpu-prices                 active-only list (optional family/model_key)
  GET /v1/gpu-prices/{model_key}     active detail; inactive/missing → 404

Admin (challenge shared token — Bearer or X-Challenge-Token):
  GET  /v1/admin/gpu-prices
  GET  /v1/admin/gpu-prices/{model_key}
  PUT  /v1/admin/gpu-prices/{model_key}
  POST /v1/admin/gpu-prices
  POST /v1/admin/gpu-prices/{model_key}/disable
  GET  /v1/admin/gpu-prices/{model_key}/history

Public bodies never include operator ``notes``. Admin may include notes +
multipliers. Unauthorized admin → 401 ``price_catalog_unauthorized``.
"""

from __future__ import annotations

from typing import Any

from base.challenge_sdk import public_route
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from hypercluster.api.auth import DbSession, RequireSharedToken
from hypercluster.domain.pricing import (
    PricingError,
    catalog_row_to_public,
    disable_catalog_price,
    get_catalog_price,
    history_row_to_public,
    list_catalog_prices,
    list_price_history,
    upsert_catalog_price,
)

router = APIRouter()


class AdminPriceUpsertBody(BaseModel):
    """Admin create/upsert body (VAL-PRICE-033)."""

    price_per_hour: float | None = Field(default=None)
    model_key: str | None = Field(default=None, max_length=128)
    family: str | None = Field(default=None, max_length=64)
    display_name: str | None = Field(default=None, max_length=256)
    currency: str | None = Field(default=None, max_length=8)
    active: bool | int | None = Field(default=True)
    notes: str | None = Field(default=None, max_length=4000)
    max_offer_multiplier: float | None = Field(default=None)
    min_offer_multiplier: float | None = Field(default=None)
    reason: str | None = Field(default=None, max_length=512)
    source: str | None = Field(default=None, max_length=32)


class AdminPriceDisableBody(BaseModel):
    reason: str | None = Field(default=None, max_length=512)


def _pricing_http(exc: PricingError) -> HTTPException:
    return HTTPException(
        status_code=int(exc.status_code or 400),
        detail={"code": exc.code, "message": exc.message},
    )


def _public_list_payload(rows: list[Any]) -> dict[str, Any]:
    items = [catalog_row_to_public(row, include_admin=False) for row in rows]
    return {
        "items": items,
        "count": len(items),
        "empty": len(items) == 0,
    }


def _admin_list_payload(rows: list[Any]) -> dict[str, Any]:
    items = [catalog_row_to_public(row, include_admin=True) for row in rows]
    return {
        "items": items,
        "count": len(items),
        "empty": len(items) == 0,
    }


# ---------------------------------------------------------------------------
# Public (active-only)
# ---------------------------------------------------------------------------


@public_route(tags=["prices"])
@router.get("/v1/gpu-prices")
async def public_list_gpu_prices(
    session: DbSession,
    family: str | None = Query(default=None),
    model_key: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict[str, Any]:
    """List active USD catalog rows for miners (VAL-PRICE-030).

    Empty catalog → 200 ``items=[]``. Never dumps operator notes. No auth.
    """

    rows = await list_catalog_prices(
        session,
        active_only=True,
        family=family,
        model_key=model_key,
        limit=limit,
    )
    return _public_list_payload(rows)


@public_route(tags=["prices"])
@router.get("/v1/gpu-prices/{model_key}")
async def public_get_gpu_price(
    model_key: str,
    session: DbSession,
) -> dict[str, Any]:
    """Active catalog detail; inactive/missing → 404 (VAL-PRICE-030)."""

    row = await get_catalog_price(session, model_key, active_only=True)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "catalog_not_found",
                "message": f"active catalog entry {model_key!r} not found",
            },
        )
    return catalog_row_to_public(row, include_admin=False)


# ---------------------------------------------------------------------------
# Admin (shared token)
# ---------------------------------------------------------------------------


@public_route(tags=["prices-admin"])
@router.get("/v1/admin/gpu-prices")
async def admin_list_gpu_prices(
    session: DbSession,
    _actor: RequireSharedToken,
    family: str | None = Query(default=None),
    model_key: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
    active_only: bool = Query(default=False),
) -> dict[str, Any]:
    """Full catalog list including inactive (VAL-PRICE-031).

    Default ``active_only=false`` so ops can inventory disabled keys.
    """

    rows = await list_catalog_prices(
        session,
        active_only=bool(active_only),
        family=family,
        model_key=model_key,
        limit=limit,
    )
    return _admin_list_payload(rows)


@public_route(tags=["prices-admin"])
@router.get("/v1/admin/gpu-prices/{model_key}/history")
async def admin_gpu_price_history(
    model_key: str,
    session: DbSession,
    _actor: RequireSharedToken,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    """Ordered history for a model_key, newest first (VAL-PRICE-033)."""

    rows = await list_price_history(session, model_key, limit=limit)
    items = [history_row_to_public(row) for row in rows]
    return {
        "model_key": model_key,
        "items": items,
        "count": len(items),
        "empty": len(items) == 0,
    }


@public_route(tags=["prices-admin"])
@router.get("/v1/admin/gpu-prices/{model_key}")
async def admin_get_gpu_price(
    model_key: str,
    session: DbSession,
    _actor: RequireSharedToken,
    history_limit: int = Query(default=10, ge=0, le=100),
) -> dict[str, Any]:
    """Admin detail for active or inactive row + history tail (VAL-PRICE-031)."""

    row = await get_catalog_price(session, model_key, active_only=False)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "catalog_not_found",
                "message": f"catalog entry {model_key!r} not found",
            },
        )
    history: list[dict[str, Any]] = []
    if history_limit > 0:
        hist_rows = await list_price_history(session, model_key, limit=history_limit)
        history = [history_row_to_public(h) for h in hist_rows]
    return {
        "catalog": catalog_row_to_public(row, include_admin=True),
        "history": history,
    }


@public_route(tags=["prices-admin"])
@router.put("/v1/admin/gpu-prices/{model_key}")
async def admin_put_gpu_price(
    model_key: str,
    body: AdminPriceUpsertBody,
    session: DbSession,
    actor: RequireSharedToken,
) -> dict[str, Any]:
    """Upsert catalog price; always append one history row (VAL-PRICE-033)."""

    if body.price_per_hour is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "invalid_price",
                "message": "price_per_hour must be a positive finite number",
            },
        )
    try:
        row = await upsert_catalog_price(
            session,
            model_key=model_key,
            price_per_hour=body.price_per_hour,
            family=body.family,
            display_name=body.display_name,
            currency=body.currency,
            active=body.active,
            source=body.source or "admin",
            notes=body.notes,
            max_offer_multiplier=body.max_offer_multiplier,
            min_offer_multiplier=body.min_offer_multiplier,
            changed_by=actor,
            reason=body.reason,
        )
        await session.commit()
    except PricingError as exc:
        await session.rollback()
        raise _pricing_http(exc) from exc
    return {
        "catalog": catalog_row_to_public(row, include_admin=True),
        "ok": True,
    }


@public_route(tags=["prices-admin"])
@router.post("/v1/admin/gpu-prices", status_code=status.HTTP_200_OK)
async def admin_post_gpu_price(
    body: AdminPriceUpsertBody,
    session: DbSession,
    actor: RequireSharedToken,
) -> dict[str, Any]:
    """Create/upsert via POST when model_key is in the body (admin convenience)."""

    key = (body.model_key or "").strip()
    if not key:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "model_key_required",
                "message": "model_key is required",
            },
        )
    if body.price_per_hour is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "invalid_price",
                "message": "price_per_hour must be a positive finite number",
            },
        )
    try:
        row = await upsert_catalog_price(
            session,
            model_key=key,
            price_per_hour=body.price_per_hour,
            family=body.family,
            display_name=body.display_name,
            currency=body.currency,
            active=body.active if body.active is not None else True,
            source=body.source or "admin",
            notes=body.notes,
            max_offer_multiplier=body.max_offer_multiplier,
            min_offer_multiplier=body.min_offer_multiplier,
            changed_by=actor,
            reason=body.reason,
        )
        await session.commit()
    except PricingError as exc:
        await session.rollback()
        raise _pricing_http(exc) from exc
    return {
        "catalog": catalog_row_to_public(row, include_admin=True),
        "ok": True,
    }


@public_route(tags=["prices-admin"])
@router.post("/v1/admin/gpu-prices/{model_key}/disable", status_code=status.HTTP_200_OK)
async def admin_disable_gpu_price(
    model_key: str,
    session: DbSession,
    actor: RequireSharedToken,
    body: AdminPriceDisableBody | None = None,
) -> dict[str, Any]:
    """Disable catalog entry (active=0) without delete; appends history."""

    reason = body.reason if body is not None else None
    try:
        row = await disable_catalog_price(
            session,
            model_key=model_key,
            changed_by=actor,
            reason=reason,
            source="admin",
        )
        await session.commit()
    except PricingError as exc:
        await session.rollback()
        raise _pricing_http(exc) from exc
    return {
        "catalog": catalog_row_to_public(row, include_admin=True),
        "ok": True,
    }


__all__ = [
    "admin_disable_gpu_price",
    "admin_get_gpu_price",
    "admin_gpu_price_history",
    "admin_list_gpu_prices",
    "admin_post_gpu_price",
    "admin_put_gpu_price",
    "public_get_gpu_price",
    "public_list_gpu_prices",
    "router",
]
