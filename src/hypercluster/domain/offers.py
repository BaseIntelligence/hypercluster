"""Offer create / withdraw / browse domain service (price + lifetime hard guards)."""

from __future__ import annotations

import json
import math
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import Node, Offer, utc_now
from hypercluster.domain.nodes import node_has_ib
from hypercluster.domain.providers import get_provider_by_hotkey

# Default hard caps (override via HyperSettings).
DEFAULT_MAX_OFFER_PRICE_PER_HOUR = 1000.0
DEFAULT_MAX_OFFER_LIFETIME_HOURS = 720.0  # 30 days

OFFER_STATUS_LISTED = "listed"
OFFER_STATUS_WITHDRAWN = "withdrawn"
OFFER_STATUS_LEASED = "leased"
OFFER_STATUS_EXPIRED = "expired"

VALID_MODES = frozenset({"single", "cluster"})
# active rental / non-terminal for later withdraw-while-leased guard (VAL-MKT-031).
ACTIVE_LEASE_BLOCK_STATUSES = frozenset({OFFER_STATUS_LEASED})


class OfferError(Exception):
    """Domain error for offer operations."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _finite_positive(value: Any, *, field: str) -> float:
    """Parse a required positive finite number; raise OfferError on failure."""

    if value is None:
        raise OfferError(
            f"missing_{field}",
            f"{field} is required and must be > 0",
            status_code=422,
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OfferError(
            f"invalid_{field}",
            f"{field} must be a positive finite number",
            status_code=422,
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise OfferError(
            f"invalid_{field}",
            f"{field} must be a positive finite number",
            status_code=422,
        )
    return number


def validate_price(
    price: Any,
    *,
    max_price: float | None = DEFAULT_MAX_OFFER_PRICE_PER_HOUR,
) -> float:
    """Hard guard: price_per_hour must be > 0 and ≤ configured system max."""

    number = _finite_positive(price, field="price_per_hour")
    if max_price is not None and max_price > 0 and number > max_price:
        raise OfferError(
            "price_over_cap",
            f"price_per_hour {number} exceeds system max {max_price}",
            status_code=422,
        )
    return number


def validate_lifetime(
    lifetime: Any,
    *,
    max_lifetime: float | None = DEFAULT_MAX_OFFER_LIFETIME_HOURS,
) -> float:
    """Hard guard: max_lifetime_hours must be > 0 and ≤ configured system max."""

    number = _finite_positive(lifetime, field="max_lifetime_hours")
    if max_lifetime is not None and max_lifetime > 0 and number > max_lifetime:
        raise OfferError(
            "lifetime_over_cap",
            f"max_lifetime_hours {number} exceeds system max {max_lifetime}",
            status_code=422,
        )
    return number


def _normalize_tee(value: str | None) -> str:
    tee = (value or "none").strip() or "none"
    return tee


async def _load_owned_nodes(
    session: AsyncSession,
    *,
    provider_id: str,
    node_ids: list[str],
) -> list[Node]:
    if not node_ids:
        raise OfferError(
            "missing_node_ids",
            "node_ids must be a non-empty list",
            status_code=422,
        )
    # Preserve order, drop duplicates while validating each id once.
    seen: set[str] = set()
    ordered: list[str] = []
    for nid in node_ids:
        sid = str(nid).strip()
        if not sid:
            raise OfferError("invalid_node_ids", "node_ids must be non-empty ids", status_code=422)
        if sid in seen:
            continue
        seen.add(sid)
        ordered.append(sid)

    result = await session.execute(select(Node).where(Node.id.in_(ordered)))
    by_id = {n.id: n for n in result.scalars().all()}
    nodes: list[Node] = []
    for nid in ordered:
        node = by_id.get(nid)
        if node is None:
            raise OfferError("node_not_found", f"node {nid} not found", status_code=404)
        if node.provider_id != provider_id:
            raise OfferError(
                "node_not_owned",
                f"node {nid} belongs to another provider",
                status_code=403,
            )
        if node.status not in {"healthy", "registered", "rented"}:
            # Offer listing only against live inventory.
            raise OfferError(
                "node_not_offerable",
                f"node {nid} status {node.status} is not offerable",
                status_code=409,
            )
        nodes.append(node)
    return nodes


async def create_offer(
    session: AsyncSession,
    *,
    hotkey: str,
    node_ids: list[str],
    price_per_hour: Any,
    max_lifetime_hours: Any,
    mode: str = "single",
    require_ib: bool = False,
    tee: str | None = None,
    gpu_model: str | None = None,
    gpu_count: int | None = None,
    location_hint: str | None = None,
    metadata: dict[str, Any] | None = None,
    max_price_cap: float | None = DEFAULT_MAX_OFFER_PRICE_PER_HOUR,
    max_lifetime_cap: float | None = DEFAULT_MAX_OFFER_LIFETIME_HOURS,
) -> Offer:
    """Create a listed offer with hard price/lifetime guards (VAL-MKT-008..011)."""

    provider = await get_provider_by_hotkey(session, hotkey)
    if provider is None:
        raise OfferError(
            "provider_not_found",
            "register provider before creating offers",
            status_code=404,
        )
    if provider.status in {"suspended", "banned"}:
        raise OfferError(
            "provider_not_active",
            f"provider status is {provider.status}",
            status_code=403,
        )

    mode_norm = (mode or "single").strip().lower()
    if mode_norm not in VALID_MODES:
        raise OfferError(
            "invalid_mode",
            "mode must be 'single' or 'cluster'",
            status_code=422,
        )

    price = validate_price(price_per_hour, max_price=max_price_cap)
    lifetime = validate_lifetime(max_lifetime_hours, max_lifetime=max_lifetime_cap)

    nodes = await _load_owned_nodes(session, provider_id=provider.id, node_ids=list(node_ids))

    if mode_norm == "single" and len(nodes) != 1:
        raise OfferError(
            "invalid_mode_nodes",
            "single mode requires exactly one node_id",
            status_code=422,
        )
    if mode_norm == "cluster" and len(nodes) < 2:
        raise OfferError(
            "invalid_mode_nodes",
            "cluster mode requires at least two node_ids",
            status_code=422,
        )

    # require_ib: all referenced nodes must declare InfiniBand (VAL-MKT-005 link).
    if require_ib:
        for node in nodes:
            if not node_has_ib(node):
                raise OfferError(
                    "ib_required",
                    f"node {node.id} lacks InfiniBand inventory for require_ib offer",
                    status_code=422,
                )

    # Derive advertised GPU from nodes when not explicitly overridden.
    models = {n.gpu_model for n in nodes}
    if gpu_model is not None and str(gpu_model).strip():
        model_out = str(gpu_model).strip()
    elif len(models) == 1:
        model_out = next(iter(models))
    else:
        # mixed cluster: caller should supply a label; otherwise use first.
        model_out = nodes[0].gpu_model

    total_gpus = sum(int(n.gpu_count) for n in nodes)
    count_out = int(gpu_count) if gpu_count is not None else total_gpus
    if count_out < 1:
        raise OfferError(
            "invalid_gpu_count",
            "gpu_count must be a positive integer",
            status_code=422,
        )

    tee_out = _normalize_tee(tee if tee is not None else nodes[0].tee_capability)
    now = utc_now()
    offer = Offer(
        id=str(uuid.uuid4()),
        provider_id=provider.id,
        node_ids_json=json.dumps([n.id for n in nodes]),
        mode=mode_norm,
        gpu_model=model_out,
        gpu_count=count_out,
        node_count=len(nodes),
        require_ib=1 if require_ib else 0,
        tee=tee_out,
        price_per_hour=price,
        max_lifetime_hours=lifetime,
        location_hint=location_hint or nodes[0].location_hint,
        status=OFFER_STATUS_LISTED,
        metadata_json=json.dumps(metadata) if metadata is not None else None,
        created_at=now,
        updated_at=now,
    )
    session.add(offer)
    await session.commit()
    await session.refresh(offer)
    return offer


async def withdraw_offer(
    session: AsyncSession,
    *,
    hotkey: str,
    offer_id: str,
) -> Offer:
    """Provider withdraw: listed → withdrawn (VAL-MKT-012).

    Active leased offers are fail-closed (VAL-MKT-031 for later rent slice;
    guard lives here so withdraw never severs an active rental row).
    """

    provider = await get_provider_by_hotkey(session, hotkey)
    if provider is None:
        raise OfferError(
            "provider_not_found",
            "provider not registered for hotkey",
            status_code=404,
        )

    result = await session.execute(select(Offer).where(Offer.id == offer_id))
    offer = result.scalar_one_or_none()
    if offer is None:
        raise OfferError("offer_not_found", "offer not found", status_code=404)
    if offer.provider_id != provider.id:
        raise OfferError(
            "offer_not_owned",
            "offer belongs to another provider",
            status_code=403,
        )
    if offer.status in ACTIVE_LEASE_BLOCK_STATUSES:
        raise OfferError(
            "offer_active_lease",
            "cannot withdraw an offer with an active lease",
            status_code=409,
        )
    if offer.status == OFFER_STATUS_WITHDRAWN:
        return offer  # idempotent

    offer.status = OFFER_STATUS_WITHDRAWN
    offer.updated_at = utc_now()
    await session.commit()
    await session.refresh(offer)
    return offer


async def get_offer(session: AsyncSession, offer_id: str) -> Offer | None:
    result = await session.execute(select(Offer).where(Offer.id == offer_id))
    return result.scalar_one_or_none()


async def list_offers(
    session: AsyncSession,
    *,
    status: str | None = OFFER_STATUS_LISTED,
    gpu_model: str | None = None,
    require_ib: bool | None = None,
    tee: str | None = None,
    provider_id: str | None = None,
    mode: str | None = None,
) -> list[Offer]:
    """Browse offers with composable filters (VAL-MKT-025..029).

    Default ``status='listed'`` keeps the rentable catalog free of withdrawn/
    leased/expired rows. Pass ``status=None`` to disable status filtering.
    """

    stmt = select(Offer).order_by(Offer.created_at.asc())
    if status is not None:
        stmt = stmt.where(Offer.status == status)
    if gpu_model is not None and gpu_model.strip():
        stmt = stmt.where(Offer.gpu_model == gpu_model.strip())
    if require_ib is True:
        stmt = stmt.where(Offer.require_ib == 1)
    elif require_ib is False:
        stmt = stmt.where(Offer.require_ib == 0)
    if tee is not None and tee.strip():
        stmt = stmt.where(Offer.tee == tee.strip())
    if provider_id is not None:
        stmt = stmt.where(Offer.provider_id == provider_id)
    if mode is not None and mode.strip():
        stmt = stmt.where(Offer.mode == mode.strip().lower())
    result = await session.execute(stmt)
    return list(result.scalars().all())


def offer_to_public(offer: Offer) -> dict[str, Any]:
    return offer.to_dict()


def parse_require_ib_query(value: str | bool | None) -> bool | None:
    """Parse require_ib query param (true/false/1/0). None = no filter."""

    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n"}:
        return False
    raise OfferError(
        "invalid_require_ib",
        "require_ib must be true or false",
        status_code=422,
    )


__all__ = [
    "ACTIVE_LEASE_BLOCK_STATUSES",
    "DEFAULT_MAX_OFFER_LIFETIME_HOURS",
    "DEFAULT_MAX_OFFER_PRICE_PER_HOUR",
    "OFFER_STATUS_EXPIRED",
    "OFFER_STATUS_LEASED",
    "OFFER_STATUS_LISTED",
    "OFFER_STATUS_WITHDRAWN",
    "OfferError",
    "VALID_MODES",
    "create_offer",
    "get_offer",
    "list_offers",
    "offer_to_public",
    "parse_require_ib_query",
    "validate_lifetime",
    "validate_price",
    "withdraw_offer",
]
