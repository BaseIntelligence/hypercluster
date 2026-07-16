"""GPU price catalog domain service (M11; VAL-PRICE-010..013, 020..021).

SQL-backed reference catalog for USD GPU ``price_per_hour`` values:

- ``upsert_catalog_price`` — create/update one physical row per model_key,
  always append history
- ``disable_catalog_price`` — set active=0 (no hard delete) + history
- ``list_catalog_prices`` / ``get_catalog_price`` / ``list_price_history``
- ``resolve_catalog_price`` — public resolve: exact active model_key prefer,
  else active family via ``normalize_gpu_model``, ordered by
  ``effective_from DESC, model_key ASC``
- ``seed_default_catalog`` — bootstrap ≥10 common USD model_keys (source=seed);
  ``only_if_empty=True`` never clobbers admin rows (VAL-PRICE-020/021)
- ``maybe_seed_prices_on_boot`` — optional ``HYPER_PRICE_SEED_ON_BOOT`` path

Validation (writes):
- finite ``price_per_hour`` > 0 only
- currency USD only (omit → USD)
- inactive rows excluded from public resolve / active-only lists

Never mutates four-factor scoring; never product Verda; never set_weights.
Pricing ladders live only in the seed path — not in FakeSsh / probe tables.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import GpuPriceCatalog, GpuPriceHistory, utc_now
from hypercluster.probe.model_table import normalize_gpu_model

if TYPE_CHECKING:
    from hypercluster.db.database import Database

ALLOWED_CURRENCY = "USD"

# Design ladder: rough public mid-market USD per GPU-hour (operators reprice).
# VAL-PRICE-020 requires ≥10 distinct model_keys spanning H100/A100/RTX/….
# Source of truth: research/m11-gpu-price-catalog-design.md §9.
DEFAULT_SEED_LADDER: tuple[dict[str, Any], ...] = (
    {
        "model_key": "H100_80GB",
        "family": "h100",
        "display_name": "NVIDIA H100 80GB",
        "price_per_hour": 2.49,
    },
    {
        "model_key": "H200_141GB",
        "family": "h200",
        "display_name": "NVIDIA H200",
        "price_per_hour": 3.49,
    },
    {
        "model_key": "A100_80GB",
        "family": "a100",
        "display_name": "NVIDIA A100 80GB",
        "price_per_hour": 1.89,
    },
    {
        "model_key": "A100_40GB",
        "family": "a100",
        "display_name": "NVIDIA A100 40GB",
        "price_per_hour": 1.29,
    },
    {
        "model_key": "A10_24GB",
        "family": "a10",
        "display_name": "NVIDIA A10",
        "price_per_hour": 0.45,
    },
    {
        "model_key": "A40_48GB",
        "family": "a40",
        "display_name": "NVIDIA A40",
        "price_per_hour": 0.85,
    },
    {
        "model_key": "A6000_48GB",
        "family": "a6000",
        "display_name": "NVIDIA RTX A6000",
        "price_per_hour": 0.75,
    },
    {
        "model_key": "L40S_48GB",
        "family": "l40s",
        "display_name": "NVIDIA L40S",
        "price_per_hour": 1.15,
    },
    {
        "model_key": "L40_48GB",
        "family": "l40",
        "display_name": "NVIDIA L40",
        "price_per_hour": 0.95,
    },
    {
        "model_key": "L4_24GB",
        "family": "l4",
        "display_name": "NVIDIA L4",
        "price_per_hour": 0.40,
    },
    {
        "model_key": "V100_32GB",
        "family": "v100",
        "display_name": "NVIDIA V100 32GB",
        "price_per_hour": 0.55,
    },
    {
        "model_key": "V100_16GB",
        "family": "v100",
        "display_name": "NVIDIA V100 16GB",
        "price_per_hour": 0.39,
    },
    {
        "model_key": "T4_16GB",
        "family": "t4",
        "display_name": "NVIDIA T4",
        "price_per_hour": 0.22,
    },
    {
        "model_key": "RTX_4090",
        "family": "rtx4090",
        "display_name": "GeForce RTX 4090",
        "price_per_hour": 0.45,
    },
    {
        "model_key": "RTX_3090",
        "family": "rtx3090",
        "display_name": "GeForce RTX 3090",
        "price_per_hour": 0.28,
    },
)


@dataclass(frozen=True, slots=True)
class SeedResult:
    """Outcome of :func:`seed_default_catalog`."""

    inserted: int
    updated: int
    skipped: bool
    total: int
    source: str



class PricingError(Exception):
    """Domain error for GPU price catalog operations."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _normalize_model_key(model_key: str | None) -> str:
    key = str(model_key or "").strip()
    if not key:
        raise PricingError(
            "model_key_required",
            "model_key is required",
            status_code=422,
        )
    return key


def _validate_price(price: Any) -> float:
    """Reject non-finite / ≤0 price (VAL-PRICE-010)."""

    if price is None:
        raise PricingError(
            "invalid_price",
            "price_per_hour must be a positive finite number",
            status_code=422,
        )
    try:
        number = float(price)
    except (TypeError, ValueError) as exc:
        raise PricingError(
            "invalid_price",
            "price_per_hour must be a positive finite number",
            status_code=422,
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise PricingError(
            "invalid_price",
            "price_per_hour must be a positive finite number",
            status_code=422,
        )
    return float(number)


def _validate_currency(currency: str | None) -> str:
    """v1 currency is USD only (omit → USD)."""

    if currency is None or (isinstance(currency, str) and currency.strip() == ""):
        return ALLOWED_CURRENCY
    cur = str(currency).strip()
    if cur != ALLOWED_CURRENCY:
        raise PricingError(
            "invalid_currency",
            f"currency must be {ALLOWED_CURRENCY}",
            status_code=422,
        )
    return ALLOWED_CURRENCY


def _coerce_active(active: bool | int | None, *, default: int = 1) -> int:
    if active is None:
        return int(default)
    if isinstance(active, bool):
        return 1 if active else 0
    try:
        return 1 if int(active) else 0
    except (TypeError, ValueError):
        return int(default)


def _derive_family(
    *,
    family: str | None,
    model_key: str,
    display_name: str | None,
) -> str:
    """Store canonical family when possible via normalize_gpu_model.

    Prefer explicit ``family`` (after normalize when free-form), else display
    name, else model_key. Fall back to lowercased model_key stem when
    unknown so writes never store an empty family.
    """

    candidates: list[str] = []
    if family is not None and str(family).strip():
        candidates.append(str(family).strip())
    if display_name is not None and str(display_name).strip():
        candidates.append(str(display_name).strip())
    candidates.append(model_key)

    for raw in candidates:
        normalized = normalize_gpu_model(raw)
        if normalized:
            return normalized
        # Accept already-canonical family tokens that fashion as lowercase keys
        lowered = raw.strip().lower().replace(" ", "").replace("_", "")
        # e.g. "h100", "rtx4090" passed explicitly as family
        again = normalize_gpu_model(lowered) or normalize_gpu_model(raw.strip().lower())
        if again:
            return again

    # Last resort: stable non-emptys string (never empty).
    fallback = (family or model_key or "unknown").strip().lower()
    # Convert model_key-ish H100_80GB → try first token
    if "_" in fallback:
        head = fallback.split("_", 1)[0]
        via = normalize_gpu_model(head)
        if via:
            return via
        return head
    return fallback or "unknown"


def _default_display_name(model_key: str, display_name: str | None) -> str:
    if display_name is not None and str(display_name).strip():
        return str(display_name).strip()
    return model_key.replace("_", " ")


def _append_history(
    session: AsyncSession,
    *,
    model_key: str,
    family: str,
    price_per_hour: float,
    currency: str,
    active_after: int,
    changed_by: str,
    reason: str | None,
    source: str,
    effective_from: Any,
) -> GpuPriceHistory:
    """Append-only history row for every catalog write."""

    now = utc_now()
    row = GpuPriceHistory(
        id=str(uuid.uuid4()),
        model_key=model_key,
        family=family,
        price_per_hour=float(price_per_hour),
        currency=currency,
        active_after=int(active_after),
        changed_by=str(changed_by or "admin")[:128],
        reason=reason,
        source=str(source or "admin")[:32],
        effective_from=effective_from or now,
        created_at=now,
    )
    session.add(row)
    return row


async def get_catalog_price(
    session: AsyncSession,
    model_key: str,
    *,
    active_only: bool = False,
) -> GpuPriceCatalog | None:
    """Load a single catalog row by model_key (optionally active only)."""

    key = str(model_key or "").strip()
    if not key:
        return None
    stmt = select(GpuPriceCatalog).where(GpuPriceCatalog.model_key == key)
    if active_only:
        stmt = stmt.where(GpuPriceCatalog.active == 1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_catalog_prices(
    session: AsyncSession,
    *,
    active_only: bool = True,
    family: str | None = None,
    model_key: str | None = None,
    limit: int = 500,
) -> list[GpuPriceCatalog]:
    """List catalog rows; public default is active-only (VAL-PRICE-013)."""

    lim = max(1, min(int(limit), 2000))
    stmt = select(GpuPriceCatalog)
    if active_only:
        stmt = stmt.where(GpuPriceCatalog.active == 1)
    if model_key is not None and str(model_key).strip():
        stmt = stmt.where(GpuPriceCatalog.model_key == str(model_key).strip())
    if family is not None and str(family).strip():
        fam = str(family).strip()
        # Match by normalized family when free-form is given
        normalized = normalize_gpu_model(fam) or fam.lower()
        stmt = stmt.where(GpuPriceCatalog.family == normalized)
    stmt = stmt.order_by(
        GpuPriceCatalog.family.asc(),
        GpuPriceCatalog.model_key.asc(),
    ).limit(lim)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_price_history(
    session: AsyncSession,
    model_key: str,
    *,
    limit: int = 100,
) -> list[GpuPriceHistory]:
    """History for a model_key, newest first."""

    key = str(model_key or "").strip()
    if not key:
        return []
    lim = max(1, min(int(limit), 1000))
    result = await session.execute(
        select(GpuPriceHistory)
        .where(GpuPriceHistory.model_key == key)
        .order_by(GpuPriceHistory.created_at.desc())
        .limit(lim)
    )
    return list(result.scalars().all())


async def upsert_catalog_price(
    session: AsyncSession,
    *,
    model_key: str,
    price_per_hour: Any,
    family: str | None = None,
    display_name: str | None = None,
    currency: str | None = None,
    active: bool | int | None = True,
    source: str = "admin",
    notes: str | None = None,
    max_offer_multiplier: float | None = None,
    min_offer_multiplier: float | None = None,
    changed_by: str | None = None,
    reason: str | None = None,
) -> GpuPriceCatalog:
    """Create or update a catalog row; always append history (VAL-PRICE-010).

    - price must be finite > 0
    - currency USD only (omit → USD)
    - active defaults to 1
    - bumps ``updated_at`` / ``effective_from`` to server now
    - one history row per call
    """

    key = _normalize_model_key(model_key)
    price = _validate_price(price_per_hour)
    cur = _validate_currency(currency)
    active_int = _coerce_active(active, default=1)
    disp = _default_display_name(key, display_name)
    fam = _derive_family(family=family, model_key=key, display_name=disp)
    src = str(source or "admin").strip()[:32] or "admin"
    actor = str(changed_by or src or "admin")[:128]
    now = utc_now()

    existing = await get_catalog_price(session, key, active_only=False)
    if existing is None:
        row = GpuPriceCatalog(
            id=str(uuid.uuid4()),
            model_key=key,
            family=fam,
            display_name=disp,
            price_per_hour=price,
            currency=cur,
            active=active_int,
            effective_from=now,
            source=src,
            notes=notes,
            max_offer_multiplier=max_offer_multiplier,
            min_offer_multiplier=min_offer_multiplier,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
    else:
        row = existing
        row.family = fam
        row.display_name = disp
        row.price_per_hour = price
        row.currency = cur
        row.active = active_int
        row.effective_from = now
        row.source = src
        row.updated_at = now
        if notes is not None:
            row.notes = notes
        if max_offer_multiplier is not None:
            row.max_offer_multiplier = max_offer_multiplier
        if min_offer_multiplier is not None:
            row.min_offer_multiplier = min_offer_multiplier

    _append_history(
        session,
        model_key=key,
        family=fam,
        price_per_hour=price,
        currency=cur,
        active_after=active_int,
        changed_by=actor,
        reason=reason,
        source=src,
        effective_from=now,
    )
    await session.flush()
    return row


async def disable_catalog_price(
    session: AsyncSession,
    *,
    model_key: str,
    changed_by: str | None = None,
    reason: str | None = None,
    source: str = "admin",
) -> GpuPriceCatalog:
    """Set active=0 without deleting; append history (VAL-PRICE-011)."""

    key = _normalize_model_key(model_key)
    row = await get_catalog_price(session, key, active_only=False)
    if row is None:
        raise PricingError(
            "catalog_not_found",
            f"catalog entry {key!r} not found",
            status_code=404,
        )
    now = utc_now()
    row.active = 0
    row.updated_at = now
    # Keep price/family/currency as last known; effective_from unchanged on
    # disable (price content did not change). History records the flip.
    src = str(source or "admin").strip()[:32] or "admin"
    actor = str(changed_by or src or "admin")[:128]
    _append_history(
        session,
        model_key=row.model_key,
        family=row.family,
        price_per_hour=float(row.price_per_hour),
        currency=row.currency or ALLOWED_CURRENCY,
        active_after=0,
        changed_by=actor,
        reason=reason,
        source=src,
        effective_from=row.effective_from or now,
    )
    await session.flush()
    return row


async def resolve_catalog_price(
    session: AsyncSession,
    *,
    model_key: str | None = None,
    gpu_model: str | None = None,
    family: str | None = None,
) -> GpuPriceCatalog | None:
    """Public resolve for offer defaults / catalog lookup (VAL-PRICE-012/013).

    Preference:
    1. Exact ``model_key`` if supplied and the row is **active**.
    2. Else active rows whose ``family`` matches normalize(gpu_model|family),
       ordered by ``effective_from DESC``, then ``model_key ASC``.
    3. Else ``None``.

    Inactive rows are never returned (VAL-PRICE-013).
    """

    key = str(model_key or "").strip() or None
    if key:
        hit = await get_catalog_price(session, key, active_only=True)
        if hit is not None:
            return hit
        # Exact inactive / missing → fall through only when family/gpu_model
        # also provided? Spec: exact key prefer; if supplied exact and
        # inactive, treat as not found for public resolve (no silent family
        # substitution under the same key). Family free-form path is separate.

    # Family resolution from free-form GPU name or explicit family.
    fam: str | None = None
    if family is not None and str(family).strip():
        fam = normalize_gpu_model(str(family).strip()) or str(family).strip().lower()
    if fam is None and gpu_model is not None and str(gpu_model).strip():
        fam = normalize_gpu_model(str(gpu_model).strip())
    # When only model_key was given and it was inactive, do not invent a family
    # from the key slug unless caller also passed gpu_model/family — for pure
    # model_key miss we stop at None.
    if fam is None and key is None:
        return None
    if fam is None:
        # Try normalize model_key as a secondary identity: only when exact
        # active miss already happened and no explicit family args. This
        # enables resolve(..., model_key omitted) via gpu_model; for pure
        # inactive exact key leave as None.
        return None

    result = await session.execute(
        select(GpuPriceCatalog)
        .where(
            GpuPriceCatalog.family == fam,
            GpuPriceCatalog.active == 1,
        )
        .order_by(
            GpuPriceCatalog.effective_from.desc(),
            GpuPriceCatalog.model_key.asc(),
        )
        .limit(1)
    )
    return result.scalar_one_or_none()


def catalog_row_to_public(
    row: GpuPriceCatalog,
    *,
    include_admin: bool = False,
) -> dict[str, Any]:
    """Serialize a catalog row (public strips notes by default)."""

    return row.to_dict(include_admin=include_admin)


def history_row_to_public(row: GpuPriceHistory) -> dict[str, Any]:
    """Serialize a history row for admin/API."""

    return row.to_dict()


async def seed_default_catalog(
    session: AsyncSession,
    *,
    only_if_empty: bool = True,
    source: str = "seed",
    ladder: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    changed_by: str | None = None,
    reason: str | None = "default seed ladder",
) -> SeedResult:
    """Bootstrap common GPU USD reference prices (VAL-PRICE-020 / 021).

    When ``only_if_empty=True`` (default): if *any* catalog row already exists
    the call is a **no-op** — never clobber admin/CLI reprices. When empty,
    insert the design ladder (≥10 model_keys: H100/A100/RTX/…).

    When ``only_if_empty=False``: upsert every ladder row (operators / tests).

    All written rows use ``source`` (default ``seed``), USD currency, active=1,
    finite ``price_per_hour > 0``. History is append-only via upsert path.
    """

    src = str(source or "seed").strip()[:32] or "seed"
    rows = tuple(ladder) if ladder is not None else DEFAULT_SEED_LADDER
    if not rows:
        return SeedResult(inserted=0, updated=0, skipped=True, total=0, source=src)

    existing_count = int(
        (
            await session.execute(select(func.count()).select_from(GpuPriceCatalog))
        ).scalar_one()
        or 0
    )

    if only_if_empty and existing_count > 0:
        return SeedResult(
            inserted=0,
            updated=0,
            skipped=True,
            total=existing_count,
            source=src,
        )

    actor = str(changed_by or src or "seed")[:128]
    why = reason if reason is not None else "default seed ladder"
    inserted = 0
    updated = 0

    for entry in rows:
        model_key = str(entry.get("model_key") or "").strip()
        if not model_key:
            continue
        price = entry.get("price_per_hour")
        family = entry.get("family")
        display_name = entry.get("display_name")
        prior = await get_catalog_price(session, model_key, active_only=False)
        await upsert_catalog_price(
            session,
            model_key=model_key,
            price_per_hour=price,
            family=str(family) if family is not None else None,
            display_name=str(display_name) if display_name is not None else None,
            currency=ALLOWED_CURRENCY,
            active=True,
            source=src,
            notes=entry.get("notes") if isinstance(entry.get("notes"), str) else None,
            changed_by=actor,
            reason=why,
        )
        if prior is None:
            inserted += 1
        else:
            updated += 1

    total = int(
        (
            await session.execute(select(func.count()).select_from(GpuPriceCatalog))
        ).scalar_one()
        or 0
    )
    return SeedResult(
        inserted=inserted,
        updated=updated,
        skipped=False,
        total=total,
        source=src,
    )


async def maybe_seed_prices_on_boot(
    database: Database,
    *,
    price_seed_on_boot: bool = False,
    only_if_empty: bool = True,
    source: str = "seed",
) -> SeedResult | None:
    """Optional boot hook for ``HYPER_PRICE_SEED_ON_BOOT`` (default off).

    When ``price_seed_on_boot`` is false, returns ``None`` without touching SQL.
    When true, opens a session and runs
    ``seed_default_catalog(only_if_empty=True, source=seed)`` then commits.
    Never clobbers admin rows while ``only_if_empty`` remains true.
    """

    if not price_seed_on_boot:
        return None
    async with database.session() as session:
        result = await seed_default_catalog(
            session,
            only_if_empty=only_if_empty,
            source=source,
            changed_by="seed",
            reason="boot seed (HYPER_PRICE_SEED_ON_BOOT)",
        )
        if not result.skipped and (result.inserted > 0 or result.updated > 0):
            await session.commit()
        else:
            await session.rollback()
        return result


__all__ = [
    "ALLOWED_CURRENCY",
    "DEFAULT_SEED_LADDER",
    "PricingError",
    "SeedResult",
    "catalog_row_to_public",
    "disable_catalog_price",
    "get_catalog_price",
    "history_row_to_public",
    "list_catalog_prices",
    "list_price_history",
    "maybe_seed_prices_on_boot",
    "resolve_catalog_price",
    "seed_default_catalog",
    "upsert_catalog_price",
]
