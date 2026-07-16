"""VAL-PRICE-010..013: domain pricing upsert / disable / resolve / history.

M11 domain CRUD slice:
- upsert sets finite >0 USD price, active=1, appends history
- disable sets active=0 + history; public resolve skips inactive
- family resolve via normalize_gpu_model (exact model_key prefer,
  else active family order effective_from DESC, model_key ASC)
- reject non-finite / <=0 price and non-USD currency
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hypercluster.db.database import Base
from hypercluster.db.models import GpuPriceCatalog, GpuPriceHistory, utc_now
from hypercluster.domain.pricing import (
    PricingError,
    disable_catalog_price,
    get_catalog_price,
    list_catalog_prices,
    list_price_history,
    resolve_catalog_price,
    upsert_catalog_price,
)
from hypercluster.probe.model_table import normalize_gpu_model


@pytest.fixture
async def db_session(tmp_path) -> AsyncSession:
    """Isolated SQLite session with full challenge metadata."""

    path = tmp_path / "gpu-price-crud.sqlite3"
    url = f"sqlite+aiosqlite:///{path}"
    engine = create_async_engine(url, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session
        await session.rollback()
    await engine.dispose()


# ---------------------------------------------------------------------------
# VAL-PRICE-010: Upsert sets price, USD, active, writes history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_sets_price_usd_active_and_appends_history(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-010: first upsert creates active USD catalog row + history."""

    row = await upsert_catalog_price(
        db_session,
        model_key="H100_80GB",
        price_per_hour=2.49,
        family="h100",
        display_name="NVIDIA H100 80GB",
        source="admin",
        changed_by="admin",
        reason="initial set",
    )
    await db_session.commit()

    assert row.model_key == "H100_80GB"
    assert row.family == "h100"
    assert row.price_per_hour == pytest.approx(2.49)
    assert row.currency == "USD"
    assert int(row.active) == 1
    assert row.source == "admin"
    assert row.effective_from is not None
    assert row.updated_at is not None

    hist = (
        (
            await db_session.execute(
                select(GpuPriceHistory).where(GpuPriceHistory.model_key == "H100_80GB")
            )
        )
        .scalars()
        .all()
    )
    assert len(hist) == 1
    assert hist[0].price_per_hour == pytest.approx(2.49)
    assert hist[0].currency == "USD"
    assert int(hist[0].active_after) == 1
    assert hist[0].changed_by == "admin"
    assert hist[0].reason == "initial set"
    assert hist[0].source == "admin"


@pytest.mark.asyncio
async def test_upsert_update_bumps_price_and_history_length(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-010: second upsert mutates same physical row; history +1 each time."""

    first = await upsert_catalog_price(
        db_session,
        model_key="A100_40GB",
        price_per_hour=1.29,
        family="a100",
        display_name="NVIDIA A100 40GB",
        source="seed",
        changed_by="seed",
    )
    await db_session.commit()
    first_id = first.id
    first_updated = first.updated_at

    second = await upsert_catalog_price(
        db_session,
        model_key="A100_40GB",
        price_per_hour=1.55,
        family="a100",
        display_name="NVIDIA A100 40GB",
        source="admin",
        changed_by="admin",
        reason="reprice",
    )
    await db_session.commit()

    assert second.id == first_id
    assert second.price_per_hour == pytest.approx(1.55)
    assert int(second.active) == 1
    assert second.currency == "USD"
    assert second.source == "admin"
    # updated_at / effective_from advanced (or at least set)
    assert second.updated_at is not None
    assert second.effective_from is not None
    if first_updated is not None and second.updated_at is not None:
        assert second.updated_at >= first_updated

    rows = (
        (
            await db_session.execute(
                select(GpuPriceCatalog).where(GpuPriceCatalog.model_key == "A100_40GB")
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1

    hist = (
        (
            await db_session.execute(
                select(GpuPriceHistory)
                .where(GpuPriceHistory.model_key == "A100_40GB")
                .order_by(GpuPriceHistory.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    assert len(hist) == 2
    assert hist[0].price_per_hour == pytest.approx(1.29)
    assert hist[1].price_per_hour == pytest.approx(1.55)
    assert hist[1].reason == "reprice"
    assert int(hist[1].active_after) == 1


@pytest.mark.asyncio
async def test_upsert_omitted_currency_defaults_usd(db_session: AsyncSession) -> None:
    """VAL-PRICE-010: currency omitted → USD."""

    row = await upsert_catalog_price(
        db_session,
        model_key="RTX_4090",
        price_per_hour=0.45,
        family="rtx4090",
        display_name="GeForce RTX 4090",
    )
    await db_session.commit()
    assert row.currency == "USD"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_price",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), "nope", None],
)
async def test_upsert_rejects_non_finite_or_non_positive_price(
    db_session: AsyncSession,
    bad_price: object,
) -> None:
    """VAL-PRICE-010: non-finite / <=0 price → invalid_price."""

    with pytest.raises(PricingError) as exc_info:
        await upsert_catalog_price(
            db_session,
            model_key="H100_80GB",
            price_per_hour=bad_price,  # type: ignore[arg-type]
            family="h100",
            display_name="NVIDIA H100 80GB",
        )
    err = exc_info.value
    assert err.code == "invalid_price"
    assert err.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_currency", ["EUR", "eur", "btc", "usd", "GBP"])
async def test_upsert_rejects_non_usd_currency(
    db_session: AsyncSession,
    bad_currency: str,
) -> None:
    """VAL-PRICE-010: non-USD currency → invalid_currency (422).

    Empty / omitted currency defaults to USD (not rejected).
    """

    with pytest.raises(PricingError) as exc_info:
        await upsert_catalog_price(
            db_session,
            model_key="H100_80GB",
            price_per_hour=2.0,
            family="h100",
            display_name="NVIDIA H100 80GB",
            currency=bad_currency,
        )
    err = exc_info.value
    assert err.code == "invalid_currency"
    assert err.status_code == 422


@pytest.mark.asyncio
async def test_upsert_empty_currency_defaults_usd(db_session: AsyncSession) -> None:
    """VAL-PRICE-010: empty-string currency treated as omitted → USD."""

    row = await upsert_catalog_price(
        db_session,
        model_key="L4_24GB",
        price_per_hour=0.4,
        family="l4",
        display_name="NVIDIA L4",
        currency="",
    )
    await db_session.commit()
    assert row.currency == "USD"


@pytest.mark.asyncio
async def test_upsert_rejects_empty_model_key(db_session: AsyncSession) -> None:
    """VAL-PRICE-010: empty model_key rejected."""

    with pytest.raises(PricingError) as exc_info:
        await upsert_catalog_price(
            db_session,
            model_key="  ",
            price_per_hour=1.0,
            family="h100",
            display_name="x",
        )
    assert exc_info.value.code == "model_key_required"
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# VAL-PRICE-011: Disable flips active=0 + history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disable_sets_active_zero_and_appends_history(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-011: disable keeps row, active=0, history active_after=0."""

    await upsert_catalog_price(
        db_session,
        model_key="L40S_48GB",
        price_per_hour=1.15,
        family="l40s",
        display_name="NVIDIA L40S",
        source="admin",
        changed_by="admin",
    )
    await db_session.commit()

    disabled = await disable_catalog_price(
        db_session,
        model_key="L40S_48GB",
        changed_by="admin",
        reason="retire sku",
        source="admin",
    )
    await db_session.commit()

    assert disabled is not None
    assert int(disabled.active) == 0
    assert disabled.model_key == "L40S_48GB"
    # Not hard-deleted
    loaded = await get_catalog_price(db_session, "L40S_48GB", active_only=False)
    assert loaded is not None
    assert int(loaded.active) == 0

    public = await get_catalog_price(db_session, "L40S_48GB", active_only=True)
    assert public is None

    hist = await list_price_history(db_session, "L40S_48GB")
    assert len(hist) == 2
    # Newest first
    assert int(hist[0].active_after) == 0
    assert hist[0].reason == "retire sku"
    assert int(hist[1].active_after) == 1


@pytest.mark.asyncio
async def test_disable_missing_model_key_raises(db_session: AsyncSession) -> None:
    """VAL-PRICE-011: disable unknown key → catalog_not_found."""

    with pytest.raises(PricingError) as exc_info:
        await disable_catalog_price(db_session, model_key="NO_SUCH_KEY")
    assert exc_info.value.code == "catalog_not_found"
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_active_only_excludes_disabled(db_session: AsyncSession) -> None:
    """VAL-PRICE-011 / 013: public list omits inactive rows."""

    await upsert_catalog_price(
        db_session,
        model_key="H100_80GB",
        price_per_hour=2.49,
        family="h100",
        display_name="H100",
    )
    await upsert_catalog_price(
        db_session,
        model_key="A100_40GB",
        price_per_hour=1.29,
        family="a100",
        display_name="A100",
    )
    await db_session.commit()
    await disable_catalog_price(db_session, model_key="A100_40GB")
    await db_session.commit()

    public = await list_catalog_prices(db_session, active_only=True)
    keys = {r.model_key for r in public}
    assert "H100_80GB" in keys
    assert "A100_40GB" not in keys

    admin = await list_catalog_prices(db_session, active_only=False)
    admin_keys = {r.model_key for r in admin}
    assert "H100_80GB" in admin_keys
    assert "A100_40GB" in admin_keys


# ---------------------------------------------------------------------------
# VAL-PRICE-012: normalize_gpu_model family joins catalog resolve
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_prefers_exact_model_key(db_session: AsyncSession) -> None:
    """VAL-PRICE-012: exact model_key wins over family free-form peers."""

    await upsert_catalog_price(
        db_session,
        model_key="A100_40GB",
        price_per_hour=1.29,
        family="a100",
        display_name="NVIDIA A100 40GB",
    )
    await upsert_catalog_price(
        db_session,
        model_key="A100_80GB",
        price_per_hour=1.89,
        family="a100",
        display_name="NVIDIA A100 80GB",
    )
    await db_session.commit()

    resolved = await resolve_catalog_price(db_session, model_key="A100_40GB")
    assert resolved is not None
    assert resolved.model_key == "A100_40GB"
    assert resolved.price_per_hour == pytest.approx(1.29)


@pytest.mark.asyncio
async def test_resolve_family_via_normalize_gpu_model(db_session: AsyncSession) -> None:
    """VAL-PRICE-012: free-form name → family via normalize_gpu_model → catalog."""

    assert normalize_gpu_model("NVIDIA H100 80GB") == "h100"

    await upsert_catalog_price(
        db_session,
        model_key="H100_80GB",
        price_per_hour=2.49,
        family="h100",
        display_name="NVIDIA H100 80GB",
    )
    await db_session.commit()

    resolved = await resolve_catalog_price(
        db_session,
        gpu_model="NVIDIA H100 80GB",
    )
    assert resolved is not None
    assert resolved.model_key == "H100_80GB"
    assert resolved.family == "h100"
    assert resolved.price_per_hour == pytest.approx(2.49)


@pytest.mark.asyncio
async def test_resolve_family_order_effective_from_desc_then_model_key_asc(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-012: family pick orders effective_from DESC, model_key ASC."""

    older = utc_now() - timedelta(hours=2)
    newer = utc_now() - timedelta(minutes=5)

    # Insert older H100_80GB first.
    row_a = await upsert_catalog_price(
        db_session,
        model_key="H100_80GB",
        price_per_hour=2.0,
        family="h100",
        display_name="H100 80GB",
    )
    # Insert another family peer with higher key alphabetically but older Unix.
    row_b = await upsert_catalog_price(
        db_session,
        model_key="H100_PCIE",
        price_per_hour=1.75,
        family="h100",
        display_name="H100 PCIe",
    )
    await db_session.commit()

    # Force effective_from ordering: H100_PCIE newest, H100_80GB older.
    row_a.effective_from = older
    row_b.effective_from = newer
    await db_session.commit()

    resolved = await resolve_catalog_price(db_session, gpu_model="h100")
    assert resolved is not None
    assert resolved.model_key == "H100_PCIE"
    assert resolved.price_per_hour == pytest.approx(1.75)

    # Tie on effective_from: model_key ASC (H100_80GB before H100_PCIE)
    same_ts = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)
    row_a.effective_from = same_ts
    row_b.effective_from = same_ts
    await db_session.commit()

    resolved_tie = await resolve_catalog_price(db_session, gpu_model="NVIDIA H100")
    assert resolved_tie is not None
    assert resolved_tie.model_key == "H100_80GB"


@pytest.mark.asyncio
async def test_resolve_unknown_family_returns_none(db_session: AsyncSession) -> None:
    """VAL-PRICE-012: unknown free-form GPU → None."""

    await upsert_catalog_price(
        db_session,
        model_key="H100_80GB",
        price_per_hour=2.49,
        family="h100",
        display_name="H100",
    )
    await db_session.commit()

    assert await resolve_catalog_price(db_session, gpu_model="TotallyFakeGPU 9000") is None
    assert await resolve_catalog_price(db_session, gpu_model="") is None
    assert await resolve_catalog_price(db_session) is None


# ---------------------------------------------------------------------------
# VAL-PRICE-013: Inactive excluded from public resolve default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inactive_excluded_from_public_resolve(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-013: disabled row never chosen for public resolve / default."""

    await upsert_catalog_price(
        db_session,
        model_key="V100_32GB",
        price_per_hour=0.55,
        family="v100",
        display_name="NVIDIA V100 32GB",
    )
    await db_session.commit()
    await disable_catalog_price(db_session, model_key="V100_32GB")
    await db_session.commit()

    # Exact key resolve (public/active) skips inactive
    assert await resolve_catalog_price(db_session, model_key="V100_32GB") is None
    # Family free-form also skips
    assert (
        await resolve_catalog_price(db_session, gpu_model="NVIDIA Tesla V100 32GB")
        is None
    )
    # Admin-style get still finds it
    admin_row = await get_catalog_price(db_session, "V100_32GB", active_only=False)
    assert admin_row is not None
    assert int(admin_row.active) == 0


@pytest.mark.asyncio
async def test_resolve_falls_through_to_active_family_peer(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-013: if exact key inactive, family path can still hit active peer."""

    await upsert_catalog_price(
        db_session,
        model_key="A100_40GB",
        price_per_hour=1.29,
        family="a100",
        display_name="A100 40",
    )
    await upsert_catalog_price(
        db_session,
        model_key="A100_80GB",
        price_per_hour=1.89,
        family="a100",
        display_name="A100 80",
    )
    await db_session.commit()
    await disable_catalog_price(db_session, model_key="A100_40GB")
    await db_session.commit()

    # Exact inactive key → None for public resolve
    assert await resolve_catalog_price(db_session, model_key="A100_40GB") is None
    # Free-form family still resolves remaining active peer
    resolved = await resolve_catalog_price(db_session, gpu_model="NVIDIA A100")
    assert resolved is not None
    assert resolved.model_key == "A100_80GB"
    assert int(resolved.active) == 1


@pytest.mark.asyncio
async def test_upsert_reactivate_via_active_flag(db_session: AsyncSession) -> None:
    """Disable then upsert with active=1 reopens for public resolve."""

    await upsert_catalog_price(
        db_session,
        model_key="T4_16GB",
        price_per_hour=0.22,
        family="t4",
        display_name="NVIDIA T4",
    )
    await db_session.commit()
    await disable_catalog_price(db_session, model_key="T4_16GB")
    await db_session.commit()
    assert await resolve_catalog_price(db_session, model_key="T4_16GB") is None

    await upsert_catalog_price(
        db_session,
        model_key="T4_16GB",
        price_per_hour=0.25,
        family="t4",
        display_name="NVIDIA T4",
        active=True,
        reason="reactivate",
    )
    await db_session.commit()
    resolved = await resolve_catalog_price(db_session, model_key="T4_16GB")
    assert resolved is not None
    assert resolved.price_per_hour == pytest.approx(0.25)
    assert int(resolved.active) == 1


@pytest.mark.asyncio
async def test_family_normalized_on_upsert_from_display_name(
    db_session: AsyncSession,
) -> None:
    """Family free-form path can derive n via normalize when family omitted."""

    row = await upsert_catalog_price(
        db_session,
        model_key="H200_141GB",
        price_per_hour=3.49,
        display_name="NVIDIA H200",
        # family omitted → derive from display_name / model_key via normalize
    )
    await db_session.commit()
    assert row.family == "h200"
    # Ensure resolve by normalize works
    assert normalize_gpu_model("NVIDIA H200 141GB") == "h200"
    resolved = await resolve_catalog_price(db_session, gpu_model="NVIDIA H200 141GB")
    assert resolved is not None
    assert resolved.model_key == "H200_141GB"


def test_pricing_error_is_domain_error() -> None:
    """Smoke: PricingError carries code + status."""

    err = PricingError("invalid_price", "bad", status_code=422)
    assert err.code == "invalid_price"
    assert err.status_code == 422
    assert "bad" in str(err)
    # NaN is not finite; module must reject it
    assert not math.isfinite(float("nan"))
