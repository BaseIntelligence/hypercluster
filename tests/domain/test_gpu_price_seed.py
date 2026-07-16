"""VAL-PRICE-020 / VAL-PRICE-021: seed default USD GPU catalog ladder.

M11 seed-defaults slice:
- Empty catalog + seed_default_catalog(only_if_empty=True, source=seed)
  inserts ≥10 common model_keys (H100/A100/RTX/… design ladder)
- only_if_empty is a no-op when rows already exist (never clobber admin)
- source=seed on seeded rows; operators reprice via admin/CLI after seed
- Optional HYPER_PRICE_SEED_ON_BOOT wires empty-table seed at database.init
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hypercluster.db.database import Base, Database
from hypercluster.db.models import GpuPriceCatalog, GpuPriceHistory
from hypercluster.domain.pricing import (
    DEFAULT_SEED_LADDER,
    seed_default_catalog,
    upsert_catalog_price,
)


@pytest.fixture
async def db_session(tmp_path) -> AsyncSession:
    """Isolated SQLite session with full challenge metadata."""

    path = tmp_path / "gpu-price-seed.sqlite3"
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


async def _catalog_count(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(GpuPriceCatalog))
    return int(result.scalar_one())


async def _price_map(session: AsyncSession) -> dict[str, float]:
    rows = (await session.execute(select(GpuPriceCatalog))).scalars().all()
    return {r.model_key: float(r.price_per_hour) for r in rows}


# ---------------------------------------------------------------------------
# VAL-PRICE-020: Seed inserts ≥10 common families when empty
# ---------------------------------------------------------------------------


def test_default_seed_ladder_has_at_least_ten_common_keys() -> None:
    """Design ladder exposes ≥10 distinct model_keys spanning multiple families."""

    keys = [row["model_key"] for row in DEFAULT_SEED_LADDER]
    assert len(keys) >= 10
    assert len(set(keys)) == len(keys)

    families = {row["family"] for row in DEFAULT_SEED_LADDER}
    # H100 / A100 / consumer RTX (... design ladder)
    assert "h100" in families
    assert "a100" in families
    assert "rtx4090" in families or "rtx3090" in families
    # Concretes named in assertion evidence
    key_set = set(keys)
    assert "H100_80GB" in key_set
    assert "A100_40GB" in key_set
    assert "RTX_4090" in key_set


@pytest.mark.asyncio
async def test_seed_empty_catalog_inserts_design_ladder(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-020: empty DB → ≥10 USD active source=seed rows after seed."""

    assert await _catalog_count(db_session) == 0

    result = await seed_default_catalog(
        db_session,
        only_if_empty=True,
        source="seed",
    )
    await db_session.commit()

    assert result.inserted >= 10
    assert result.skipped is False
    assert result.total >= 10

    rows = (
        (await db_session.execute(select(GpuPriceCatalog).order_by(GpuPriceCatalog.model_key)))
        .scalars()
        .all()
    )
    assert len(rows) >= 10

    keys = {r.model_key for r in rows}
    assert "H100_80GB" in keys
    assert "A100_40GB" in keys
    assert "RTX_4090" in keys
    # Multiple families on the design ladder
    families = {r.family for r in rows}
    assert "h100" in families
    assert "a100" in families
    assert "rtx4090" in families or "rtx3090" in families

    for row in rows:
        assert row.currency == "USD"
        assert int(row.active) == 1
        assert row.source == "seed"
        assert float(row.price_per_hour) > 0.0
        assert row.price_per_hour == row.price_per_hour  # not NaN
        assert row.effective_from is not None

    # One history row per seeded model_key
    hist_count = (
        await db_session.execute(select(func.count()).select_from(GpuPriceHistory))
    ).scalar_one()
    assert int(hist_count) == len(rows)
    hist_sources = (
        (await db_session.execute(select(GpuPriceHistory.source).distinct())).scalars().all()
    )
    assert set(hist_sources) == {"seed"}


@pytest.mark.asyncio
async def test_seed_ladder_prices_match_design_defaults(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-020: seeded prices match the design USD ladder (mid-market OOM)."""

    await seed_default_catalog(db_session, only_if_empty=True, source="seed")
    await db_session.commit()

    by_key = await _price_map(db_session)
    # Spot-check design values (operators reprice after seed)
    assert by_key["H100_80GB"] == pytest.approx(2.49)
    assert by_key["H200_141GB"] == pytest.approx(3.49)
    assert by_key["A100_80GB"] == pytest.approx(1.89)
    assert by_key["A100_40GB"] == pytest.approx(1.29)
    assert by_key["RTX_4090"] == pytest.approx(0.45)
    assert by_key["T4_16GB"] == pytest.approx(0.22)


# ---------------------------------------------------------------------------
# VAL-PRICE-021: Seed is no-op when catalog non-empty (only_if_empty)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_only_if_empty_is_noop_when_rows_exist(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-021: second seed with only_if_empty leaves count + prices alone."""

    first = await seed_default_catalog(
        db_session,
        only_if_empty=True,
        source="seed",
    )
    await db_session.commit()
    assert first.inserted >= 10
    count_after_first = await _catalog_count(db_session)
    prices_after_first = await _price_map(db_session)

    # Simulate operator reprice of a seeded row
    await upsert_catalog_price(
        db_session,
        model_key="H100_80GB",
        price_per_hour=9.99,
        family="h100",
        display_name="NVIDIA H100 80GB",
        source="admin",
        changed_by="admin",
        reason="operator reprice",
    )
    await db_session.commit()
    assert (await _price_map(db_session))["H100_80GB"] == pytest.approx(9.99)

    second = await seed_default_catalog(
        db_session,
        only_if_empty=True,
        source="seed",
    )
    await db_session.commit()

    assert second.inserted == 0
    assert second.skipped is True
    assert await _catalog_count(db_session) == count_after_first
    # Admin reprice must not be clobbered back to 2.49
    assert (await _price_map(db_session))["H100_80GB"] == pytest.approx(9.99)
    # Unrelated keys still stable
    for key, price in prices_after_first.items():
        if key == "H100_80GB":
            continue
        assert (await _price_map(db_session))[key] == pytest.approx(price)


@pytest.mark.asyncio
async def test_seed_only_if_empty_skips_when_single_admin_row_exists(
    db_session: AsyncSession,
) -> None:
    """VAL-PRICE-021: any existing row (even one admin key) blocks seed ladder."""

    await upsert_catalog_price(
        db_session,
        model_key="CUSTOM_GPU",
        price_per_hour=1.11,
        family="h100",
        display_name="Custom",
        source="admin",
        changed_by="admin",
    )
    await db_session.commit()
    assert await _catalog_count(db_session) == 1

    result = await seed_default_catalog(
        db_session,
        only_if_empty=True,
        source="seed",
    )
    await db_session.commit()

    assert result.skipped is True
    assert result.inserted == 0
    assert await _catalog_count(db_session) == 1
    row = (
        await db_session.execute(
            select(GpuPriceCatalog).where(GpuPriceCatalog.model_key == "CUSTOM_GPU")
        )
    ).scalar_one()
    assert row.price_per_hour == pytest.approx(1.11)
    assert row.source == "admin"


@pytest.mark.asyncio
async def test_seed_only_if_empty_false_upserts_missing_keys(
    db_session: AsyncSession,
) -> None:
    """Force path (only_if_empty=False) can fill missing keys without requiring empty.

    Used by operators/tests; default boot stays only_if_empty=True.
    """

    await upsert_catalog_price(
        db_session,
        model_key="H100_80GB",
        price_per_hour=9.99,
        family="h100",
        display_name="NVIDIA H100 80GB",
        source="admin",
        changed_by="admin",
    )
    await db_session.commit()

    result = await seed_default_catalog(
        db_session,
        only_if_empty=False,
        source="seed",
    )
    await db_session.commit()

    assert result.skipped is False
    # At least fills other ladder keys; total becomes full ladder size
    assert await _catalog_count(db_session) >= 10
    # Admin key still present (upsert may reprice depending on force semantics;
    # force path upgrades to ladder prices for all ladder keys including H100)
    keys = set((await _price_map(db_session)).keys())
    assert "H100_80GB" in keys
    assert "RTX_4090" in keys


# ---------------------------------------------------------------------------
# HYPER_PRICE_SEED_ON_BOOT optional wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_database_init_seeds_when_price_seed_on_boot_enabled(
    tmp_path,
) -> None:
    """Boot path: Database.init + price_seed_on_boot seeds empty catalog."""

    from hypercluster.domain.pricing import maybe_seed_prices_on_boot

    path = tmp_path / "boot-seed.sqlite3"
    db = Database(f"sqlite+aiosqlite:///{path}")
    await db.init()
    try:
        async with db.session() as session:
            assert await _catalog_count(session) == 0

        # Flag on → seed
        result = await maybe_seed_prices_on_boot(db, price_seed_on_boot=True)
        assert result is not None
        assert result.inserted >= 10
        assert result.skipped is False

        async with db.session() as session:
            assert await _catalog_count(session) >= 10
            rows = (await session.execute(select(GpuPriceCatalog))).scalars().all()
            assert all(r.source == "seed" for r in rows)

        # Second boot path is no-op (only_if_empty)
        result2 = await maybe_seed_prices_on_boot(db, price_seed_on_boot=True)
        assert result2 is not None
        assert result2.skipped is True
        assert result2.inserted == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_database_init_skips_seed_when_price_seed_on_boot_disabled(
    tmp_path,
) -> None:
    """Default flag off leaves catalog empty after boot helper."""

    from hypercluster.domain.pricing import maybe_seed_prices_on_boot

    path = tmp_path / "boot-noseed.sqlite3"
    db = Database(f"sqlite+aiosqlite:///{path}")
    await db.init()
    try:
        result = await maybe_seed_prices_on_boot(db, price_seed_on_boot=False)
        assert result is None
        async with db.session() as session:
            assert await _catalog_count(session) == 0
    finally:
        await db.close()


def test_hyper_settings_price_seed_on_boot_default_false(monkeypatch) -> None:
    """HYPER_PRICE_SEED_ON_BOOT defaults false; env can flip true."""

    from hypercluster.settings import clear_settings_cache, get_hyper_settings

    clear_settings_cache()
    monkeypatch.delenv("HYPER_PRICE_SEED_ON_BOOT", raising=False)
    assert get_hyper_settings().price_seed_on_boot is False

    clear_settings_cache()
    monkeypatch.setenv("HYPER_PRICE_SEED_ON_BOOT", "true")
    assert get_hyper_settings().price_seed_on_boot is True

    clear_settings_cache()
    monkeypatch.delenv("HYPER_PRICE_SEED_ON_BOOT", raising=False)
    clear_settings_cache()
