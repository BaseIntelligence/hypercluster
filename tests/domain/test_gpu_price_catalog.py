"""VAL-PRICE-001 / VAL-PRICE-002: GPU price catalog + history SQLite models.

M11 schema slice only:
- Durable ``gpu_price_catalog`` / ``gpu_price_history`` on challenge SQLite.
- Catalog ``model_key`` UNIQUE; history is append-only (no unique on key).
- No product pricing ladder hardcoded outside seed (this module is schema only).
- Never set_weights; four-factor formula untouched; no product Verda.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hypercluster.db.database import Base, Database
from hypercluster.db.models import GpuPriceCatalog, GpuPriceHistory, utc_now


@pytest.fixture
async def db_session(tmp_path) -> AsyncSession:
    """Isolated SQLite session with full challenge metadata."""

    path = tmp_path / "gpu-prices.sqlite3"
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


@pytest.fixture
async def database(tmp_path) -> Database:
    """Database wrapper so init path also creates price tables (VAL-PRICE-001)."""

    path = tmp_path / "challenge-prices.sqlite3"
    db = Database(f"sqlite+aiosqlite:///{path}")
    await db.init()
    try:
        yield db
    finally:
        await db.close()


def _catalog_row(
    *,
    model_key: str = "H100_80GB",
    family: str = "h100",
    display_name: str = "NVIDIA H100 80GB",
    price_per_hour: float = 2.49,
    currency: str = "USD",
    active: int = 1,
    source: str = "admin",
    notes: str | None = None,
    max_offer_multiplier: float | None = None,
    min_offer_multiplier: float | None = None,
) -> GpuPriceCatalog:
    now = utc_now()
    return GpuPriceCatalog(
        id=str(uuid.uuid4()),
        model_key=model_key,
        family=family,
        display_name=display_name,
        price_per_hour=float(price_per_hour),
        currency=currency,
        active=int(active),
        effective_from=now,
        source=source,
        notes=notes,
        max_offer_multiplier=max_offer_multiplier,
        min_offer_multiplier=min_offer_multiplier,
        created_at=now,
        updated_at=now,
    )


def _history_row(
    *,
    model_key: str = "H100_80GB",
    family: str = "h100",
    price_per_hour: float = 2.49,
    currency: str = "USD",
    active_after: int = 1,
    changed_by: str = "admin",
    reason: str | None = "set price",
    source: str = "admin",
) -> GpuPriceHistory:
    now = utc_now()
    return GpuPriceHistory(
        id=str(uuid.uuid4()),
        model_key=model_key,
        family=family,
        price_per_hour=float(price_per_hour),
        currency=currency,
        active_after=int(active_after),
        changed_by=changed_by,
        reason=reason,
        source=source,
        effective_from=now,
        created_at=now,
    )


# ---------------------------------------------------------------------------
# VAL-PRICE-001: tables exist after create_all / Database.init
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_price_tables_created_by_database_init(database: Database) -> None:
    """VAL-PRICE-001: Database.init creates durable catalog + history tables."""

    async with database.engine.connect() as conn:
        table_names = await conn.run_sync(
            lambda sync_conn: set(inspect(sync_conn).get_table_names())
        )
        catalog_cols = await conn.run_sync(
            lambda sync_conn: {
                c["name"] for c in inspect(sync_conn).get_columns("gpu_price_catalog")
            }
        )
        history_cols = await conn.run_sync(
            lambda sync_conn: {
                c["name"] for c in inspect(sync_conn).get_columns("gpu_price_history")
            }
        )

    assert "gpu_price_catalog" in table_names
    assert "gpu_price_history" in table_names

    for required in (
        "id",
        "model_key",
        "family",
        "display_name",
        "price_per_hour",
        "currency",
        "active",
        "effective_from",
        "source",
        "created_at",
        "updated_at",
    ):
        assert required in catalog_cols, f"catalog missing column {required}"

    for required in (
        "id",
        "model_key",
        "family",
        "price_per_hour",
        "currency",
        "active_after",
        "changed_by",
        "source",
        "effective_from",
        "created_at",
    ):
        assert required in history_cols, f"history missing column {required}"


@pytest.mark.asyncio
async def test_price_tables_created_by_metadata_create_all(db_session: AsyncSession) -> None:
    """VAL-PRICE-001: Base.metadata.create_all registers price tables."""

    assert "gpu_price_catalog" in Base.metadata.tables
    assert "gpu_price_history" in Base.metadata.tables
    bind = db_session.get_bind()
    assert bind is not None
    # Reflect via a real connection (inspect needs Connection/Engine, not Session).
    conn = await db_session.connection()
    table_names = await conn.run_sync(
        lambda sync_conn: set(inspect(sync_conn).get_table_names())
    )
    assert "gpu_price_catalog" in table_names
    assert "gpu_price_history" in table_names


@pytest.mark.asyncio
async def test_catalog_and_history_accept_rows_roundtrip(db_session: AsyncSession) -> None:
    """VAL-PRICE-001: both tables accept rows and round-trip under ORM."""

    cat = _catalog_row(model_key="A100_40GB", family="a100", price_per_hour=1.55)
    hist = _history_row(
        model_key="A100_40GB",
        family="a100",
        price_per_hour=1.55,
        changed_by="seed",
        source="seed",
    )
    db_session.add(cat)
    db_session.add(hist)
    await db_session.commit()

    loaded_cat = (
        await db_session.execute(
            select(GpuPriceCatalog).where(GpuPriceCatalog.model_key == "A100_40GB")
        )
    ).scalar_one()
    assert loaded_cat.family == "a100"
    assert loaded_cat.price_per_hour == pytest.approx(1.55)
    assert loaded_cat.currency == "USD"
    assert loaded_cat.active == 1
    assert loaded_cat.source == "admin"
    public = loaded_cat.to_dict()
    assert public["model_key"] == "A100_40GB"
    assert public["price_per_hour"] == pytest.approx(1.55)
    assert public["currency"] == "USD"
    assert "effective_from" in public

    loaded_hist = (
        await db_session.execute(
            select(GpuPriceHistory).where(GpuPriceHistory.model_key == "A100_40GB")
        )
    ).scalar_one()
    assert loaded_hist.active_after == 1
    assert loaded_hist.changed_by == "seed"
    assert loaded_hist.price_per_hour == pytest.approx(1.55)
    hist_public = loaded_hist.to_dict()
    assert hist_public["model_key"] == "A100_40GB"
    assert hist_public["active_after"] == 1
    assert "created_at" in hist_public


@pytest.mark.asyncio
async def test_history_accepts_multiple_appends_per_model_key(db_session: AsyncSession) -> None:
    """VAL-PRICE-001: history is append-only; multiple rows per model_key OK."""

    db_session.add(_catalog_row(model_key="RTX_4090", family="rtx4090", price_per_hour=0.6))
    db_session.add(
        _history_row(model_key="RTX_4090", family="rtx4090", price_per_hour=0.6, source="seed")
    )
    db_session.add(
        _history_row(
            model_key="RTX_4090",
            family="rtx4090",
            price_per_hour=0.55,
            source="admin",
            changed_by="admin",
            reason="price cut",
        )
    )
    await db_session.commit()

    rows = (
        (
            await db_session.execute(
                select(GpuPriceHistory).where(GpuPriceHistory.model_key == "RTX_4090")
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2
    prices = sorted(r.price_per_hour for r in rows)
    assert prices == pytest.approx([0.55, 0.6])


# ---------------------------------------------------------------------------
# VAL-PRICE-002: model_key UNIQUE on catalog
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_catalog_model_key_unique_constraint(db_session: AsyncSession) -> None:
    """VAL-PRICE-002: second physical catalog row for same model_key is rejected."""

    first = _catalog_row(model_key="H100_80GB", price_per_hour=2.49)
    db_session.add(first)
    await db_session.commit()

    second = _catalog_row(model_key="H100_80GB", price_per_hour=3.0, display_name="dup")
    db_session.add(second)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()

    rows = (
        (
            await db_session.execute(
                select(GpuPriceCatalog).where(GpuPriceCatalog.model_key == "H100_80GB")
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].id == first.id
    assert rows[0].price_per_hour == pytest.approx(2.49)


@pytest.mark.asyncio
async def test_catalog_unique_does_not_block_history_append(db_session: AsyncSession) -> None:
    """VAL-PRICE-002: unique is on catalog only; history still appends freely."""

    db_session.add(_catalog_row(model_key="L40S", family="l40s", price_per_hour=1.2))
    await db_session.commit()

    for i, price in enumerate((1.2, 1.1, 1.0)):
        db_session.add(
            _history_row(
                model_key="L40S",
                family="l40s",
                price_per_hour=price,
                reason=f"rev-{i}",
                source="admin",
            )
        )
    await db_session.commit()

    catalog_count = (
        await db_session.execute(
            select(GpuPriceCatalog).where(GpuPriceCatalog.model_key == "L40S")
        )
    ).scalars().all()
    history_count = (
        await db_session.execute(
            select(GpuPriceHistory).where(GpuPriceHistory.model_key == "L40S")
        )
    ).scalars().all()
    assert len(catalog_count) == 1
    assert len(history_count) == 3


@pytest.mark.asyncio
async def test_catalog_row_raw_sql_insert(database: Database) -> None:
    """VAL-PRICE-001: raw SQL insert works (durable create_all surface)."""

    row_id = str(uuid.uuid4())
    now = utc_now().isoformat()
    async with database.session() as session:
        await session.execute(
            text(
                """
                INSERT INTO gpu_price_catalog (
                    id, model_key, family, display_name, price_per_hour,
                    currency, active, effective_from, source, notes,
                    max_offer_multiplier, min_offer_multiplier,
                    created_at, updated_at
                ) VALUES (
                    :id, :model_key, :family, :display_name, :price_per_hour,
                    :currency, :active, :effective_from, :source, :notes,
                    :max_offer_multiplier, :min_offer_multiplier,
                    :created_at, :updated_at
                )
                """
            ),
            {
                "id": row_id,
                "model_key": "B200_180GB",
                "family": "b200",
                "display_name": "NVIDIA B200 180GB",
                "price_per_hour": 4.5,
                "currency": "USD",
                "active": 1,
                "effective_from": now,
                "source": "seed",
                "notes": None,
                "max_offer_multiplier": None,
                "min_offer_multiplier": None,
                "created_at": now,
                "updated_at": now,
            },
        )
        await session.commit()
        result = await session.execute(
            text(
                "SELECT model_key, family, price_per_hour, currency, active "
                "FROM gpu_price_catalog WHERE id = :id"
            ),
            {"id": row_id},
        )
        row = result.one()
        assert row[0] == "B200_180GB"
        assert row[1] == "b200"
        assert float(row[2]) == pytest.approx(4.5)
        assert row[3] == "USD"
        assert int(row[4]) == 1
