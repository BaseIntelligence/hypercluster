"""VAL-WGT-001 / VAL-WGT-020: points ledger (+ balances) durable SQLite model.

M10 model slice only:
- Durable ``points_ledger`` / optional ``points_balances`` on challenge SQLite.
- Columns: hotkey, delta, reason, attempt_id/score_id, timestamps; unique earn
  per attempt_id for score_earn rows.
- No SQL ``gpu_price_catalog``; offer ``price_per_hour`` create path unchanged.
- Never set_weights; four-factor formula untouched (no earn side-effects here).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hypercluster.db.database import Base, Database
from hypercluster.db.models import Offer, PointsBalance, PointsLedger, utc_now

HOTKEY_A = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
HOTKEY_B = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


@pytest.fixture
async def db_session(tmp_path) -> AsyncSession:
    """Isolated SQLite session with full challenge metadata."""

    path = tmp_path / "points.sqlite3"
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
    """Database wrapper so init path also creates points tables (VAL-WGT-001)."""

    path = tmp_path / "challenge-points.sqlite3"
    db = Database(f"sqlite+aiosqlite:///{path}")
    await db.init()
    try:
        yield db
    finally:
        await db.close()


def _ledger_row(
    *,
    hotkey: str = HOTKEY_A,
    delta: float = 1.25,
    balance_after: float = 1.25,
    reason: str = "score_earn",
    attempt_id: str | None = None,
    score_id: str | None = None,
    role: str | None = "demand",
    details: dict[str, Any] | None = None,
) -> PointsLedger:
    row_id = str(uuid.uuid4())
    attempt = attempt_id if attempt_id is not None else str(uuid.uuid4())
    return PointsLedger(
        id=row_id,
        hotkey=hotkey,
        role=role,
        delta=float(delta),
        balance_after=float(balance_after),
        reason=reason,
        score_id=score_id,
        attempt_id=attempt if reason == "score_earn" else attempt_id,
        details_json=None if details is None else __import__("json").dumps(details),
        created_at=utc_now(),
    )


# ---------------------------------------------------------------------------
# VAL-WGT-001: schema exists + holds earn rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_points_tables_created_by_database_init(database: Database) -> None:
    """VAL-WGT-001: Database.init creates durable points_ledger + points_balances."""

    async with database.engine.connect() as conn:
        table_names = await conn.run_sync(
            lambda sync_conn: set(inspect(sync_conn).get_table_names())
        )
    assert "points_ledger" in table_names
    assert "points_balances" in table_names
    # Explicit: no SQL gpu_price_catalog this round (VAL-WGT-001 / VAL-WGT-020).
    assert "gpu_price_catalog" not in table_names
    assert "gpu_price_revisions" not in table_names


@pytest.mark.asyncio
async def test_points_ledger_columns_support_earn_row(db_session: AsyncSession) -> None:
    """VAL-WGT-001: ledger columns capture hotkey/delta/reason/attempt|score/ts."""

    attempt_id = str(uuid.uuid4())
    score_id = str(uuid.uuid4())
    row = _ledger_row(
        hotkey=HOTKEY_A,
        delta=2.5,
        balance_after=2.5,
        attempt_id=attempt_id,
        score_id=score_id,
        role="supply",
        details={"composite": 2.5, "scale": 1.0},
    )
    db_session.add(row)
    await db_session.commit()

    loaded = (
        await db_session.execute(select(PointsLedger).where(PointsLedger.id == row.id))
    ).scalar_one()
    assert loaded.hotkey == HOTKEY_A
    assert loaded.delta == pytest.approx(2.5)
    assert loaded.balance_after == pytest.approx(2.5)
    assert loaded.reason == "score_earn"
    assert loaded.attempt_id == attempt_id
    assert loaded.score_id == score_id
    assert loaded.role == "supply"
    assert loaded.created_at is not None
    public = loaded.to_dict()
    assert public["hotkey"] == HOTKEY_A
    assert public["delta"] == pytest.approx(2.5)
    assert public["attempt_id"] == attempt_id
    assert public["score_id"] == score_id
    assert public["details"] == {"composite": 2.5, "scale": 1.0}
    assert "created_at" in public


@pytest.mark.asyncio
async def test_points_balances_rollup_optional_store(db_session: AsyncSession) -> None:
    """VAL-WGT-001: optional points_balances denorm tracks hotkey balance."""

    bal = PointsBalance(hotkey=HOTKEY_A, balance=3.0, updated_at=utc_now())
    db_session.add(bal)
    await db_session.commit()
    loaded = (
        await db_session.execute(select(PointsBalance).where(PointsBalance.hotkey == HOTKEY_A))
    ).scalar_one()
    assert loaded.balance == pytest.approx(3.0)
    assert loaded.to_dict()["hotkey"] == HOTKEY_A
    assert loaded.to_dict()["balance"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Unique earn per attempt_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unique_earn_per_attempt_id(db_session: AsyncSession) -> None:
    """VAL-WGT-001 / prep VAL-WGT-004: second score_earn for same attempt_id raises."""

    attempt_id = str(uuid.uuid4())
    first = _ledger_row(attempt_id=attempt_id, delta=1.0, balance_after=1.0)
    db_session.add(first)
    await db_session.commit()

    second = _ledger_row(attempt_id=attempt_id, delta=1.0, balance_after=2.0)
    db_session.add(second)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()

    rows = (
        (
            await db_session.execute(
                select(PointsLedger).where(PointsLedger.attempt_id == attempt_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].id == first.id


@pytest.mark.asyncio
async def test_admin_adjust_rows_allow_null_attempt_id(db_session: AsyncSession) -> None:
    """Admin adjusts may omit attempt_id without colliding on unique earn key."""

    a = _ledger_row(
        reason="admin_adjust",
        attempt_id=None,
        delta=0.5,
        balance_after=0.5,
        role=None,
    )
    b = _ledger_row(
        reason="admin_adjust",
        attempt_id=None,
        delta=-0.1,
        balance_after=0.4,
        role=None,
    )
    # score_earn paths force attempt; for admin we pass explicit None.
    a.attempt_id = None
    b.attempt_id = None
    db_session.add_all([a, b])
    await db_session.commit()
    count = (
        (
            await db_session.execute(
                select(PointsLedger).where(PointsLedger.reason == "admin_adjust")
            )
        )
        .scalars()
        .all()
    )
    assert len(count) == 2


# ---------------------------------------------------------------------------
# VAL-WGT-020: offer price path independent of price catalog
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_offer_model_keeps_price_per_hour_without_price_catalog(
    db_session: AsyncSession,
) -> None:
    """VAL-WGT-020: offer rows store price_per_hour without gpu_price_catalog table."""

    # Safety: metadata must not register a GPU price catalog this round.
    assert "gpu_price_catalog" not in Base.metadata.tables
    assert "gpu_price_revisions" not in Base.metadata.tables

    from hypercluster.db.models import Provider

    provider = Provider(
        id=str(uuid.uuid4()),
        hotkey=HOTKEY_A,
        display_name="points-regression",
        status="active",
    )
    db_session.add(provider)
    await db_session.flush()

    offer = Offer(
        id=str(uuid.uuid4()),
        provider_id=provider.id,
        node_ids_json="[]",
        mode="single",
        gpu_model="H100",
        gpu_count=1,
        node_count=1,
        require_ib=0,
        tee="none",
        price_per_hour=2.5,
        max_lifetime_hours=24.0,
        status="listed",
    )
    db_session.add(offer)
    await db_session.commit()

    loaded = (await db_session.execute(select(Offer).where(Offer.id == offer.id))).scalar_one()
    assert loaded.price_per_hour == pytest.approx(2.5)
    body = loaded.to_dict()
    assert body["price_per_hour"] == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_offer_create_api_still_works_without_price_catalog(
    settings_factory,
    tmp_path,
) -> None:
    """VAL-WGT-020 regression: signed offer create with price_per_hour succeeds."""

    import json

    from httpx import ASGITransport, AsyncClient

    from hypercluster.api.auth import build_signed_headers
    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    token = "test-challenge-shared-token"
    hotkey = HOTKEY_A

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'offer-no-catalog.sqlite3'}",
        shared_token=token,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        max_offer_price_per_hour=100.0,
        max_offer_lifetime_hours=168.0,
    )
    app = create_app(settings, hyper_settings=hyper)

    def _sign(body: bytes) -> dict[str, str]:
        headers = build_signed_headers(secret=token, hotkey=hotkey, body=body)
        headers["Content-Type"] = "application/json"
        return headers

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Register provider + node
            raw = json.dumps({"display_name": "NoCatalog"}).encode()
            r = await client.post("/v1/providers/register", content=raw, headers=_sign(raw))
            assert r.status_code == 200, r.text

            raw = json.dumps(
                {
                    "gpu_model": "H100",
                    "gpu_count": 2,
                    "ssh_endpoint": "10.0.0.9:22",
                }
            ).encode()
            r = await client.post("/v1/nodes", content=raw, headers=_sign(raw))
            assert r.status_code == 200, r.text
            node_id = r.json()["id"]

            offer_body = {
                "node_ids": [node_id],
                "mode": "single",
                "price_per_hour": 3.75,
                "max_lifetime_hours": 12.0,
                "require_ib": False,
            }
            raw = json.dumps(offer_body).encode()
            r = await client.post("/v1/offers", content=raw, headers=_sign(raw))
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["price_per_hour"] == pytest.approx(3.75)
            assert data["status"] == "listed"

            # points tables present after boot
            db = app.state.database
            async with db.engine.connect() as conn:
                names = await conn.run_sync(
                    lambda sync_conn: set(inspect(sync_conn).get_table_names())
                )
            assert "points_ledger" in names
            assert "gpu_price_catalog" not in names


@pytest.mark.asyncio
async def test_package_exports_points_models() -> None:
    """Models are importable from hypercluster.db.models / package."""

    from hypercluster.db import models as m

    assert hasattr(m, "PointsLedger")
    assert hasattr(m, "PointsBalance")
    assert m.PointsLedger.__tablename__ == "points_ledger"
    assert m.PointsBalance.__tablename__ == "points_balances"


@pytest.mark.asyncio
async def test_direct_sql_insert_earn_row(database: Database) -> None:
    """VAL-WGT-001: raw SQL can insert one earn row after schema create_all."""

    attempt_id = str(uuid.uuid4())
    row_id = str(uuid.uuid4())
    async with database.session() as session:
        await session.execute(
            text(
                """
                INSERT INTO points_ledger (
                    id, hotkey, role, delta, balance_after, reason,
                    score_id, attempt_id, details_json, created_at
                ) VALUES (
                    :id, :hotkey, :role, :delta, :balance_after, :reason,
                    :score_id, :attempt_id, :details_json, :created_at
                )
                """
            ),
            {
                "id": row_id,
                "hotkey": HOTKEY_A,
                "role": "demand",
                "delta": 1.0,
                "balance_after": 1.0,
                "reason": "score_earn",
                "score_id": None,
                "attempt_id": attempt_id,
                "details_json": None,
                "created_at": utc_now().isoformat(),
            },
        )
        await session.commit()
        result = await session.execute(
            text("SELECT hotkey, delta, reason, attempt_id FROM points_ledger WHERE id = :id"),
            {"id": row_id},
        )
        row = result.one()
        assert row[0] == HOTKEY_A
        assert float(row[1]) == pytest.approx(1.0)
        assert row[2] == "score_earn"
        assert row[3] == attempt_id
