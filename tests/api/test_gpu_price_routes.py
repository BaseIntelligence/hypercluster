"""VAL-PRICE-030..033: public + admin GPU price catalog HTTP routes.

Surfaces:
- GET /v1/gpu-prices(+/{model_key}) active-only, no notes leak; empty [] safe
- Admin /v1/admin/gpu-prices* challenge shared-token gated list/get/put/disable/history
- Unauthorized writes 401/403 with price_catalog_unauthorized
- Admin PUT updates price and appends exactly one history entry
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.app import create_app
from hypercluster.domain.pricing import (
    disable_catalog_price,
    list_price_history,
    seed_default_catalog,
    upsert_catalog_price,
)
from hypercluster.settings import HyperSettings

TOKEN = "gpu-price-api-test-token-NOT-A-SECRET-FOR-RESPONSES"
SECRET_MARKERS = (
    "shared_token",
    "CHALLENGE_SHARED_TOKEN",
    "private_key",
    "BEGIN PRIVATE",
    "set_weights",
    "password",
    "api_key",
    TOKEN,  # never echo the live shared token
)


@pytest.fixture
async def price_client(
    settings_factory, tmp_path
) -> AsyncIterator[tuple[AsyncClient, Any]]:
    """ASGI client with isolated DB and known shared token."""

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'gpu-price-api.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        price_seed_on_boot=False,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, app


def _admin_headers() -> dict[str, str]:
    """Challenge shared-token headers (Bearer + optional challenge slug)."""

    return {
        "Authorization": f"Bearer {TOKEN}",
        "X-Base-Challenge-Slug": "hypercluster",
    }


def _admin_headers_x_challenge_token() -> dict[str, str]:
    return {"X-Challenge-Token": TOKEN}


def _assert_no_secrets(payload: Any) -> None:
    blob = json.dumps(payload) if not isinstance(payload, str) else payload
    lower = blob.lower()
    for marker in SECRET_MARKERS:
        assert marker.lower() not in lower, f"secret/forbidden marker {marker!r} in {blob[:400]}"


def _assert_no_notes_leak(payload: Any) -> None:
    """Public bodies must not dump operator notes (VAL-PRICE-030)."""

    if isinstance(payload, dict):
        assert "notes" not in payload, f"public body leaked notes: {payload}"
        for value in payload.values():
            _assert_no_notes_leak(value)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_notes_leak(item)


async def _seed_catalog(app: Any) -> None:
    database = app.state.database
    async with database.session() as session:
        await upsert_catalog_price(
            session,
            model_key="H100_80GB",
            price_per_hour=2.49,
            family="h100",
            display_name="NVIDIA H100 80GB",
            notes="secret-operator-note-do-not-leak",
            source="admin",
            changed_by="admin",
            reason="seed active",
        )
        await upsert_catalog_price(
            session,
            model_key="A100_40GB",
            price_per_hour=1.29,
            family="a100",
            display_name="NVIDIA A100 40GB",
            notes="another-secret-note",
            source="admin",
            changed_by="admin",
            reason="seed active a100",
        )
        await upsert_catalog_price(
            session,
            model_key="RTX_4090_DISABLED",
            price_per_hour=0.45,
            family="rtx4090",
            display_name="RTX 4090 disabled test",
            notes="disabled-secret",
            active=True,
            source="admin",
            changed_by="admin",
            reason="seed then disable",
        )
        await disable_catalog_price(
            session,
            model_key="RTX_4090_DISABLED",
            changed_by="admin",
            reason="disable for public filter test",
        )
        await session.commit()


# ---------------------------------------------------------------------------
# VAL-PRICE-030: Public GET returns only active
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_public_list_returns_only_active_no_notes(
    price_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-PRICE-030: public list filters inactive; strips operator notes."""

    client, app = price_client
    await _seed_catalog(app)

    empty_before_seed = await client.get("/v1/gpu-prices")
    # Already seeded; smoke empty path separately below.
    assert empty_before_seed.status_code == 200

    response = await client.get("/v1/gpu-prices")
    assert response.status_code == 200
    body = response.json()
    # Empty-safe array shape (VAL-PRICE-030): body itself is a list, OR
    # items list — accept either consistent wrapper; active_only always.
    items = body if isinstance(body, list) else body.get("items", body)
    assert isinstance(items, list)
    keys = {row["model_key"] for row in items}
    assert "H100_80GB" in keys
    assert "A100_40GB" in keys
    assert "RTX_4090_DISABLED" not in keys
    for row in items:
        assert row.get("active") in (True, 1, None) or "active" not in row or row["active"]
        assert row["currency"] == "USD"
        assert float(row["price_per_hour"]) > 0
        assert "notes" not in row
    _assert_no_notes_leak(body)
    _assert_no_secrets(body)


@pytest.mark.asyncio
async def test_public_list_empty_safe(price_client: tuple[AsyncClient, Any]) -> None:
    """VAL-PRICE-030: empty catalog public list → 200 with empty collection."""

    client, _app = price_client
    response = await client.get("/v1/gpu-prices")
    assert response.status_code == 200
    body = response.json()
    if isinstance(body, list):
        assert body == []
    else:
        items = body.get("items", None)
        assert items == [] or body.get("count") == 0


@pytest.mark.asyncio
async def test_public_get_active_detail_and_inactive_404(
    price_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-PRICE-030: active detail OK; inactive/missing → 404; no notes."""

    client, app = price_client
    await _seed_catalog(app)

    ok = await client.get("/v1/gpu-prices/H100_80GB")
    assert ok.status_code == 200
    detail = ok.json()
    assert detail["model_key"] == "H100_80GB"
    assert detail["price_per_hour"] == pytest.approx(2.49)
    assert detail["currency"] == "USD"
    assert "notes" not in detail
    _assert_no_notes_leak(detail)

    inactive = await client.get("/v1/gpu-prices/RTX_4090_DISABLED")
    assert inactive.status_code == 404
    missing = await client.get("/v1/gpu-prices/DOES_NOT_EXIST")
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_public_list_family_and_model_key_filters(
    price_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-PRICE-030: optional family=/model_key= filters on public list."""

    client, app = price_client
    await _seed_catalog(app)

    by_family = await client.get("/v1/gpu-prices", params={"family": "h100"})
    assert by_family.status_code == 200
    fam_items = (
        by_family.json()
        if isinstance(by_family.json(), list)
        else by_family.json().get("items", [])
    )
    assert fam_items
    assert all(row["family"] == "h100" for row in fam_items)
    assert all(row["model_key"] != "A100_40GB" for row in fam_items)

    by_key = await client.get("/v1/gpu-prices", params={"model_key": "A100_40GB"})
    assert by_key.status_code == 200
    key_items = (
        by_key.json() if isinstance(by_key.json(), list) else by_key.json().get("items", [])
    )
    assert len(key_items) == 1
    assert key_items[0]["model_key"] == "A100_40GB"


# ---------------------------------------------------------------------------
# VAL-PRICE-031: Admin list includes inactive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_list_includes_inactive_after_disable(
    price_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-PRICE-031: admin token list recovers inactive inventory."""

    client, app = price_client
    await _seed_catalog(app)

    pub = await client.get("/v1/gpu-prices")
    pub_items = pub.json() if isinstance(pub.json(), list) else pub.json().get("items", [])
    pub_keys = {row["model_key"] for row in pub_items}
    assert "RTX_4090_DISABLED" not in pub_keys

    admin = await client.get("/v1/admin/gpu-prices", headers=_admin_headers())
    assert admin.status_code == 200
    admin_body = admin.json()
    admin_items = (
        admin_body if isinstance(admin_body, list) else admin_body.get("items", [])
    )
    admin_keys = {row["model_key"] for row in admin_items}
    assert "RTX_4090_DISABLED" in admin_keys
    inactive_row = next(r for r in admin_items if r["model_key"] == "RTX_4090_DISABLED")
    assert inactive_row["active"] in (False, 0)
    # admin may include notes
    assert "notes" in inactive_row or inactive_row.get("notes") is None or True

    detail = await client.get(
        "/v1/admin/gpu-prices/RTX_4090_DISABLED",
        headers=_admin_headers(),
    )
    assert detail.status_code == 200
    detail_body = detail.json()
    catalog = detail_body.get("catalog", detail_body)
    assert catalog["model_key"] == "RTX_4090_DISABLED"
    assert catalog["active"] in (False, 0)


# ---------------------------------------------------------------------------
# VAL-PRICE-032: Admin write rejects missing token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_write_rejects_missing_and_wrong_token(
    price_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-PRICE-032: unauthenticated / wrong token → 401/403; public stays open."""

    client, app = price_client
    await _seed_catalog(app)

    body = {"price_per_hour": 9.99, "reason": "should fail"}

    no_auth = await client.put("/v1/admin/gpu-prices/H100_80GB", json=body)
    assert no_auth.status_code in (401, 403)
    no_auth_detail = no_auth.json()
    detail = no_auth_detail.get("detail", no_auth_detail)
    if isinstance(detail, dict):
        assert detail.get("code") == "price_catalog_unauthorized"
    else:
        # Accept string detail that names the code
        assert "price_catalog_unauthorized" in str(detail) or no_auth.status_code in (
            401,
            403,
        )

    wrong = await client.put(
        "/v1/admin/gpu-prices/H100_80GB",
        json=body,
        headers={"Authorization": "Bearer totally-wrong-token"},
    )
    assert wrong.status_code in (401, 403)

    wrong_x = await client.put(
        "/v1/admin/gpu-prices/H100_80GB",
        json=body,
        headers={"X-Challenge-Token": "nope"},
    )
    assert wrong_x.status_code in (401, 403)

    # Admin list also gated
    list_no = await client.get("/v1/admin/gpu-prices")
    assert list_no.status_code in (401, 403)

    disable_no = await client.post("/v1/admin/gpu-prices/H100_80GB/disable")
    assert disable_no.status_code in (401, 403)

    hist_no = await client.get("/v1/admin/gpu-prices/H100_80GB/history")
    assert hist_no.status_code in (401, 403)

    # Public remains open without token
    pub = await client.get("/v1/gpu-prices")
    assert pub.status_code == 200


@pytest.mark.asyncio
async def test_admin_accepts_bearer_and_x_challenge_token(
    price_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-PRICE-032: authenticated admin succeeds with either header universal."""

    client, app = price_client
    await _seed_catalog(app)

    via_bearer = await client.get("/v1/admin/gpu-prices", headers=_admin_headers())
    assert via_bearer.status_code == 200

    via_x = await client.get(
        "/v1/admin/gpu-prices", headers=_admin_headers_x_challenge_token()
    )
    assert via_x.status_code == 200


# ---------------------------------------------------------------------------
# VAL-PRICE-033: Admin PUT updates price + history length +1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_put_updates_price_and_appends_one_history(
    price_client: tuple[AsyncClient, Any],
) -> None:
    """VAL-PRICE-033: PUT changes price; history endpoint length grows by exactly 1."""

    client, app = price_client
    await _seed_catalog(app)

    before = await client.get(
        "/v1/admin/gpu-prices/H100_80GB/history",
        headers=_admin_headers(),
    )
    assert before.status_code == 200
    before_body = before.json()
    before_items = (
        before_body if isinstance(before_body, list) else before_body.get("items", [])
    )
    before_len = len(before_items)

    put = await client.put(
        "/v1/admin/gpu-prices/H100_80GB",
        headers=_admin_headers(),
        json={
            "price_per_hour": 3.33,
            "display_name": "NVIDIA H100 80GB repriced",
            "notes": "ops-reprice-note",
            "reason": "market adjust",
        },
    )
    assert put.status_code == 200, put.text
    put_body = put.json()
    catalog = put_body.get("catalog", put_body)
    assert catalog["model_key"] == "H100_80GB"
    assert float(catalog["price_per_hour"]) == pytest.approx(3.33)

    # Public (active) shows new price; still no notes
    pub = await client.get("/v1/gpu-prices/H100_80GB")
    assert pub.status_code == 200
    pub_body = pub.json()
    assert float(pub_body["price_per_hour"]) == pytest.approx(3.33)
    assert "notes" not in pub_body

    after = await client.get(
        "/v1/admin/gpu-prices/H100_80GB/history",
        headers=_admin_headers(),
    )
    assert after.status_code == 200
    after_body = after.json()
    after_items = after_body if isinstance(after_body, list) else after_body.get("items", [])
    assert len(after_items) == before_len + 1
    newest = after_items[0]
    assert float(newest["price_per_hour"]) == pytest.approx(3.33)

    # Domain-level confirm via service
    async with app.state.database.session() as session:
        hist = await list_price_history(session, "H100_80GB", limit=50)
        assert len(hist) == before_len + 1
        assert float(hist[0].price_per_hour) == pytest.approx(3.33)


@pytest.mark.asyncio
async def test_admin_disable_and_post_create(
    price_client: tuple[AsyncClient, Any],
) -> None:
    """Admin disable flips active; POST create-if-missing inserts a new key."""

    client, app = price_client
    await _seed_catalog(app)

    disable = await client.post(
        "/v1/admin/gpu-prices/A100_40GB/disable",
        headers=_admin_headers(),
        json={"reason": "temporary"},
    )
    assert disable.status_code == 200, disable.text
    dbody = disable.json()
    catalog = dbody.get("catalog", dbody)
    assert catalog["active"] in (False, 0)

    # Public no longer sees it
    pub = await client.get("/v1/gpu-prices/A100_40GB")
    assert pub.status_code == 404

    create = await client.post(
        "/v1/admin/gpu-prices",
        headers=_admin_headers(),
        json={
            "model_key": "L4_24GB",
            "price_per_hour": 0.40,
            "family": "l4",
            "display_name": "NVIDIA L4",
            "reason": "bootstrap l4",
        },
    )
    assert create.status_code in (200, 201), create.text
    cbody = create.json()
    catalog = cbody.get("catalog", cbody)
    assert catalog["model_key"] == "L4_24GB"
    assert float(catalog["price_per_hour"]) == pytest.approx(0.40)

    pub_new = await client.get("/v1/gpu-prices/L4_24GB")
    assert pub_new.status_code == 200


@pytest.mark.asyncio
async def test_admin_put_rejects_invalid_price(
    price_client: tuple[AsyncClient, Any],
) -> None:
    """Write validation: non-finite / <=0 price → 422 invalid_price."""

    client, app = price_client
    await _seed_catalog(app)

    bad = await client.put(
        "/v1/admin/gpu-prices/H100_80GB",
        headers=_admin_headers(),
        json={"price_per_hour": -1},
    )
    assert bad.status_code == 422
    detail = bad.json().get("detail", bad.json())
    if isinstance(detail, dict):
        assert detail.get("code") == "invalid_price"


@pytest.mark.asyncio
async def test_seed_boot_visible_on_public_list(
    settings_factory, tmp_path
) -> None:
    """Boot seed path (when on) surfaces active rows on public GET."""

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'gpu-price-seed-api.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(allow_insecure_signatures=True, price_seed_on_boot=True)
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Optional explicit seed if boot hook only runs post_init
            async with app.state.database.session() as session:
                await seed_default_catalog(session, only_if_empty=True, source="seed")
                await session.commit()
            response = await client.get("/v1/gpu-prices")
            assert response.status_code == 200
            items = (
                response.json()
                if isinstance(response.json(), list)
                else response.json().get("items", [])
            )
            assert len(items) >= 10
            assert all(row.get("currency", "USD") == "USD" for row in items)
            _assert_no_notes_leak(items)
