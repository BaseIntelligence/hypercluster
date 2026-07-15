"""VAL-MKT-008..012, VAL-MKT-025..029: offer create/withdraw + list filters."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.api.auth import build_signed_headers

HOTKEY_A = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
HOTKEY_B = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
TOKEN = "test-challenge-shared-token"


def _sign(body: bytes, *, hotkey: str = HOTKEY_A, nonce: str | None = None) -> dict[str, str]:
    return build_signed_headers(secret=TOKEN, hotkey=hotkey, body=body, nonce=nonce)


@pytest.fixture
async def market_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    """App client with marketplace caps + insecure signature mode."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'offers.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        node_liveness_seconds=120,
        max_offer_price_per_hour=100.0,
        max_offer_lifetime_hours=168.0,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


async def _register_provider(
    client: AsyncClient,
    *,
    hotkey: str = HOTKEY_A,
    display_name: str = "Farm",
) -> dict[str, Any]:
    raw = json.dumps({"display_name": display_name}).encode()
    headers = _sign(raw, hotkey=hotkey)
    headers["Content-Type"] = "application/json"
    response = await client.post("/v1/providers/register", content=raw, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


async def _register_node(
    client: AsyncClient,
    *,
    hotkey: str = HOTKEY_A,
    gpu_model: str = "H100",
    gpu_count: int = 8,
    inventory: dict[str, Any] | None = None,
    tee_capability: str = "none",
    ssh_endpoint: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "tee_capability": tee_capability,
        "ssh_endpoint": ssh_endpoint
        or f"10.0.0.{abs(hash(gpu_model + tee_capability)) % 200 + 1}:22",
    }
    if inventory is not None:
        body["inventory"] = inventory
    raw = json.dumps(body).encode()
    headers = _sign(raw, hotkey=hotkey)
    headers["Content-Type"] = "application/json"
    response = await client.post("/v1/nodes", content=raw, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


async def _create_offer(
    client: AsyncClient,
    *,
    hotkey: str = HOTKEY_A,
    node_ids: list[str],
    price_per_hour: float | None = 2.5,
    max_lifetime_hours: float | None = 24.0,
    mode: str = "single",
    require_ib: bool = False,
    tee: str | None = None,
    gpu_model: str | None = None,
    gpu_count: int | None = None,
    extra: dict[str, Any] | None = None,
    omit_keys: set[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "node_ids": node_ids,
        "mode": mode,
        "require_ib": require_ib,
    }
    if price_per_hour is not None:
        body["price_per_hour"] = price_per_hour
    if max_lifetime_hours is not None:
        body["max_lifetime_hours"] = max_lifetime_hours
    if tee is not None:
        body["tee"] = tee
    if gpu_model is not None:
        body["gpu_model"] = gpu_model
    if gpu_count is not None:
        body["gpu_count"] = gpu_count
    if extra:
        body.update(extra)
    if omit_keys:
        for key in omit_keys:
            body.pop(key, None)
    raw = json.dumps(body).encode()
    headers = _sign(raw, hotkey=hotkey)
    headers["Content-Type"] = "application/json"
    response = await client.post("/v1/offers", content=raw, headers=headers)
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = {"raw": response.text}
    return {"status": response.status_code, "json": payload}


async def _withdraw_offer(
    client: AsyncClient,
    offer_id: str,
    *,
    hotkey: str = HOTKEY_A,
) -> dict[str, Any]:
    # Empty body for signature binding; httpx delete() has no content kw —
    # use generic request so auth covers the same empty payload stream.
    raw = b""
    headers = _sign(raw, hotkey=hotkey)
    response = await client.request(
        "DELETE",
        f"/v1/offers/{offer_id}",
        content=raw,
        headers=headers,
    )
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = {"raw": response.text}
    return {"status": response.status_code, "json": payload}


# ----- fixtures helpers -------------------------------------------------------


async def _ready_provider_and_node(
    client: AsyncClient,
    *,
    gpu_model: str = "H100",
    gpu_count: int = 8,
    inventory: dict[str, Any] | None = None,
    tee_capability: str = "none",
    hotkey: str = HOTKEY_A,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provider = await _register_provider(client, hotkey=hotkey)
    node = await _register_node(
        client,
        hotkey=hotkey,
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        inventory=inventory if inventory is not None else {"has_ib": False},
        tee_capability=tee_capability,
    )
    return provider, node


# ----- VAL-MKT-008 create success --------------------------------------------


@pytest.mark.asyncio
async def test_offer_create_valid_price_and_lifetime(market_client: AsyncClient) -> None:
    """VAL-MKT-008: valid price/lifetime creates listed offer."""

    _provider, node = await _ready_provider_and_node(market_client)
    result = await _create_offer(
        market_client,
        node_ids=[node["id"]],
        price_per_hour=3.5,
        max_lifetime_hours=12.0,
        mode="single",
    )
    assert result["status"] == 200, result
    offer = result["json"]
    assert offer["id"]
    assert offer["status"] == "listed"
    assert offer["price_per_hour"] == 3.5
    assert offer["max_lifetime_hours"] == 12.0
    assert offer["gpu_model"] == "H100"
    assert offer["gpu_count"] == 8
    assert offer["mode"] == "single"
    assert node["id"] in offer["node_ids"]
    assert offer["provider_id"] == node["provider_id"]
    assert "require_ib" in offer
    assert "tee" in offer


# ----- VAL-MKT-009 missing / non-positive ------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"price_per_hour": 0, "max_lifetime_hours": 10},
        {"price_per_hour": -1, "max_lifetime_hours": 10},
        {"price_per_hour": 2.0, "max_lifetime_hours": 0},
        {"price_per_hour": 2.0, "max_lifetime_hours": -5},
        {"omit_keys": {"price_per_hour"}, "max_lifetime_hours": 10},
        {"omit_keys": {"max_lifetime_hours"}, "price_per_hour": 2.0},
    ],
)
async def test_offer_create_rejects_missing_or_non_positive(
    market_client: AsyncClient,
    kwargs: dict[str, Any],
) -> None:
    """VAL-MKT-009: missing/non-positive price or lifetime → 4xx, no listed offer."""

    _, node = await _ready_provider_and_node(market_client)
    before = await market_client.get("/v1/offers")
    assert before.status_code == 200
    counted = len(before.json()["items"])

    create_kwargs: dict[str, Any] = {
        "node_ids": [node["id"]],
        **kwargs,
    }
    result = await _create_offer(market_client, **create_kwargs)
    assert 400 <= result["status"] < 500, result

    after = await market_client.get("/v1/offers")
    assert after.status_code == 200
    assert len(after.json()["items"]) == counted


# ----- VAL-MKT-010 price caps -------------------------------------------------


@pytest.mark.asyncio
async def test_offer_create_rejects_price_over_system_cap(market_client: AsyncClient) -> None:
    """VAL-MKT-010: price above HYPER_MAX_OFFER_PRICE_PER_HOUR is rejected."""

    _, node = await _ready_provider_and_node(market_client)
    # Cap configured to 100.0 in fixture
    result = await _create_offer(
        market_client,
        node_ids=[node["id"]],
        price_per_hour=100.01,
        max_lifetime_hours=10.0,
    )
    assert 400 <= result["status"] < 500, result
    detail = result["json"].get("detail") or result["json"]
    code = detail.get("code") if isinstance(detail, dict) else None
    assert code in {"price_over_cap", "validation_error", "price_cap_exceeded", "invalid_price"}

    # Under-cap still works (regression of 008)
    ok = await _create_offer(
        market_client,
        node_ids=[node["id"]],
        price_per_hour=99.99,
        max_lifetime_hours=10.0,
    )
    assert ok["status"] == 200, ok
    assert ok["json"]["status"] == "listed"


# ----- VAL-MKT-011 lifetime caps ---------------------------------------------


@pytest.mark.asyncio
async def test_offer_create_rejects_lifetime_over_system_cap(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-011: lifetime above HYPER_MAX_OFFER_LIFETIME_HOURS is rejected."""

    _, node = await _ready_provider_and_node(market_client)
    # Cap configured to 168.0 in fixture
    result = await _create_offer(
        market_client,
        node_ids=[node["id"]],
        price_per_hour=5.0,
        max_lifetime_hours=169.0,
    )
    assert 400 <= result["status"] < 500, result
    detail = result["json"].get("detail") or result["json"]
    code = detail.get("code") if isinstance(detail, dict) else None
    assert code in {
        "lifetime_over_cap",
        "validation_error",
        "lifetime_cap_exceeded",
        "invalid_lifetime",
    }


# ----- VAL-MKT-012 withdraw --------------------------------------------------


@pytest.mark.asyncio
async def test_offer_withdraw_removes_from_browse(market_client: AsyncClient) -> None:
    """VAL-MKT-012: withdraw transitions to withdrawn and drops from default browse."""

    _, node = await _ready_provider_and_node(market_client)
    created = await _create_offer(
        market_client,
        node_ids=[node["id"]],
        price_per_hour=2.0,
        max_lifetime_hours=8.0,
    )
    assert created["status"] == 200, created
    offer_id = created["json"]["id"]

    listed = await market_client.get("/v1/offers")
    assert listed.status_code == 200
    assert any(o["id"] == offer_id for o in listed.json()["items"])

    withdrawn = await _withdraw_offer(market_client, offer_id)
    assert withdrawn["status"] == 200, withdrawn
    assert withdrawn["json"]["id"] == offer_id
    assert withdrawn["json"]["status"] == "withdrawn"

    browse = await market_client.get("/v1/offers")
    assert browse.status_code == 200
    assert all(o["id"] != offer_id for o in browse.json()["items"])

    # Explicit status filter may still surface withdrawn if requested.
    by_status = await market_client.get("/v1/offers", params={"status": "withdrawn"})
    assert by_status.status_code == 200
    items = by_status.json()["items"]
    assert any(o["id"] == offer_id and o["status"] == "withdrawn" for o in items)


@pytest.mark.asyncio
async def test_offer_withdraw_rejects_foreign_hotkey(market_client: AsyncClient) -> None:
    """Owner-only withdraw (fail-closed for foreign hotkeys)."""

    _, node = await _ready_provider_and_node(market_client, hotkey=HOTKEY_A)
    created = await _create_offer(
        market_client,
        hotkey=HOTKEY_A,
        node_ids=[node["id"]],
        price_per_hour=1.5,
        max_lifetime_hours=4.0,
    )
    offer_id = created["json"]["id"]

    await _register_provider(market_client, hotkey=HOTKEY_B, display_name="Other")
    foreign = await _withdraw_offer(market_client, offer_id, hotkey=HOTKEY_B)
    assert foreign["status"] in {403, 404}, foreign

    still = await market_client.get("/v1/offers")
    assert any(o["id"] == offer_id for o in still.json()["items"])


# ----- VAL-MKT-025..029 list / filters ---------------------------------------


@pytest.mark.asyncio
async def test_list_offers_capability_fields(market_client: AsyncClient) -> None:
    """VAL-MKT-025: list returns listed offers with required capability fields."""

    _, node = await _ready_provider_and_node(
        market_client,
        inventory={
            "ib_devices": [{"name": "mlx5_0", "rate_gbps": 200.0, "state": "ACTIVE"}],
        },
        tee_capability="tdx",
    )
    created = await _create_offer(
        market_client,
        node_ids=[node["id"]],
        price_per_hour=4.0,
        max_lifetime_hours=6.0,
        require_ib=True,
        tee="tdx",
    )
    assert created["status"] == 200, created

    response = await market_client.get("/v1/offers")
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) >= 1
    offer = next(o for o in items if o["id"] == created["json"]["id"])
    for key in (
        "id",
        "gpu_model",
        "gpu_count",
        "price_per_hour",
        "max_lifetime_hours",
        "require_ib",
        "tee",
        "mode",
        "status",
    ):
        assert key in offer, f"missing field {key}"
    assert offer["status"] == "listed"
    assert offer["require_ib"] is True
    assert offer["tee"] == "tdx"


@pytest.mark.asyncio
async def test_filter_offers_by_gpu_model(market_client: AsyncClient) -> None:
    """VAL-MKT-026: gpu_model filter returns only matching offers."""

    await _register_provider(market_client)
    h100 = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=8,
        ssh_endpoint="10.0.1.1:22",
        inventory={"has_ib": False},
    )
    a100 = await _register_node(
        market_client,
        gpu_model="A100",
        gpu_count=4,
        ssh_endpoint="10.0.1.2:22",
        inventory={"has_ib": False},
    )
    o1 = await _create_offer(market_client, node_ids=[h100["id"]], gpu_model="H100", gpu_count=8)
    o2 = await _create_offer(market_client, node_ids=[a100["id"]], gpu_model="A100", gpu_count=4)
    assert o1["status"] == 200 and o2["status"] == 200

    filtered = await market_client.get("/v1/offers", params={"gpu_model": "H100"})
    assert filtered.status_code == 200
    models = {o["gpu_model"] for o in filtered.json()["items"]}
    assert models == {"H100"}
    ids = {o["id"] for o in filtered.json()["items"]}
    assert o1["json"]["id"] in ids
    assert o2["json"]["id"] not in ids


@pytest.mark.asyncio
async def test_filter_offers_by_require_ib(market_client: AsyncClient) -> None:
    """VAL-MKT-027: require_ib=true returns only IB offers."""

    await _register_provider(market_client)
    ib_node = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=8,
        ssh_endpoint="10.0.2.1:22",
        inventory={
            "ib_devices": [{"name": "mlx5_0", "rate_gbps": 400.0, "state": "ACTIVE"}],
        },
    )
    eth_node = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=4,
        ssh_endpoint="10.0.2.2:22",
        inventory={"has_ib": False},
    )
    ib_offer = await _create_offer(
        market_client,
        node_ids=[ib_node["id"]],
        require_ib=True,
        price_per_hour=5.0,
        max_lifetime_hours=12.0,
    )
    eth_offer = await _create_offer(
        market_client,
        node_ids=[eth_node["id"]],
        require_ib=False,
        price_per_hour=2.0,
        max_lifetime_hours=12.0,
    )
    assert ib_offer["status"] == 200, ib_offer
    assert eth_offer["status"] == 200, eth_offer

    filtered = await market_client.get("/v1/offers", params={"require_ib": "true"})
    assert filtered.status_code == 200
    items = filtered.json()["items"]
    assert all(o["require_ib"] is True for o in items)
    ids = {o["id"] for o in items}
    assert ib_offer["json"]["id"] in ids
    assert eth_offer["json"]["id"] not in ids


@pytest.mark.asyncio
async def test_filter_offers_by_tee(market_client: AsyncClient) -> None:
    """VAL-MKT-028: tee filter returns only matching tee tier offers."""

    await _register_provider(market_client)
    tdx_node = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=8,
        ssh_endpoint="10.0.3.1:22",
        tee_capability="tdx",
        inventory={"has_ib": False},
    )
    plain_node = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=4,
        ssh_endpoint="10.0.3.2:22",
        tee_capability="none",
        inventory={"has_ib": False},
    )
    tdx_offer = await _create_offer(
        market_client,
        node_ids=[tdx_node["id"]],
        tee="tdx",
        price_per_hour=6.0,
        max_lifetime_hours=10.0,
    )
    plain_offer = await _create_offer(
        market_client,
        node_ids=[plain_node["id"]],
        tee="none",
        price_per_hour=3.0,
        max_lifetime_hours=10.0,
    )
    assert tdx_offer["status"] == 200 and plain_offer["status"] == 200

    filtered = await market_client.get("/v1/offers", params={"tee": "tdx"})
    assert filtered.status_code == 200
    items = filtered.json()["items"]
    assert all(o["tee"] == "tdx" for o in items)
    ids = {o["id"] for o in items}
    assert tdx_offer["json"]["id"] in ids
    assert plain_offer["json"]["id"] not in ids


@pytest.mark.asyncio
async def test_capability_filters_compose_with_listed_status(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-029: default browse is listed-only; filters do not resurrect withdrawn."""

    await _register_provider(market_client)
    node_a = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=8,
        ssh_endpoint="10.0.4.1:22",
        inventory={
            "ib_devices": [{"name": "mlx5_0", "rate_gbps": 200.0, "state": "ACTIVE"}],
        },
    )
    node_b = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=8,
        ssh_endpoint="10.0.4.2:22",
        inventory={
            "ib_devices": [{"name": "mlx5_0", "rate_gbps": 200.0, "state": "ACTIVE"}],
        },
    )
    keep = await _create_offer(
        market_client,
        node_ids=[node_a["id"]],
        require_ib=True,
        price_per_hour=4.0,
        max_lifetime_hours=8.0,
    )
    drop = await _create_offer(
        market_client,
        node_ids=[node_b["id"]],
        require_ib=True,
        price_per_hour=4.0,
        max_lifetime_hours=8.0,
    )
    assert keep["status"] == 200 and drop["status"] == 200
    await _withdraw_offer(market_client, drop["json"]["id"])

    # Default browse: listed-only, composed with require_ib
    combined = await market_client.get(
        "/v1/offers",
        params={"require_ib": "true", "gpu_model": "H100"},
    )
    assert combined.status_code == 200
    items = combined.json()["items"]
    ids = {o["id"] for o in items}
    assert keep["json"]["id"] in ids
    assert drop["json"]["id"] not in ids
    assert all(o["status"] == "listed" for o in items)
    assert all(o["require_ib"] is True for o in items)
    assert all(o["gpu_model"] == "H100" for o in items)


@pytest.mark.asyncio
async def test_require_ib_offer_rejects_non_ib_nodes(market_client: AsyncClient) -> None:
    """Offer create with require_ib against non-IB inventory fails closed (VAL-MKT-005 link)."""

    _, node = await _ready_provider_and_node(market_client, inventory={"has_ib": False})
    result = await _create_offer(
        market_client,
        node_ids=[node["id"]],
        require_ib=True,
        price_per_hour=2.0,
        max_lifetime_hours=5.0,
    )
    assert 400 <= result["status"] < 500, result


@pytest.mark.asyncio
async def test_offer_create_rejects_unsigned(market_client: AsyncClient) -> None:
    """Write path fail-closed without signature (partial VAL-MKT-022)."""

    _, node = await _ready_provider_and_node(market_client)
    response = await market_client.post(
        "/v1/offers",
        json={
            "node_ids": [node["id"]],
            "price_per_hour": 1.0,
            "max_lifetime_hours": 2.0,
            "mode": "single",
        },
    )
    assert response.status_code in {401, 403}


@pytest.mark.asyncio
async def test_list_offers_unsigned_read_ok(market_client: AsyncClient) -> None:
    """VAL-MKT-024 partial: GET /v1/offers browse works without write signatures."""

    response = await market_client.get("/v1/offers")
    assert response.status_code == 200
    assert "items" in response.json()


@pytest.mark.asyncio
async def test_offers_table_created_on_init(settings_factory, tmp_path) -> None:
    from sqlalchemy import text

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'offers-schema.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    app = create_app(settings, hyper_settings=HyperSettings())
    async with app.router.lifespan_context(app):
        db = app.state.database
        async with db.engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            )
            names = {r[0] for r in rows.fetchall()}
    assert "offers" in names
