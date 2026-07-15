"""VAL-MKT-001..007: provider register/list/heartbeat and node inventory."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.api.auth import build_signed_headers

HOTKEY_A = "5TestMinerHotkeyAAAA"
HOTKEY_B = "5TestMinerHotkeyBBBB"
TOKEN = "test-challenge-shared-token"


def _sign(body: bytes, *, hotkey: str = HOTKEY_A, nonce: str | None = None) -> dict[str, str]:
    return build_signed_headers(secret=TOKEN, hotkey=hotkey, body=body, nonce=nonce)


@pytest.fixture
async def market_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    """App client with insecure HMAC signatures for marketplace tests."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'mkt.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        node_liveness_seconds=120,
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
    display_name: str | None = "GPU Farm",
) -> dict[str, Any]:
    body_obj: dict[str, Any] = {}
    if display_name is not None:
        body_obj["display_name"] = display_name
    raw = json.dumps(body_obj).encode()
    headers = _sign(raw, hotkey=hotkey)
    headers["Content-Type"] = "application/json"
    response = await client.post("/v1/providers/register", content=raw, headers=headers)
    return {"status": response.status_code, "json": response.json()}


async def _heartbeat_provider(client: AsyncClient, *, hotkey: str = HOTKEY_A) -> dict[str, Any]:
    raw = b"{}"
    headers = _sign(raw, hotkey=hotkey)
    headers["Content-Type"] = "application/json"
    response = await client.post("/v1/providers/heartbeat", content=raw, headers=headers)
    return {"status": response.status_code, "json": response.json()}


async def _register_node(
    client: AsyncClient,
    *,
    hotkey: str = HOTKEY_A,
    gpu_model: str = "H100",
    gpu_count: int = 8,
    inventory: dict[str, Any] | None = None,
    ssh_endpoint: str | None = "10.0.0.1:22",
    tee_capability: str = "none",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "tee_capability": tee_capability,
    }
    if ssh_endpoint is not None:
        body["ssh_endpoint"] = ssh_endpoint
    if inventory is not None:
        body["inventory"] = inventory
    raw = json.dumps(body).encode()
    headers = _sign(raw, hotkey=hotkey)
    headers["Content-Type"] = "application/json"
    response = await client.post("/v1/nodes", content=raw, headers=headers)
    return {"status": response.status_code, "json": response.json()}


# ----- VAL-MKT-001..003 providers --------------------------------------------


@pytest.mark.asyncio
async def test_provider_register_creates_active_bound_to_hotkey(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-001: register returns active provider with id + hotkey."""

    result = await _register_provider(market_client, display_name="Alpha")
    assert result["status"] == 200, result
    body = result["json"]
    assert body["id"]
    assert body["hotkey"] == HOTKEY_A
    assert body["status"] == "active"
    assert body.get("display_name") == "Alpha"
    assert body.get("created") is True

    # Idempotent second register: same id, 2xx
    again = await _register_provider(market_client, display_name="Alpha")
    assert again["status"] == 200, again
    assert again["json"]["id"] == body["id"]
    assert again["json"]["hotkey"] == HOTKEY_A
    assert again["json"]["status"] == "active"


@pytest.mark.asyncio
async def test_provider_list_returns_registered(market_client: AsyncClient) -> None:
    """VAL-MKT-002: list includes registered provider; scoped list by hotkey."""

    reg = await _register_provider(market_client)
    provider_id = reg["json"]["id"]

    # Unscoped public list contains the provider
    listing = await market_client.get("/v1/providers")
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert any(p["id"] == provider_id and p["hotkey"] == HOTKEY_A for p in items)

    # Scoped by owner hotkey header
    scoped = await market_client.get("/v1/providers", headers={"X-Hotkey": HOTKEY_A})
    assert scoped.status_code == 200
    scoped_items = scoped.json()["items"]
    assert len(scoped_items) == 1
    assert scoped_items[0]["id"] == provider_id

    # Other hotkey scope is empty
    other = await market_client.get("/v1/providers", headers={"X-Hotkey": HOTKEY_B})
    assert other.status_code == 200
    assert other.json()["items"] == []


@pytest.mark.asyncio
async def test_provider_heartbeat_updates_liveness_no_identity_mutate(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-003: heartbeat advances updated/last_seen without changing id/hotkey."""

    reg = await _register_provider(market_client)
    before = reg["json"]
    provider_id = before["id"]
    updated_before = before.get("updated_at") or before.get("last_seen_at")

    await asyncio.sleep(0.02)
    hb = await _heartbeat_provider(market_client)
    assert hb["status"] == 200, hb
    after = hb["json"]
    assert after["id"] == provider_id
    assert after["hotkey"] == HOTKEY_A
    assert after["status"] == "active"
    updated_after = after.get("updated_at") or after.get("last_seen_at")
    assert updated_after is not None
    assert updated_before is None or updated_after >= updated_before


@pytest.mark.asyncio
async def test_provider_register_rejects_unsigned(market_client: AsyncClient) -> None:
    """Write path must fail closed without signed headers (partial VAL-MKT-022)."""

    response = await market_client.post(
        "/v1/providers/register",
        json={"display_name": "No Auth"},
    )
    assert response.status_code in {401, 403}


# ----- VAL-MKT-004..007 nodes --------------------------------------------------


@pytest.mark.asyncio
async def test_node_register_gpu_count_and_inventory(market_client: AsyncClient) -> None:
    """VAL-MKT-004: node register accepts GPU count and returns inventory fields."""

    await _register_provider(market_client)
    result = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=8,
        ssh_endpoint="10.0.0.5:22",
        inventory={"has_ib": False},
    )
    assert result["status"] == 200, result
    body = result["json"]
    assert body["id"]
    assert body["provider_id"]
    assert body["gpu_count"] == 8
    assert body["gpu_model"] == "H100"
    assert body["status"] in {"healthy", "registered"}


@pytest.mark.asyncio
async def test_node_inventory_ib_capability_flags(market_client: AsyncClient) -> None:
    """VAL-MKT-005: IB-capable inventory sets has_ib / ib_rate_gbps flags."""

    await _register_provider(market_client)

    ib_result = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=8,
        ssh_endpoint="10.0.0.10:22",
        inventory={
            "ib_devices": [{"name": "mlx5_0", "port": 1, "rate_gbps": 400.0, "state": "ACTIVE"}],
            "ib_rate_gbps": 400.0,
        },
    )
    assert ib_result["status"] == 200, ib_result
    ib_node = ib_result["json"]
    assert ib_node["has_ib"] is True
    assert ib_node["ib_rate_gbps"] == 400.0
    assert ib_node["inventory"] is not None

    eth_result = await _register_node(
        market_client,
        gpu_model="A100",
        gpu_count=4,
        ssh_endpoint="10.0.0.11:22",
        inventory={"has_ib": False, "ib_devices": []},
    )
    assert eth_result["status"] == 200, eth_result
    eth_node = eth_result["json"]
    assert eth_node["has_ib"] is False


@pytest.mark.asyncio
async def test_node_heartbeat_refreshes_last_heartbeat(market_client: AsyncClient) -> None:
    """VAL-MKT-006: node heartbeat updates last_heartbeat and keeps healthy."""

    await _register_provider(market_client)
    reg = await _register_node(market_client)
    node_id = reg["json"]["id"]
    before_hb = reg["json"].get("last_heartbeat")

    await asyncio.sleep(0.02)
    raw = json.dumps({"node_id": node_id}).encode()
    headers = _sign(raw)
    headers["Content-Type"] = "application/json"
    response = await market_client.post("/v1/nodes/heartbeat", content=raw, headers=headers)
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert len(items) == 1
    node = items[0]
    assert node["id"] == node_id
    assert node["status"] == "healthy"
    assert node["last_heartbeat"] is not None
    if before_hb is not None:
        assert node["last_heartbeat"] >= before_hb

    # GET shows the posted heartbeat
    got = await market_client.get(f"/v1/nodes/{node_id}")
    assert got.status_code == 200
    assert got.json()["last_heartbeat"] == node["last_heartbeat"]
    assert got.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_node_list_and_get_expose_capabilities(market_client: AsyncClient) -> None:
    """VAL-MKT-007: list/get expose id, GPU model/count, status, IB/tee capabilities."""

    await _register_provider(market_client)
    reg = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=8,
        inventory={
            "ib_devices": [{"name": "mlx5_0", "rate_gbps": 200.0, "state": "ACTIVE"}],
        },
        tee_capability="tdx",
    )
    node_id = reg["json"]["id"]

    listing = await market_client.get("/v1/nodes", headers={"X-Hotkey": HOTKEY_A})
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert len(items) >= 1
    node = next(n for n in items if n["id"] == node_id)
    for key in (
        "id",
        "gpu_model",
        "gpu_count",
        "status",
        "has_ib",
        "tee_capability",
        "provider_id",
    ):
        assert key in node, f"missing capability field {key}"
    assert node["gpu_count"] == 8
    assert node["gpu_model"] == "H100"
    assert node["has_ib"] is True
    assert node["tee_capability"] == "tdx"

    detail = await market_client.get(f"/v1/nodes/{node_id}")
    assert detail.status_code == 200
    d = detail.json()
    assert d["id"] == node_id
    assert d["gpu_count"] == 8
    assert d["has_ib"] is True
    assert d["tee_capability"] == "tdx"


@pytest.mark.asyncio
async def test_node_register_requires_provider_first(market_client: AsyncClient) -> None:
    """Node register without provider is 404 fail-closed."""

    result = await _register_node(market_client)
    assert result["status"] == 404
    assert result["json"]["detail"]["code"] == "provider_not_found"


@pytest.mark.asyncio
async def test_node_register_rejects_non_positive_gpu_count(
    market_client: AsyncClient,
) -> None:
    await _register_provider(market_client)
    raw = json.dumps({"gpu_model": "H100", "gpu_count": 0}).encode()
    headers = _sign(raw)
    headers["Content-Type"] = "application/json"
    response = await market_client.post("/v1/nodes", content=raw, headers=headers)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_tables_created_on_init(settings_factory, tmp_path) -> None:
    """providers / nodes / nonces exist after Database.init()."""

    from sqlalchemy import text

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'schema.sqlite3'}",
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
        assert "providers" in names
        assert "nodes" in names
        assert "nonces" in names
