"""VAL-FAB-010 API-level: require_ib rent re-checks fabric after IB strip."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.api.auth import build_signed_headers
from hypercluster.fabric.discovery import build_fabric_report
from hypercluster.fabric.gates import evaluate_require_ib_nodes

PROVIDER_HK = "provider-hotkey-fab010aaaaaaaaaaaaaaaaaaaaaa"
RENTER_HK = "renter-hotkey-fab010bbbbbbbbbbbbbbbbbbbbbb"
TOKEN = "test-challenge-shared-token"


def _sign(body: bytes, *, hotkey: str) -> dict[str, str]:
    return build_signed_headers(secret=TOKEN, hotkey=hotkey, body=body)


@pytest.fixture
async def market_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'fab010.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        max_offer_price_per_hour=100.0,
        max_offer_lifetime_hours=168.0,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # expose app for domain session access in strip test
            client._hyper_app = app  # type: ignore[attr-defined]
            yield client


async def _register_provider(client: AsyncClient) -> dict[str, Any]:
    raw = json.dumps({"display_name": "FabProvider"}).encode()
    headers = _sign(raw, hotkey=PROVIDER_HK)
    headers["Content-Type"] = "application/json"
    r = await client.post("/v1/providers/register", content=raw, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


async def _register_node(
    client: AsyncClient,
    *,
    inventory: dict[str, Any],
    gpu_count: int = 2,
) -> dict[str, Any]:
    body = {
        "gpu_model": "H100",
        "gpu_count": gpu_count,
        "inventory": inventory,
        "hostname": "fab-host",
    }
    raw = json.dumps(body).encode()
    headers = _sign(raw, hotkey=PROVIDER_HK)
    headers["Content-Type"] = "application/json"
    r = await client.post("/v1/nodes", content=raw, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.asyncio
async def test_require_ib_rent_ok_with_ib_nodes(market_client: AsyncClient) -> None:
    """VAL-FAB-010: healthy IB inventory rent succeeds for require_ib offer."""

    await _register_provider(market_client)
    node = await _register_node(
        market_client,
        inventory={
            "has_ib": True,
            "ib_rate_gbps": 200,
            "ib_devices": [
                {"name": "mlx5_0", "port": 1, "rate_gbps": 200.0, "state": "Active"}
            ],
        },
    )
    node_id = node["id"]
    assert node["has_ib"] is True

    offer_body = {
        "node_ids": [node_id],
        "price_per_hour": 1.5,
        "max_lifetime_hours": 4,
        "mode": "single",
        "require_ib": True,
    }
    raw = json.dumps(offer_body).encode()
    offer_resp = await market_client.post(
        "/v1/offers",
        content=raw,
        headers={**_sign(raw, hotkey=PROVIDER_HK), "Content-Type": "application/json"},
    )
    assert offer_resp.status_code == 200, offer_resp.text
    offer_id = offer_resp.json()["id"]

    rent_body = json.dumps({"lifetime_hours": 1}).encode()
    rent = await market_client.post(
        f"/v1/offers/{offer_id}/rent",
        content=rent_body,
        headers={**_sign(rent_body, hotkey=RENTER_HK), "Content-Type": "application/json"},
    )
    assert rent.status_code == 200, rent.text
    body = rent.json()
    lease = body.get("lease") or body
    assert lease.get("status") in {"active", "requested"}


@pytest.mark.asyncio
async def test_require_ib_rent_rejected_after_ib_stripped(market_client: AsyncClient) -> None:
    """VAL-FAB-010: stripped IB re-report prevents new rents for require_ib offers."""

    await _register_provider(market_client)
    node = await _register_node(
        market_client,
        inventory={
            "has_ib": True,
            "ib_rate_gbps": 200.0,
            "ib_devices": [
                {"name": "mlx5_0", "port": 1, "rate_gbps": 200.0, "state": "Active"}
            ],
        },
    )
    node_id = node["id"]

    offer_body = {
        "node_ids": [node_id],
        "price_per_hour": 2.0,
        "max_lifetime_hours": 4,
        "mode": "single",
        "require_ib": True,
    }
    raw = json.dumps(offer_body).encode()
    offer_resp = await market_client.post(
        "/v1/offers",
        content=raw,
        headers={**_sign(raw, hotkey=PROVIDER_HK), "Content-Type": "application/json"},
    )
    assert offer_resp.status_code == 200, offer_resp.text
    offer_id = offer_resp.json()["id"]

    # Strip IB via fabric-report + denormalized inventory (fabric-scan inject style).
    eth_report = build_fabric_report(
        node_id=node_id,
        ib_devices=[],
        eth_ifaces=["eth0"],
        gpu_count=2,
        source="inject",
    )
    app = market_client._hyper_app  # type: ignore[attr-defined]
    db = app.state.database
    from hypercluster.domain.fabric_reports import persist_fabric_report
    from hypercluster.domain.nodes import get_node

    async with db.session() as session:
        n = await get_node(session, node_id)
        assert n is not None
        n.has_ib = 0
        n.ib_rate_gbps = None
        n.inventory_json = json.dumps({"has_ib": False, "ib_devices": []})
        await persist_fabric_report(session, eth_report, node=n, update_node_inventory=True)

    rent_body = json.dumps({"lifetime_hours": 1}).encode()
    rent = await market_client.post(
        f"/v1/offers/{offer_id}/rent",
        content=rent_body,
        headers={**_sign(rent_body, hotkey=RENTER_HK), "Content-Type": "application/json"},
    )
    assert rent.status_code in {400, 409, 422}, rent.text
    text = rent.text.lower()
    assert "ib" in text or "require_ib" in text or rent.status_code == 409


def test_require_ib_check_function_docs_rent_policy() -> None:
    """Unit mirror of VAL-FAB-010 for pure function."""

    ib = build_fabric_report(
        node_id="n1",
        ib_devices=[
            {"name": "mlx5_0", "rate_gbps": 200.0, "port": 1, "state": "Active"}
        ],
        gpu_count=1,
    )
    eth = build_fabric_report(node_id="n1", ib_devices=[], gpu_count=1)
    assert evaluate_require_ib_nodes(
        require_ib=True, reports=[ib], node_ids=["n1"]
    ).may_rent
    assert not evaluate_require_ib_nodes(
        require_ib=True, reports=[eth], node_ids=["n1"]
    ).may_rent
