"""VAL-MKT-013..021, VAL-MKT-031: rent / lease / pod lifecycle."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.api.auth import build_signed_headers

# Distinct non-secret tester hotkeys (ss58-shaped lengths not required in insecure mode).
PROVIDER_HK = "provider-hotkey-aaaaaaaaaaaaaaaaaaaaaaaaaaaa"
RENTER_HK = "renter-hotkey-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
RENTER2_HK = "renter2-hotkey-cccccccccccccccccccccccccccccc"
TOKEN = "test-challenge-shared-token"


def _sign(body: bytes, *, hotkey: str, nonce: str | None = None) -> dict[str, str]:
    return build_signed_headers(secret=TOKEN, hotkey=hotkey, body=body, nonce=nonce)


@pytest.fixture
async def market_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    """App client with marketplace caps + insecure signature mode + short node liveness."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'leases.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        node_liveness_seconds=30,
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
    hotkey: str = PROVIDER_HK,
    display_name: str = "Provider",
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
    hotkey: str = PROVIDER_HK,
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
        or f"10.20.0.{abs(hash(gpu_model + hotkey + (ssh_endpoint or ''))) % 200 + 1}:22",
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
    hotkey: str = PROVIDER_HK,
    node_ids: list[str],
    price_per_hour: float = 2.5,
    max_lifetime_hours: float = 24.0,
    mode: str = "single",
    require_ib: bool = False,
    tee: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "node_ids": node_ids,
        "price_per_hour": price_per_hour,
        "max_lifetime_hours": max_lifetime_hours,
        "mode": mode,
        "require_ib": require_ib,
    }
    if tee is not None:
        body["tee"] = tee
    raw = json.dumps(body).encode()
    headers = _sign(raw, hotkey=hotkey)
    headers["Content-Type"] = "application/json"
    response = await client.post("/v1/offers", content=raw, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


async def _rent_offer(
    client: AsyncClient,
    offer_id: str,
    *,
    hotkey: str = RENTER_HK,
    lifetime_hours: float | None = None,
    max_price: float | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if lifetime_hours is not None:
        body["lifetime_hours"] = lifetime_hours
    if max_price is not None:
        body["max_price"] = max_price
    raw = json.dumps(body).encode()
    headers = _sign(raw, hotkey=hotkey)
    headers["Content-Type"] = "application/json"
    response = await client.post(
        f"/v1/offers/{offer_id}/rent",
        content=raw,
        headers=headers,
    )
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = {"raw": response.text}
    return {"status": response.status_code, "json": payload}


async def _terminate_lease(
    client: AsyncClient,
    lease_id: str,
    *,
    hotkey: str = RENTER_HK,
    reason: str | None = "renter_cancel",
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if reason is not None:
        body["reason"] = reason
    raw = json.dumps(body).encode()
    headers = _sign(raw, hotkey=hotkey)
    headers["Content-Type"] = "application/json"
    response = await client.post(
        f"/v1/leases/{lease_id}/terminate",
        content=raw,
        headers=headers,
    )
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = {"raw": response.text}
    return {"status": response.status_code, "json": payload}


async def _withdraw_offer(
    client: AsyncClient,
    offer_id: str,
    *,
    hotkey: str = PROVIDER_HK,
) -> dict[str, Any]:
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


async def _ready_listed_single(
    client: AsyncClient,
    *,
    price_per_hour: float = 2.5,
    max_lifetime_hours: float = 24.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    provider = await _register_provider(client)
    node = await _register_node(client, inventory={"has_ib": False})
    offer = await _create_offer(
        client,
        node_ids=[node["id"]],
        price_per_hour=price_per_hour,
        max_lifetime_hours=max_lifetime_hours,
        mode="single",
    )
    return provider, node, offer


# ----- VAL-MKT-013 rent creates lease + pod ----------------------------------


@pytest.mark.asyncio
async def test_rent_creates_lease_and_pod(market_client: AsyncClient) -> None:
    """VAL-MKT-013: rent on listed offer yields lease + pod with bound fields."""

    _provider, node, offer = await _ready_listed_single(market_client, price_per_hour=3.25)
    rent = await _rent_offer(market_client, offer["id"], lifetime_hours=4.0)
    assert rent["status"] == 200, rent
    body = rent["json"]

    assert "lease" in body and "pod" in body
    lease = body["lease"]
    pod = body["pod"]
    assert lease["id"]
    assert lease["offer_id"] == offer["id"]
    assert lease["renter_hotkey"] == RENTER_HK
    assert lease["price_per_hour"] == 3.25
    assert lease["status"] in {"requested", "active"}
    assert pod["id"]
    assert pod["lease_id"] == lease["id"]
    assert pod["status"] in {"provisioning", "running"}
    assert node["id"] in pod["node_ids"]

    # Durable GETs.
    got_lease = await market_client.get(f"/v1/leases/{lease['id']}")
    assert got_lease.status_code == 200
    assert got_lease.json()["id"] == lease["id"]

    got_pod = await market_client.get(f"/v1/pods/{pod['id']}")
    assert got_pod.status_code == 200
    assert got_pod.json()["id"] == pod["id"]
    assert got_pod.json()["lease_id"] == lease["id"]

    # Offer is no longer browse-listed as rentable exclusive capacity.
    browse = await market_client.get("/v1/offers")
    assert all(o["id"] != offer["id"] for o in browse.json()["items"])
    still = await market_client.get(f"/v1/offers/{offer['id']}")
    assert still.status_code == 200
    assert still.json()["status"] == "leased"


# ----- VAL-MKT-014 double-rent rejected --------------------------------------


@pytest.mark.asyncio
async def test_double_rent_exclusive_rejected(market_client: AsyncClient) -> None:
    """VAL-MKT-014: second rent on exclusive capacity fails 4xx, no second active lease."""

    _p, _n, offer = await _ready_listed_single(market_client)
    first = await _rent_offer(market_client, offer["id"], hotkey=RENTER_HK)
    assert first["status"] == 200, first
    lease_id = first["json"]["lease"]["id"]

    second = await _rent_offer(market_client, offer["id"], hotkey=RENTER2_HK)
    assert 400 <= second["status"] < 500, second
    detail = second["json"].get("detail") or second["json"]
    code = detail.get("code") if isinstance(detail, dict) else None
    assert code in {
        "offer_not_listed",
        "offer_already_leased",
        "capacity_unavailable",
        "already_leased",
    }

    # Only one lease for this offer.
    listed = await market_client.get(
        "/v1/leases",
        headers={"X-Hotkey": RENTER_HK},
    )
    assert listed.status_code == 200
    items = listed.json()["items"]
    offer_leases = [x for x in items if x["offer_id"] == offer["id"]]
    assert len(offer_leases) == 1
    assert offer_leases[0]["id"] == lease_id
    assert offer_leases[0]["status"] in {"requested", "active"}


# ----- VAL-MKT-017 pod running under sim -------------------------------------


@pytest.mark.asyncio
async def test_pod_reaches_running_under_sim(market_client: AsyncClient) -> None:
    """VAL-MKT-017: after rent, local sim path leaves pod running and bound."""

    _p, node, offer = await _ready_listed_single(market_client)
    rent = await _rent_offer(market_client, offer["id"])
    assert rent["status"] == 200, rent
    pod_id = rent["json"]["pod"]["id"]
    lease_id = rent["json"]["lease"]["id"]

    got = await market_client.get(f"/v1/pods/{pod_id}")
    assert got.status_code == 200
    pod = got.json()
    assert pod["status"] == "running"
    assert pod["mode"] == "single"
    assert node["id"] in pod["node_ids"]
    assert pod["lease_id"] == lease_id
    # endpoints / binding readable
    assert "endpoints" in pod or "node_ids" in pod

    lease = (await market_client.get(f"/v1/leases/{lease_id}")).json()
    assert lease["status"] == "active"


# ----- VAL-MKT-015 / 018 terminate frees capacity + stops pod ---------------


@pytest.mark.asyncio
async def test_terminate_ends_lease_and_frees_capacity(market_client: AsyncClient) -> None:
    """VAL-MKT-015/018: renter terminate → terminal lease, stopped pod, capacity free."""

    _p, node, offer = await _ready_listed_single(market_client)
    rent = await _rent_offer(market_client, offer["id"], lifetime_hours=6.0)
    assert rent["status"] == 200, rent
    lease_id = rent["json"]["lease"]["id"]
    pod_id = rent["json"]["pod"]["id"]

    stop = await _terminate_lease(market_client, lease_id, reason="done_early")
    assert stop["status"] == 200, stop
    lease = stop["json"].get("lease") or stop["json"]
    assert lease["status"] in {"terminated", "expired"}
    assert lease.get("termination_reason") in {"done_early", "renter_cancel", "terminated"} or (
        lease.get("termination_reason")
    )

    got_lease = await market_client.get(f"/v1/leases/{lease_id}")
    assert got_lease.json()["status"] in {"terminated", "expired"}

    got_pod = await market_client.get(f"/v1/pods/{pod_id}")
    assert got_pod.status_code == 200
    assert got_pod.json()["status"] in {"stopping", "stopped"}

    # Capacity free: re-list same node or re-rent via new offer.
    reoffer = await _create_offer(
        market_client,
        node_ids=[node["id"]],
        price_per_hour=2.0,
        max_lifetime_hours=5.0,
    )
    assert reoffer["status"] == "listed"
    rerent = await _rent_offer(market_client, reoffer["id"], hotkey=RENTER2_HK)
    assert rerent["status"] == 200, rerent


# ----- VAL-MKT-016 lease list renter/provider views -------------------------


@pytest.mark.asyncio
async def test_lease_list_renter_and_provider_views(market_client: AsyncClient) -> None:
    """VAL-MKT-016: renter and provider each see the lease; third party does not."""

    await _ready_listed_single(market_client)
    # need offer id from helper rebuilt:
    _p, _n, offer = await _ready_listed_single(market_client)
    rent = await _rent_offer(market_client, offer["id"], hotkey=RENTER_HK)
    assert rent["status"] == 200, rent
    lease_id = rent["json"]["lease"]["id"]

    renter_list = await market_client.get(
        "/v1/leases",
        headers={"X-Hotkey": RENTER_HK},
    )
    assert renter_list.status_code == 200
    renter_ids = {x["id"] for x in renter_list.json()["items"]}
    assert lease_id in renter_ids

    provider_list = await market_client.get(
        "/v1/leases",
        headers={"X-Hotkey": PROVIDER_HK},
    )
    assert provider_list.status_code == 200
    provider_ids = {x["id"] for x in provider_list.json()["items"]}
    assert lease_id in provider_ids

    stranger = await market_client.get(
        "/v1/leases",
        headers={"X-Hotkey": RENTER2_HK},
    )
    assert stranger.status_code == 200
    stranger_ids = {x["id"] for x in stranger.json()["items"]}
    assert lease_id not in stranger_ids

    detail = await market_client.get(f"/v1/leases/{lease_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["offer_id"] == offer["id"]
    assert "price_per_hour" in body
    assert "status" in body


@pytest.mark.asyncio
async def test_lease_list_without_hotkey_returns_empty(market_client: AsyncClient) -> None:
    """VAL-MKT-016 fail-closed: GET /v1/leases without X-Hotkey yields empty items.

    Identity-scoped list must never dump the full lease table when the caller
    provides no X-Hotkey. Prefer 200 + items=[] over unscoped dump.
    """

    _p, _n, offer = await _ready_listed_single(market_client)
    rent = await _rent_offer(market_client, offer["id"], hotkey=RENTER_HK)
    assert rent["status"] == 200, rent
    lease_id = rent["json"]["lease"]["id"]

    # Sanity: scoped renter view still sees the lease.
    scoped = await market_client.get("/v1/leases", headers={"X-Hotkey": RENTER_HK})
    assert scoped.status_code == 200
    assert any(x["id"] == lease_id for x in scoped.json()["items"])

    # Fail-closed: missing identity must not enumerate all rentals.
    unscoped = await market_client.get("/v1/leases")
    assert unscoped.status_code == 200, unscoped.text
    body = unscoped.json()
    assert "items" in body
    assert body["items"] == [], body


# ----- VAL-MKT-019 cluster multi-node binding -------------------------------


@pytest.mark.asyncio
async def test_cluster_mode_rent_binds_all_nodes(market_client: AsyncClient) -> None:
    """VAL-MKT-019: cluster offer yields pod mode=cluster with ≥2 bound nodes."""

    await _register_provider(market_client)
    n1 = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=8,
        ssh_endpoint="10.30.0.1:22",
        inventory={
            "ib_devices": [{"name": "mlx5_0", "rate_gbps": 200.0, "state": "ACTIVE"}],
        },
    )
    n2 = await _register_node(
        market_client,
        gpu_model="H100",
        gpu_count=8,
        ssh_endpoint="10.30.0.2:22",
        inventory={
            "ib_devices": [{"name": "mlx5_0", "rate_gbps": 200.0, "state": "ACTIVE"}],
        },
    )
    offer = await _create_offer(
        market_client,
        node_ids=[n1["id"], n2["id"]],
        mode="cluster",
        require_ib=True,
        price_per_hour=10.0,
        max_lifetime_hours=12.0,
    )
    assert offer["mode"] == "cluster"
    assert offer["node_count"] == 2

    rent = await _rent_offer(market_client, offer["id"], lifetime_hours=2.0)
    assert rent["status"] == 200, rent
    pod = rent["json"]["pod"]
    assert pod["mode"] == "cluster"
    assert set(pod["node_ids"]) == {n1["id"], n2["id"]}
    assert len(pod["node_ids"]) >= 2
    assert pod["status"] in {"provisioning", "running"}


# ----- VAL-MKT-020 / 021 tenant short-circuit idle reclaim -------------------


@pytest.mark.asyncio
async def test_active_rental_survives_idle_reclaim(market_client: AsyncClient) -> None:
    """VAL-MKT-020: idle-only reclaim must not kill active lease/pod."""

    _p, node, offer = await _ready_listed_single(market_client)
    rent = await _rent_offer(market_client, offer["id"])
    assert rent["status"] == 200, rent
    lease_id = rent["json"]["lease"]["id"]
    pod_id = rent["json"]["pod"]["id"]

    # AGE the node heartbeat artificially past liveness, then run idle reclaim.
    from sqlalchemy import select

    from hypercluster.db.models import Node
    from hypercluster.domain.leases import run_idle_reclaim_sweep

    app = market_client._transport.app  # type: ignore[attr-defined]
    database = app.state.database
    async with database.session() as session:
        result = await session.execute(select(Node).where(Node.id == node["id"]))
        db_node = result.scalar_one()
        db_node.last_heartbeat = datetime.now(UTC) - timedelta(seconds=3600)
        # Ensure status reflects rental so reclaim can try to abuse it.
        if db_node.status != "rented":
            db_node.status = "rented"
        await session.commit()

        swept = await run_idle_reclaim_sweep(session, liveness_seconds=30)
        # We only assert no harm was done; sweep may reclaim other idle free nodes.
        assert isinstance(swept, int)

    lease = (await market_client.get(f"/v1/leases/{lease_id}")).json()
    pod = (await market_client.get(f"/v1/pods/{pod_id}")).json()
    assert lease["status"] == "active"
    assert pod["status"] == "running"

    # Node still rental-bound (not offline-killed).
    node_after = (await market_client.get(f"/v1/nodes/{node['id']}")).json()
    assert node_after["status"] in {"rented", "healthy"}
    assert node_after["status"] != "offline"


@pytest.mark.asyncio
async def test_active_rental_ends_on_terminate_despite_protection(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-021: tenant short-circuit does not block legitimate terminate."""

    _p, _n, offer = await _ready_listed_single(market_client)
    rent = await _rent_offer(market_client, offer["id"])
    assert rent["status"] == 200
    lease_id = rent["json"]["lease"]["id"]
    pod_id = rent["json"]["pod"]["id"]

    stop = await _terminate_lease(market_client, lease_id)
    assert stop["status"] == 200, stop
    assert (await market_client.get(f"/v1/leases/{lease_id}")).json()["status"] in {
        "terminated",
        "expired",
    }
    assert (await market_client.get(f"/v1/pods/{pod_id}")).json()["status"] in {
        "stopping",
        "stopped",
    }


@pytest.mark.asyncio
async def test_active_rental_ends_on_max_lifetime_expiry(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-021: max lifetime expiry ends lease/pod and frees capacity."""

    _p, node, offer = await _ready_listed_single(
        market_client,
        max_lifetime_hours=24.0,
    )
    # Request very short lifetime; expire via domain sweep with backdated ends_at.
    rent = await _rent_offer(market_client, offer["id"], lifetime_hours=1.0)
    assert rent["status"] == 200, rent
    lease_id = rent["json"]["lease"]["id"]
    pod_id = rent["json"]["pod"]["id"]

    from sqlalchemy import select

    from hypercluster.db.models import Lease
    from hypercluster.domain.leases import expire_due_leases

    app = market_client._transport.app  # type: ignore[attr-defined]
    database = app.state.database
    async with database.session() as session:
        result = await session.execute(select(Lease).where(Lease.id == lease_id))
        lease_row = result.scalar_one()
        lease_row.ends_at = datetime.now(UTC) - timedelta(seconds=5)
        await session.commit()
        n = await expire_due_leases(session)
        assert n >= 1

    assert (await market_client.get(f"/v1/leases/{lease_id}")).json()["status"] == "expired"
    assert (await market_client.get(f"/v1/pods/{pod_id}")).json()["status"] in {
        "stopping",
        "stopped",
    }

    reoffer = await _create_offer(
        market_client,
        node_ids=[node["id"]],
        price_per_hour=1.1,
        max_lifetime_hours=3.0,
    )
    assert reoffer["status"] == "listed"


# ----- VAL-MKT-031 withdraw while leased fails closed -----------------------


@pytest.mark.asyncio
async def test_withdraw_while_leased_rejected(market_client: AsyncClient) -> None:
    """VAL-MKT-031: provider cannot withdraw offer under active lease; pod intact."""

    _p, _n, offer = await _ready_listed_single(market_client)
    rent = await _rent_offer(market_client, offer["id"])
    assert rent["status"] == 200, rent
    lease_id = rent["json"]["lease"]["id"]
    pod_id = rent["json"]["pod"]["id"]

    denied = await _withdraw_offer(market_client, offer["id"])
    assert denied["status"] in {409, 400, 422}, denied
    detail = denied["json"].get("detail") or denied["json"]
    code = detail.get("code") if isinstance(detail, dict) else None
    assert code in {"offer_active_lease", "offer_leased", "conflict", "active_lease"}

    # Lease/pod still non-terminal.
    assert (await market_client.get(f"/v1/leases/{lease_id}")).json()["status"] == "active"
    assert (await market_client.get(f"/v1/pods/{pod_id}")).json()["status"] == "running"
    assert (await market_client.get(f"/v1/offers/{offer['id']}")).json()["status"] == "leased"

    # After terminal lease, withdraw of a new listed offer path is allowed;
    # terminate first then re-list goes through ordinary create.
    stop = await _terminate_lease(market_client, lease_id)
    assert stop["status"] == 200

    # Sometimes offer stays leased until free; create new offer on freed node.
    node_id = rent["json"]["pod"]["node_ids"][0]
    reoffer = await _create_offer(
        market_client,
        node_ids=[node_id],
        price_per_hour=1.5,
        max_lifetime_hours=2.0,
    )
    # New listed offer can be withdrawn freely (no leases).
    withdrawn = await _withdraw_offer(market_client, reoffer["id"])
    assert withdrawn["status"] == 200, withdrawn
    assert withdrawn["json"]["status"] == "withdrawn"


# ----- renter max_price guard (VAL-MKT-010 rent side) -----------------------


@pytest.mark.asyncio
async def test_rent_rejects_price_above_renter_max(market_client: AsyncClient) -> None:
    """Renter max_price bound rejects rent without creating lease/pod."""

    _p, _n, offer = await _ready_listed_single(market_client, price_per_hour=10.0)
    before = await market_client.get("/v1/leases", headers={"X-Hotkey": RENTER_HK})
    counted = len(before.json()["items"]) if before.status_code == 200 else 0

    rent = await _rent_offer(
        market_client,
        offer["id"],
        max_price=5.0,
        lifetime_hours=2.0,
    )
    assert 400 <= rent["status"] < 500, rent

    after = await market_client.get("/v1/leases", headers={"X-Hotkey": RENTER_HK})
    if after.status_code == 200:
        assert len(after.json()["items"]) == counted
    still = await market_client.get(f"/v1/offers/{offer['id']}")
    assert still.json()["status"] == "listed"
