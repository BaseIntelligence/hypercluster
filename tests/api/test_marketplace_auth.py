"""VAL-MKT-022 / 023 / 024: marketplace signed write auth and safe reads.

Fail-closed state for mutating routes; browse/list reads stay policy-open.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.api.auth import (
    TIMESTAMP_HEADER,
    build_signed_headers,
    sign_dev,
)

PROVIDER_HK = "auth-provider-hotkey-aaaaaaaaaaaaaaaaaaaaaaaa"
RENTER_HK = "auth-renter-hotkey-bbbbbbbbbbbbbbbbbbbbbbbbbbbb"
FOREIGN_HK = "auth-foreign-hotkey-cccccccccccccccccccccccccc"
TOKEN = "test-challenge-shared-token"

# All marketplace write endpoints that must fail closed without valid signatures.
WRITE_ROUTES: list[tuple[str, str, bytes]] = [
    ("POST", "/v1/providers/register", b'{"display_name":"probe"}'),
    ("POST", "/v1/providers/heartbeat", b"{}"),
    ("POST", "/v1/nodes", b'{"gpu_model":"H100","gpu_count":1}'),
    ("POST", "/v1/offers", b'{"node_ids":["x"],"price_per_hour":1,"max_lifetime_hours":1}'),
    ("DELETE", "/v1/offers/placeholder-offer-id", b""),
    ("POST", "/v1/offers/placeholder-offer-id/rent", b'{"lifetime_hours":1}'),
    ("POST", "/v1/leases/placeholder-lease-id/terminate", b'{"reason":"probe"}'),
]


def _sign(
    body: bytes,
    *,
    hotkey: str = PROVIDER_HK,
    nonce: str | None = None,
    timestamp: int | None = None,
) -> dict[str, str]:
    return build_signed_headers(
        secret=TOKEN,
        hotkey=hotkey,
        body=body,
        nonce=nonce,
        timestamp=timestamp,
    )


@pytest.fixture
async def market_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    """App with insecure HMAC signatures for auth matrix tests."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'auth.sqlite3'}",
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


async def _seed_offer(client: AsyncClient) -> dict[str, Any]:
    """Register provider+node+offer for foreign-signature / happy paths."""

    reg = json.dumps({"display_name": "Auth Farm"}).encode()
    headers = _sign(reg, hotkey=PROVIDER_HK)
    headers["Content-Type"] = "application/json"
    response = await client.post("/v1/providers/register", content=reg, headers=headers)
    assert response.status_code == 200, response.text

    node_body = json.dumps(
        {
            "gpu_model": "H100",
            "gpu_count": 4,
            "ssh_endpoint": "10.1.2.3:22",
            "inventory": {"ib_devices": ["mlx5_0"], "ib_rate_gbps": 200.0},
        }
    ).encode()
    headers = _sign(node_body, hotkey=PROVIDER_HK)
    headers["Content-Type"] = "application/json"
    node_resp = await client.post("/v1/nodes", content=node_body, headers=headers)
    assert node_resp.status_code == 200, node_resp.text
    node_id = node_resp.json()["id"]

    offer_body = json.dumps(
        {
            "node_ids": [node_id],
            "price_per_hour": 1.5,
            "max_lifetime_hours": 12.0,
            "require_ib": True,
        }
    ).encode()
    headers = _sign(offer_body, hotkey=PROVIDER_HK)
    headers["Content-Type"] = "application/json"
    offer_resp = await client.post("/v1/offers", content=offer_body, headers=headers)
    assert offer_resp.status_code == 200, offer_resp.text
    return offer_resp.json()


# ----- VAL-MKT-022: missing signature headers ---------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path", "body"), WRITE_ROUTES, ids=[r[1] for r in WRITE_ROUTES])
async def test_write_routes_reject_missing_signature_headers(
    market_client: AsyncClient,
    method: str,
    path: str,
    body: bytes,
) -> None:
    """VAL-MKT-022: every mutating marketplace route fail-closes without signatures."""

    headers = {"Content-Type": "application/json"} if body else {}
    response = await market_client.request(method, path, content=body, headers=headers)
    assert response.status_code in {401, 403}, (
        f"{method} {path} expected 401/403 without auth, "
        f"got {response.status_code}: {response.text}"
    )
    detail = response.json().get("detail")
    # Prefer structured code when present.
    if isinstance(detail, dict):
        assert (
            detail.get("code")
            in {
                "missing_auth_headers",
                "invalid_signature",
                "missing_hotkey",
            }
            or "auth" in str(detail).lower()
            or "sign" in str(detail).lower()
        )


@pytest.mark.asyncio
async def test_missing_auth_does_not_register_provider(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-022: unsigned register must not create a provider row."""

    raw = json.dumps({"display_name": "Ghost"}).encode()
    response = await market_client.post(
        "/v1/providers/register",
        content=raw,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in {401, 403}, response.text

    listed = await market_client.get("/v1/providers")
    assert listed.status_code == 200
    assert listed.json()["items"] == []


# ----- VAL-MKT-023: invalid / foreign signatures ------------------------------


@pytest.mark.asyncio
async def test_write_rejects_bad_signature(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-023: forged signature → 401/403, no mutation."""

    raw = json.dumps({"display_name": "Forged"}).encode()
    headers = _sign(raw, hotkey=PROVIDER_HK)
    headers["X-Signature"] = "deadbeef" * 8  # wrong digest
    headers["Content-Type"] = "application/json"
    response = await market_client.post("/v1/providers/register", content=raw, headers=headers)
    assert response.status_code in {401, 403}, response.text
    detail = response.json().get("detail")
    if isinstance(detail, dict):
        assert detail.get("code") in {"invalid_signature", "missing_auth_headers"}

    listed = await market_client.get("/v1/providers")
    assert listed.json()["items"] == []


@pytest.mark.asyncio
async def test_write_rejects_stale_timestamp(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-023: timestamp outside skew window is rejected."""

    raw = json.dumps({"display_name": "Stale"}).encode()
    # Far outside default 300s TTL
    headers = _sign(raw, hotkey=PROVIDER_HK, timestamp=int(time.time()) - 10_000)
    headers["Content-Type"] = "application/json"
    response = await market_client.post("/v1/providers/register", content=raw, headers=headers)
    assert response.status_code in {401, 403}, response.text
    detail = response.json().get("detail")
    if isinstance(detail, dict):
        assert detail.get("code") in {"stale_signature", "invalid_signature"}


@pytest.mark.asyncio
async def test_write_rejects_reused_nonce(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-023: nonce replay is fail-closed."""

    raw = json.dumps({"display_name": "First"}).encode()
    fixed_nonce = "replay-nonce-fixed-001"
    headers = _sign(raw, hotkey=PROVIDER_HK, nonce=fixed_nonce)
    headers["Content-Type"] = "application/json"
    first = await market_client.post("/v1/providers/register", content=raw, headers=headers)
    assert first.status_code == 200, first.text

    # Re-use same nonce with a fresh body signature still records same nonce key.
    raw2 = json.dumps({"display_name": "Replay"}).encode()
    headers2 = _sign(raw2, hotkey=PROVIDER_HK, nonce=fixed_nonce)
    headers2["Content-Type"] = "application/json"
    second = await market_client.post("/v1/providers/register", content=raw2, headers=headers2)
    assert second.status_code in {401, 403}, second.text
    detail = second.json().get("detail")
    if isinstance(detail, dict):
        assert detail.get("code") == "nonce_replay"


@pytest.mark.asyncio
async def test_foreign_hotkey_cannot_withdraw_owner_offer(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-023: valid sig for hotkey A cannot withdraw B's offer."""

    offer = await _seed_offer(market_client)
    offer_id = offer["id"]

    # Foreign hotkey must register first so domain ownership fails cleanly / at nest.
    reg = json.dumps({"display_name": "Foreign"}).encode()
    headers = _sign(reg, hotkey=FOREIGN_HK)
    headers["Content-Type"] = "application/json"
    reg_resp = await market_client.post("/v1/providers/register", content=reg, headers=headers)
    assert reg_resp.status_code == 200, reg_resp.text

    raw = b""
    headers = _sign(raw, hotkey=FOREIGN_HK)
    response = await market_client.request(
        "DELETE",
        f"/v1/offers/{offer_id}",
        content=raw,
        headers=headers,
    )
    assert response.status_code in {403, 404}, response.text

    # Offer still listed (not withdrawn).
    got = await market_client.get(f"/v1/offers/{offer_id}")
    assert got.status_code == 200
    assert got.json()["status"] == "listed"
    assert got.json()["id"] == offer_id


@pytest.mark.asyncio
async def test_mismatched_body_binding_rejects(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-023: signature bound to body A cannot authorize body B."""

    body_a = json.dumps({"display_name": "Alpha"}).encode()
    body_b = json.dumps({"display_name": "Beta"}).encode()
    headers = _sign(body_a, hotkey=PROVIDER_HK)
    headers["Content-Type"] = "application/json"
    # Send different body than was signed.
    response = await market_client.post("/v1/providers/register", content=body_b, headers=headers)
    assert response.status_code in {401, 403}, response.text


# ----- VAL-MKT-024: safe reads without write signatures -----------------------


@pytest.mark.asyncio
async def test_safe_reads_without_signatures(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-024: browse/list and identity remain usable without miner write sigs."""

    # Seed so browse returns something.
    offer = await _seed_offer(market_client)

    health = await market_client.get("/health")
    assert health.status_code == 200
    assert health.json()["slug"] == "hypercluster"

    version = await market_client.get("/version")
    assert version.status_code == 200
    assert version.json()["challenge_slug"] == "hypercluster"

    offers = await market_client.get("/v1/offers")
    assert offers.status_code == 200
    items = offers.json()["items"]
    assert any(o["id"] == offer["id"] for o in items)

    providers = await market_client.get("/v1/providers")
    assert providers.status_code == 200
    assert len(providers.json()["items"]) >= 1

    # GET does not mutate: re-list still same listed offer.
    again = await market_client.get("/v1/offers")
    assert any(o["id"] == offer["id"] and o["status"] == "listed" for o in again.json()["items"])


@pytest.mark.asyncio
async def test_safe_reads_paired_with_unsigned_post_denial(
    market_client: AsyncClient,
) -> None:
    """VAL-MKT-024: unsigned GET ok, paired unsigned POST remains fail-closed."""

    offers = await market_client.get("/v1/offers")
    assert offers.status_code == 200

    raw = json.dumps(
        {
            "node_ids": ["does-not-matter"],
            "price_per_hour": 1.0,
            "max_lifetime_hours": 1.0,
        }
    ).encode()
    denied = await market_client.post(
        "/v1/offers",
        content=raw,
        headers={"Content-Type": "application/json"},
    )
    assert denied.status_code in {401, 403}, denied.text


@pytest.mark.asyncio
async def test_dev_hmac_helpers_roundtrip() -> None:
    """Sanity: sign_dev / build_signed_headers produce usable digests."""

    body = b'{"ok":true}'
    headers = build_signed_headers(
        secret=TOKEN,
        hotkey=PROVIDER_HK,
        body=body,
        timestamp=1_700_000_000,
    )
    assert headers["X-Hotkey"] == PROVIDER_HK
    assert headers["X-Signature"]
    assert headers[TIMESTAMP_HEADER] == "1700000000"
    # Re-derive with same inputs (fixed nonce branches through helper).
    message = (
        f"hypercluster:{PROVIDER_HK}:{headers['X-Nonce']}:{headers['X-Timestamp']}:"
        f"{__import__('hashlib').sha256(body).hexdigest()}"
    ).encode()
    assert sign_dev(TOKEN, message) == headers["X-Signature"]
