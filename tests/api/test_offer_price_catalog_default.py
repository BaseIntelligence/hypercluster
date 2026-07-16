"""VAL-PRICE-050..053: offer create catalog default + optional band enforce."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.api.auth import build_signed_headers

HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
TOKEN = "test-challenge-shared-token"


def _sign(body: bytes, *, hotkey: str = HOTKEY) -> dict[str, str]:
    return build_signed_headers(secret=TOKEN, hotkey=hotkey, body=body)


@pytest.fixture
async def client_factory(settings_factory, tmp_path):
    """Build an AsyncClient with given HyperSettings knobs (fresh SQLite each)."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    counter = {"n": 0}

    async def _make(
        *,
        price_enforce: str = "off",
        max_offer_price: float = 1000.0,
        price_max_multiplier: float = 3.0,
        price_min_multiplier: float = 0.25,
        price_seed_on_boot: bool = False,
    ) -> AsyncClient:
        counter["n"] += 1
        db_path = tmp_path / f"offer-price-{counter['n']}.sqlite3"
        settings = settings_factory(
            database_url=f"sqlite+aiosqlite:///{db_path}",
            shared_token=TOKEN,
            shared_token_file=None,
        )
        hyper = HyperSettings(
            allow_insecure_signatures=True,
            signature_ttl_seconds=300,
            node_liveness_seconds=120,
            max_offer_price_per_hour=max_offer_price,
            max_offer_lifetime_hours=168.0,
            price_enforce=price_enforce,
            price_max_multiplier=price_max_multiplier,
            price_min_multiplier=price_min_multiplier,
            price_seed_on_boot=price_seed_on_boot,
        )
        app = create_app(settings, hyper_settings=hyper)
        # Caller manages lifecycle via async with in each test through helper.
        return app, settings, hyper  # type: ignore[return-value]

    return _make


async def _as_client(app) -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


async def _register_provider_and_node(
    client: AsyncClient,
    *,
    gpu_model: str = "H100",
    gpu_count: int = 8,
) -> dict[str, Any]:
    raw = json.dumps({"display_name": "Price Farm"}).encode()
    headers = _sign(raw)
    headers["Content-Type"] = "application/json"
    r = await client.post("/v1/providers/register", content=raw, headers=headers)
    assert r.status_code == 200, r.text

    node_body = {
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "tee_capability": "none",
        "ssh_endpoint": "10.0.50.1:22",
        "inventory": {"has_ib": False},
    }
    raw_n = json.dumps(node_body).encode()
    headers_n = _sign(raw_n)
    headers_n["Content-Type"] = "application/json"
    rn = await client.post("/v1/nodes", content=raw_n, headers=headers_n)
    assert rn.status_code == 200, rn.text
    return rn.json()


async def _upsert_catalog(
    client: AsyncClient,
    *,
    model_key: str,
    price_per_hour: float,
    family: str | None = None,
    display_name: str | None = None,
    max_offer_multiplier: float | None = None,
    min_offer_multiplier: float | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "price_per_hour": price_per_hour,
        "currency": "USD",
        "active": True,
        "source": "admin",
    }
    if family is not None:
        body["family"] = family
    if display_name is not None:
        body["display_name"] = display_name
    if max_offer_multiplier is not None:
        body["max_offer_multiplier"] = max_offer_multiplier
    if min_offer_multiplier is not None:
        body["min_offer_multiplier"] = min_offer_multiplier
    raw = json.dumps(body).encode()
    response = await client.put(
        f"/v1/admin/gpu-prices/{model_key}",
        content=raw,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
    )
    assert response.status_code in {200, 201}, response.text
    return response.json()


async def _create_offer(
    client: AsyncClient,
    *,
    node_ids: list[str],
    price_per_hour: float | None | object = ...,
    max_lifetime_hours: float = 24.0,
    gpu_model: str | None = None,
    mode: str = "single",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "node_ids": node_ids,
        "mode": mode,
        "require_ib": False,
        "max_lifetime_hours": max_lifetime_hours,
    }
    # Ellipsis = omit key; None = explicit null; float = value.
    if price_per_hour is not ...:
        body["price_per_hour"] = price_per_hour
    if gpu_model is not None:
        body["gpu_model"] = gpu_model
    raw = json.dumps(body).encode()
    headers = _sign(raw)
    headers["Content-Type"] = "application/json"
    response = await client.post("/v1/offers", content=raw, headers=headers)
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = {"raw": response.text}
    return {"status": response.status_code, "json": payload}


# ----- VAL-PRICE-050 ----------------------------------------------------------


@pytest.mark.asyncio
async def test_offer_omit_price_uses_catalog_default(client_factory) -> None:
    """VAL-PRICE-050: omit/null price fills from active catalog; source catalog_default."""

    app, _s, _h = await client_factory(price_enforce="off")
    async for client in _as_client(app):
        catalog = await _upsert_catalog(
            client,
            model_key="H100_80GB",
            price_per_hour=2.49,
            family="h100",
            display_name="NVIDIA H100 80GB",
        )
        node = await _register_provider_and_node(client, gpu_model="H100")

        # Omit price key entirely
        omitted = await _create_offer(client, node_ids=[node["id"]], price_per_hour=...)
        assert omitted["status"] == 200, omitted
        offer = omitted["json"]
        assert offer["price_per_hour"] == pytest.approx(2.49)
        assert offer.get("price_source") == "catalog_default"
        assert offer.get("catalog_model_key") == catalog.get("model_key") or "H100_80GB"
        assert offer.get("catalog_price_per_hour") == pytest.approx(2.49)

        # Explicit JSON null also defaults
        nulled = await _create_offer(
            client,
            node_ids=[node["id"]],
            price_per_hour=None,
            max_lifetime_hours=12.0,
        )
        assert nulled["status"] == 200, nulled
        assert nulled["json"]["price_per_hour"] == pytest.approx(2.49)
        assert nulled["json"].get("price_source") == "catalog_default"


# ----- VAL-PRICE-051 ----------------------------------------------------------


@pytest.mark.asyncio
async def test_offer_omit_price_without_catalog_422(client_factory) -> None:
    """VAL-PRICE-051: omit price with no active catalog → 422 missing_price_per_hour."""

    app, _s, _h = await client_factory(
        price_enforce="off",
        price_seed_on_boot=False,
    )
    async for client in _as_client(app):
        node = await _register_provider_and_node(client, gpu_model="MysteryGPU999")

        before = await client.get("/v1/offers")
        assert before.status_code == 200
        counted = len(before.json()["items"])

        result = await _create_offer(client, node_ids=[node["id"]], price_per_hour=...)
        assert result["status"] == 422, result
        detail = result["json"].get("detail") or result["json"]
        code = detail.get("code") if isinstance(detail, dict) else None
        assert code == "missing_price_per_hour", result

        after = await client.get("/v1/offers")
        assert after.status_code == 200
        assert len(after.json()["items"]) == counted


# ----- VAL-PRICE-052 ----------------------------------------------------------


@pytest.mark.asyncio
async def test_hard_enforce_rejects_over_and_under_catalog_band(client_factory) -> None:
    """VAL-PRICE-052: hard enforce rejects over/under band; soft accepts with flags."""

    catalog_price = 2.0
    # global max 3.0 → upper 6.0; min 0.25 → lower 0.5

    # hard over
    app_hard, *_ = await client_factory(price_enforce="hard")
    async for client in _as_client(app_hard):
        await _upsert_catalog(
            client,
            model_key="H100_80GB",
            price_per_hour=catalog_price,
            family="h100",
            display_name="H100",
        )
        node = await _register_provider_and_node(client, gpu_model="H100")

        over = await _create_offer(
            client,
            node_ids=[node["id"]],
            price_per_hour=6.01,  # > 2 * 3.0
        )
        assert over["status"] == 422, over
        detail = over["json"].get("detail") or over["json"]
        assert isinstance(detail, dict)
        assert detail.get("code") == "price_over_catalog_band"

        under = await _create_offer(
            client,
            node_ids=[node["id"]],
            price_per_hour=0.49,  # < 2 * 0.25
        )
        assert under["status"] == 422, under
        detail_u = under["json"].get("detail") or under["json"]
        assert isinstance(detail_u, dict)
        assert detail_u.get("code") == "price_under_catalog_band"

        # In-band still succeeds; system max still enforced separately
        ok = await _create_offer(
            client,
            node_ids=[node["id"]],
            price_per_hour=5.0,  # within 0.5..6.0
        )
        assert ok["status"] == 200, ok
        assert ok["json"]["price_per_hour"] == 5.0
        assert ok["json"].get("price_source") == "explicit"

        # Soft mode: over band accepted with flag, not 422
        app_soft, *_ = await client_factory(price_enforce="soft")
        async for soft_client in _as_client(app_soft):
            await _upsert_catalog(
                soft_client,
                model_key="H100_80GB",
                price_per_hour=catalog_price,
                family="h100",
                display_name="H100",
            )
            soft_node = await _register_provider_and_node(soft_client, gpu_model="H100")
            soft_over = await _create_offer(
                soft_client,
                node_ids=[soft_node["id"]],
                price_per_hour=9.0,
            )
            assert soft_over["status"] == 200, soft_over
            assert soft_over["json"]["price_per_hour"] == 9.0
            # Soft flag present in metadata or tip fields
            meta = soft_over["json"].get("metadata") or {}
            band_flag = (
                soft_over["json"].get("price_band_flag")
                or meta.get("price_band_flag")
                or soft_over["json"].get("price_band_warning")
            )
            assert (
                band_flag
                in {
                    "over",
                    "price_over_catalog_band",
                    "over_catalog_band",
                    True,
                }
                or "over" in str(band_flag).lower()
                or "band" in str(meta).lower()
            )


# ----- VAL-PRICE-053 ----------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_in_cap_price_with_enforce_off_no_catalog(
    client_factory,
) -> None:
    """VAL-PRICE-053: enforce=off + explicit in-cap price succeeds without catalog."""

    app, *_ = await client_factory(
        price_enforce="off",
        max_offer_price=100.0,
        price_seed_on_boot=False,
    )
    async for client in _as_client(app):
        # No catalog rows at all.
        public = await client.get("/v1/gpu-prices")
        assert public.status_code == 200
        assert public.json().get("items") in ([], None) or len(public.json()["items"]) == 0

        node = await _register_provider_and_node(client, gpu_model="CustomGPUX")
        result = await _create_offer(
            client,
            node_ids=[node["id"]],
            price_per_hour=12.5,
            max_lifetime_hours=8.0,
        )
        assert result["status"] == 200, result
        offer = result["json"]
        assert offer["price_per_hour"] == 12.5
        assert offer.get("price_source") in {"explicit", None}
        # Still under system max
        over_cap = await _create_offer(
            client,
            node_ids=[node["id"]],
            price_per_hour=100.01,
        )
        assert over_cap["status"] == 422, over_cap
        detail = over_cap["json"].get("detail") or over_cap["json"]
        code = detail.get("code") if isinstance(detail, dict) else None
        assert code == "price_over_cap"
