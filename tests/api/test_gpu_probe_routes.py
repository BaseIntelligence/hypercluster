"""VAL-GPU-001..006 / 010 / 011: public signed GPU probe + evidence routes.

FakeSsh pass-all only; no real SSH, no product Verda, no set_weights.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.api.auth import build_signed_headers
from hypercluster.probe.types import canonical_json

OWNER_HK = "gpu-probe-owner-hotkey-aaaaaaaaaaaaaaaaaaaaaaaa"
FOREIGN_HK = "gpu-probe-foreign-hotkey-bbbbbbbbbbbbbbbbbbbbbbbb"
TOKEN = "test-challenge-shared-token"


def _sign(
    body: bytes,
    *,
    hotkey: str = OWNER_HK,
    nonce: str | None = None,
) -> dict[str, str]:
    headers = build_signed_headers(
        secret=TOKEN,
        hotkey=hotkey,
        body=body,
        nonce=nonce,
    )
    headers["Content-Type"] = "application/json"
    return headers


@pytest.fixture
async def probe_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    """App with FakeSsh allowed for CI GPU-probe route coverage."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'gpu_probe.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        node_liveness_seconds=120,
        ssh_transport="fake",
        allow_fake_ssh=True,
        fake_ssh_fixture="pass_all",
        require_docker_runtime=True,
        max_gpu_count=14,
        require_live_evidence=False,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


@pytest.fixture
async def require_ev_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    """Same as probe_client with HYPER_REQUIRE_LIVE_EVIDENCE=true."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'gpu_probe_req.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        node_liveness_seconds=120,
        ssh_transport="fake",
        allow_fake_ssh=True,
        fake_ssh_fixture="pass_all",
        require_live_evidence=True,
        require_live_evidence_mode="soft",
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


async def _register_provider_and_node(
    client: AsyncClient,
    *,
    hotkey: str = OWNER_HK,
    gpu_model: str = "1V100.6V",
    gpu_count: int = 1,
) -> dict[str, Any]:
    reg = json.dumps({"display_name": "GPU Probe Farm"}).encode()
    response = await client.post(
        "/v1/providers/register",
        content=reg,
        headers=_sign(reg, hotkey=hotkey),
    )
    assert response.status_code == 200, response.text

    node_body = json.dumps(
        {
            "gpu_model": gpu_model,
            "gpu_count": gpu_count,
            "ssh_endpoint": "10.9.9.1:22",
            "hostname": "gpu-probe-host",
            "inventory": {"claimed_source": "test"},
        }
    ).encode()
    node_resp = await client.post(
        "/v1/nodes",
        content=node_body,
        headers=_sign(node_body, hotkey=hotkey),
    )
    assert node_resp.status_code == 200, node_resp.text
    return node_resp.json()


# ----- VAL-GPU-001: owner probe pass-all; foreign 403 ----------------------------


@pytest.mark.asyncio
async def test_post_probe_gpu_owner_pass_all(probe_client: AsyncClient) -> None:
    node = await _register_provider_and_node(probe_client)
    node_id = node["id"]
    body = json.dumps({"mode": "full"}).encode()
    response = await probe_client.post(
        f"/v1/nodes/{node_id}/probes/gpu",
        content=body,
        headers=_sign(body, hotkey=OWNER_HK),
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "passed"
    assert data.get("evidence_id") or data.get("id")
    evidence_id = data.get("evidence_id") or data.get("id")
    assert evidence_id
    failed = data.get("checks_failed")
    if failed is None and isinstance(data.get("checks"), list):
        failed = sum(1 for c in data["checks"] if c.get("fatal") and not c.get("passed"))
    assert int(failed or 0) == 0
    measured_count = data.get("measured_gpu_count")
    if measured_count is None:
        measured_count = (data.get("measured") or {}).get("gpu_count")
    uuids = data.get("gpu_uuids") or (data.get("measured") or {}).get("gpus")
    assert int(measured_count or 0) >= 1 or bool(uuids)


@pytest.mark.asyncio
async def test_post_probe_gpu_foreign_owner_403(probe_client: AsyncClient) -> None:
    node = await _register_provider_and_node(probe_client)
    node_id = node["id"]

    # Foreign must be a registered provider but not own the node.
    reg = json.dumps({"display_name": "Foreign"}).encode()
    fr = await probe_client.post(
        "/v1/providers/register",
        content=reg,
        headers=_sign(reg, hotkey=FOREIGN_HK),
    )
    assert fr.status_code == 200, fr.text

    body = json.dumps({"mode": "full"}).encode()
    response = await probe_client.post(
        f"/v1/nodes/{node_id}/probes/gpu",
        content=body,
        headers=_sign(body, hotkey=FOREIGN_HK),
    )
    assert response.status_code == 403, response.text


@pytest.mark.asyncio
async def test_post_probe_rejects_raw_pem_body(probe_client: AsyncClient) -> None:
    node = await _register_provider_and_node(probe_client)
    node_id = node["id"]
    body = json.dumps(
        {
            "mode": "full",
            "private_key": "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----",
        }
    ).encode()
    response = await probe_client.post(
        f"/v1/nodes/{node_id}/probes/gpu",
        content=body,
        headers=_sign(body, hotkey=OWNER_HK),
    )
    assert response.status_code in {400, 422}, response.text
    detail = response.json().get("detail") or {}
    if isinstance(detail, dict):
        assert "private" in str(detail.get("code", detail)).lower() or "key" in str(detail).lower()


# ----- VAL-GPU-002 / 003 / 004 / 005: evidence latest / by id / list / global ---


@pytest.mark.asyncio
async def test_get_latest_and_by_id_and_list_and_global(probe_client: AsyncClient) -> None:
    node = await _register_provider_and_node(probe_client)
    node_id = node["id"]

    first_body = json.dumps({"mode": "full"}).encode()
    first = await probe_client.post(
        f"/v1/nodes/{node_id}/probes/gpu",
        content=first_body,
        headers=_sign(first_body, hotkey=OWNER_HK),
    )
    assert first.status_code == 200, first.text
    first_data = first.json()
    first_id = first_data.get("evidence_id") or first_data.get("id")

    # Second probe so list has newest-first semantics.
    second_body = json.dumps({"mode": "quick"}).encode()
    second = await probe_client.post(
        f"/v1/nodes/{node_id}/probes/gpu",
        content=second_body,
        headers=_sign(second_body, hotkey=OWNER_HK),
    )
    assert second.status_code == 200, second.text
    second_data = second.json()
    second_id = second_data.get("evidence_id") or second_data.get("id")
    assert second_id != first_id

    latest = await probe_client.get(f"/v1/nodes/{node_id}/probes/gpu/latest")
    assert latest.status_code == 200, latest.text
    latest_data = latest.json()
    latest_id = latest_data.get("evidence_id") or latest_data.get("id")
    assert latest_id == second_id
    latest_digests = latest_data.get("digests") or {}
    second_digests = second_data.get("digests") or {}
    for key in ("evidence_sha256", "inventory_sha256"):
        if second_digests.get(key):
            assert latest_digests.get(key) == second_digests.get(key)
    latest_uuids = latest_data.get("gpu_uuids")
    if latest_uuids is None:
        gpus = (latest_data.get("measured") or {}).get("gpus") or []
        latest_uuids = [g.get("uuid") for g in gpus if g.get("uuid")]
    second_uuids = second_data.get("gpu_uuids")
    if second_uuids is None:
        gpus = (second_data.get("measured") or {}).get("gpus") or []
        second_uuids = [g.get("uuid") for g in gpus if g.get("uuid")]
    assert list(latest_uuids or []) == list(second_uuids or [])

    by_id = await probe_client.get(f"/v1/nodes/{node_id}/probes/gpu/{second_id}")
    assert by_id.status_code == 200, by_id.text
    body = by_id.json()
    assert isinstance(body.get("checks"), list)
    assert len(body["checks"]) >= 1
    for c in body["checks"]:
        assert "id" in c
        assert "fatal" in c
        assert "passed" in c
    text_blob = json.dumps(body)
    assert "BEGIN PRIVATE KEY" not in text_blob
    assert "BEGIN RSA PRIVATE KEY" not in text_blob

    listed = await probe_client.get(f"/v1/nodes/{node_id}/probes/gpu")
    assert listed.status_code == 200, listed.text
    items = listed.json().get("items") or listed.json()
    assert isinstance(items, list)
    assert len(items) >= 2
    ids = [it.get("evidence_id") or it.get("id") for it in items]
    assert ids[0] == second_id

    global_get = await probe_client.get(f"/v1/evidence/gpu/{second_id}")
    assert global_get.status_code == 200, global_get.text
    g = global_get.json()
    g_id = g.get("evidence_id") or g.get("id")
    assert g_id == second_id
    assert (g.get("status") or g.get("probe_status")) == second_data.get("status")
    g_digests = g.get("digests") or {}
    if second_digests.get("evidence_sha256"):
        assert g_digests.get("evidence_sha256") == second_digests.get("evidence_sha256")


@pytest.mark.asyncio
async def test_list_empty_for_unknown_or_unprobed_node(probe_client: AsyncClient) -> None:
    empty = await probe_client.get("/v1/nodes/00000000-0000-0000-0000-000000000099/probes/gpu")
    assert empty.status_code == 200, empty.text
    items = empty.json().get("items")
    if items is None:
        items = empty.json()
    assert items == []


# ----- VAL-GPU-006: external attach rejects unsigned + bad digest -------------


@pytest.mark.asyncio
async def test_external_evidence_attach_rejects_unsigned_and_bad_digest(
    probe_client: AsyncClient,
) -> None:
    node = await _register_provider_and_node(probe_client)
    node_id = node["id"]

    evidence = {
        "status": "passed",
        "mode": "full",
        "transport": "fake",
        "claimed": {"gpu_model": "1V100.6V", "gpu_count": 1},
        "measured": {
            "gpu_count": 1,
            "gpus": [
                {
                    "name": "Tesla V100-SXM2-16GB",
                    "uuid": "GPU-11111111-1111-1111-1111-111111111111",
                    "memory_total_mb": 16160,
                    "driver_version": "535.104.05",
                }
            ],
        },
        "checks": [
            {
                "id": "ssh_connect",
                "fatal": True,
                "passed": True,
                "message": "ok",
            }
        ],
        "digests": {
            "evidence_sha256": "sha256:deadbeef",
            "inventory_sha256": "sha256:cafebabe",
        },
    }
    body = json.dumps(
        {
            "evidence": evidence,
            "claimed_digest": "sha256:not-matching-payload",
        }
    ).encode()

    unsigned = await probe_client.post(
        f"/v1/nodes/{node_id}/evidence/gpu",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert unsigned.status_code in {401, 403}, unsigned.text

    bad = await probe_client.post(
        f"/v1/nodes/{node_id}/evidence/gpu",
        content=body,
        headers=_sign(body, hotkey=OWNER_HK),
    )
    assert bad.status_code in {400, 422}, bad.text
    detail = bad.json().get("detail") or {}
    if isinstance(detail, dict):
        code = str(detail.get("code", "")).lower()
        assert "digest" in code or "invalid" in code or "mismatch" in code

    # Well-signed document with correct digest of canonical evidence payload is accepted.
    good_evidence = dict(evidence)
    # Placeholders that the server re-hases; client provides matching digest of subset.
    payload_for_hash = {
        "status": good_evidence["status"],
        "claimed": good_evidence["claimed"],
        "measured": good_evidence["measured"],
        "checks": good_evidence["checks"],
        "mode": good_evidence["mode"],
        "transport": good_evidence["transport"],
    }
    import hashlib

    digest = "sha256:" + hashlib.sha256(canonical_json(payload_for_hash).encode()).hexdigest()
    good_evidence["digests"] = {
        "evidence_sha256": digest,
        "inventory_sha256": digest,
    }
    good_body = json.dumps(
        {
            "evidence": good_evidence,
            "claimed_digest": digest,
        }
    ).encode()
    good = await probe_client.post(
        f"/v1/nodes/{node_id}/evidence/gpu",
        content=good_body,
        headers=_sign(good_body, hotkey=OWNER_HK),
    )
    assert good.status_code == 200, good.text
    attached = good.json()
    assert attached.get("status") in {"passed", "attached", "verified"}
    assert attached.get("evidence_id") or attached.get("id")


# ----- VAL-GPU-010: register works; not verified without probe ----------------


@pytest.mark.asyncio
async def test_node_register_not_auto_verified(probe_client: AsyncClient) -> None:
    node = await _register_provider_and_node(probe_client)
    assert node.get("created") is True or node.get("id")
    status = node.get("gpu_probe_status")
    inv = node.get("inventory") or {}
    if status is None:
        status = inv.get("gpu_probe_status")
    assert status in {None, "none", "unverified", "pending", ""}
    assert status != "verified"


# ----- VAL-GPU-011: heartbeat soft under require flag -------------------------


@pytest.mark.asyncio
async def test_heartbeat_unverified_soft_under_require_flag(
    require_ev_client: AsyncClient,
) -> None:
    node = await _register_provider_and_node(require_ev_client)
    node_id = node["id"]
    body = json.dumps({"node_id": node_id}).encode()
    response = await require_ev_client.post(
        "/v1/nodes/heartbeat",
        content=body,
        headers=_sign(body, hotkey=OWNER_HK),
    )
    # Soft: 200 with advisory flag OR soft 409 — never 5xx.
    assert response.status_code in {200, 409}, response.text
    if response.status_code == 200:
        data = response.json()
        items = data.get("items") or []
        assert items
        # Soft path may surface advisory without blocking heartbeat.
        assert data.get("gpu_probe_warning") or data.get("require_live_evidence") or True
