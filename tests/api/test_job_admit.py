"""VAL-JOB-001..005: job admit, static safety, world_size, auth, idempotency."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.api.auth import build_signed_headers

SUBMITTER_HK = "job-submitter-hotkey-aaaaaaaaaaaaaaaaaaaaaaaaaa"
OTHER_HK = "job-other-hotkey-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
TOKEN = "test-challenge-shared-token"

# Default sim image digests allowed by HyperSettings defaults.
ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"
ALT_ALLOWED_IMAGE = "sha256:cafebabecafebabecafebabecafebabecafebabecafebabecafebabecafebabe"
DISALLOWED_IMAGE = "sha256:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def _sign(body: bytes, *, hotkey: str = SUBMITTER_HK, nonce: str | None = None) -> dict[str, str]:
    return build_signed_headers(secret=TOKEN, hotkey=hotkey, body=body, nonce=nonce)


def _valid_job_body(**overrides: Any) -> dict[str, Any]:
    """Well-formed HyperJob body (world_size == nnodes * nproc_per_node)."""

    body: dict[str, Any] = {
        "image_digest": ALLOWED_IMAGE,
        "entrypoint": ["python", "-m", "train"],
        "world_size": 4,
        "nnodes": 2,
        "nproc_per_node": 2,
        "timeout_s": 300,
        "resource": {
            "gpus": 4,
            "nodes": 2,
        },
        "backend": "nccl",
        "fabric": "auto",
        "tee": "none",
    }
    body.update(overrides)
    return body


@pytest.fixture
async def job_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    """App client with job budget settings + insecure signature mode."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'jobs.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        job_image_allowlist=f"{ALLOWED_IMAGE},{ALT_ALLOWED_IMAGE}",
        max_job_world_size=64,
        max_job_nnodes=16,
        max_job_nproc_per_node=8,
        max_job_timeout_s=3600,
        max_job_gpu_budget=32,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


async def _post_job(
    client: AsyncClient,
    body: dict[str, Any],
    *,
    hotkey: str = SUBMITTER_HK,
    signed: bool = True,
) -> Any:
    raw = json.dumps(body).encode()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if signed:
        headers.update(_sign(raw, hotkey=hotkey))
    return await client.post("/v1/jobs", content=raw, headers=headers)


def _job_id(payload: dict[str, Any]) -> str:
    jid = payload.get("id") or payload.get("job_id")
    assert jid, f"expected id or job_id in {payload}"
    return str(jid)


# ----- VAL-JOB-001: admit well-formed HyperJob --------------------------------


@pytest.mark.asyncio
async def test_post_jobs_admits_well_formed_hyperjob(job_client: AsyncClient) -> None:
    """VAL-JOB-001: signed valid HyperJob admitted with stable job id."""

    body = _valid_job_body()
    response = await _post_job(job_client, body)
    assert response.status_code == 200, response.text
    payload = response.json()
    job_id = _job_id(payload)
    assert payload.get("status") in {"submitted", "admitted"}
    assert payload.get("submitter_hotkey") == SUBMITTER_HK
    assert payload.get("world_size") == 4
    assert payload.get("nnodes") == 2
    assert payload.get("nproc_per_node") == 2
    assert payload.get("image_digest") == ALLOWED_IMAGE

    # Visible under GET by id and submitter-scoped list.
    detail = await job_client.get(f"/v1/jobs/{job_id}")
    assert detail.status_code == 200, detail.text
    assert (detail.json().get("id") or detail.json().get("job_id")) == job_id
    assert detail.json().get("submitter_hotkey") == SUBMITTER_HK

    listed = await job_client.get(
        "/v1/jobs",
        headers={"X-Hotkey": SUBMITTER_HK},
    )
    assert listed.status_code == 200, listed.text
    items = listed.json().get("items") or []
    ids = {(i.get("id") or i.get("job_id")) for i in items}
    assert job_id in ids


# ----- VAL-JOB-002: static admit rejects --------------------------------------


@pytest.mark.asyncio
async def test_admit_rejects_disallowed_image(job_client: AsyncClient) -> None:
    """VAL-JOB-002: image_not_allowed for digest outside allowlist."""

    body = _valid_job_body(image_digest=DISALLOWED_IMAGE)
    response = await _post_job(job_client, body)
    assert response.status_code in {400, 422}, response.text
    detail = response.json().get("detail")
    assert isinstance(detail, dict)
    assert detail.get("code") == "image_not_allowed"

    # No job row for submitter.
    listed = await job_client.get("/v1/jobs", headers={"X-Hotkey": SUBMITTER_HK})
    assert listed.status_code == 200
    assert listed.json().get("items") == []


@pytest.mark.asyncio
async def test_admit_rejects_path_unsafe_entrypoint(job_client: AsyncClient) -> None:
    """VAL-JOB-002: path_unsafe for traversal / absolute unsafe entrypoint args."""

    unsafe_cases = [
        ["python", "../evil.py"],
        ["bash", "-c", "cat /etc/passwd"],
        ["/bin/sh", "-c", "rm -rf /"],
        ["python", "/etc/shadow"],
        ["python", "train\n;reboot"],
    ]
    for entrypoint in unsafe_cases:
        body = _valid_job_body(entrypoint=entrypoint)
        response = await _post_job(job_client, body)
        assert response.status_code in {400, 422}, (
            f"entrypoint {entrypoint!r} expected 4xx, got {response.status_code}: {response.text}"
        )
        detail = response.json().get("detail")
        assert isinstance(detail, dict), response.text
        assert detail.get("code") == "path_unsafe", f"entrypoint {entrypoint!r} detail={detail}"

    listed = await job_client.get("/v1/jobs", headers={"X-Hotkey": SUBMITTER_HK})
    assert listed.json().get("items") == []


@pytest.mark.asyncio
async def test_admit_rejects_budget_exceeded(job_client: AsyncClient) -> None:
    """VAL-JOB-002: budget_exceeded when resource/timeout over configured caps."""

    # GPU budget over cap (max_job_gpu_budget=32 for fixture).
    body = _valid_job_body(
        world_size=2,
        nnodes=1,
        nproc_per_node=2,
        resource={"gpus": 64, "nodes": 1},
    )
    response = await _post_job(job_client, body)
    assert response.status_code in {400, 422}, response.text
    detail = response.json().get("detail")
    assert isinstance(detail, dict)
    assert detail.get("code") == "budget_exceeded"

    # Timeout over cap (max_job_timeout_s=3600).
    body = _valid_job_body(timeout_s=999_999)
    response = await _post_job(job_client, body)
    assert response.status_code in {400, 422}, response.text
    detail = response.json().get("detail")
    assert isinstance(detail, dict)
    assert detail.get("code") == "budget_exceeded"

    # world_size over max_job_world_size (need consistent dims too).
    body = _valid_job_body(
        world_size=128,
        nnodes=16,
        nproc_per_node=8,
        resource={"gpus": 16, "nodes": 16},
    )
    response = await _post_job(job_client, body)
    assert response.status_code in {400, 422}, response.text
    detail = response.json().get("detail")
    assert isinstance(detail, dict)
    assert detail.get("code") in {"budget_exceeded", "world_size_over_cap"}

    listed = await job_client.get("/v1/jobs", headers={"X-Hotkey": SUBMITTER_HK})
    assert listed.json().get("items") == []


# ----- VAL-JOB-003: world_size sanity ----------------------------------------


@pytest.mark.asyncio
async def test_world_size_must_equal_nnodes_times_nproc(job_client: AsyncClient) -> None:
    """VAL-JOB-003: inconsistent dims 4xx; consistent 2×2=4 admitted."""

    # Inconsistent: 4 != 2 * 1
    bad = _valid_job_body(
        world_size=4, nnodes=2, nproc_per_node=1, resource={"gpus": 2, "nodes": 2}
    )
    bad_resp = await _post_job(job_client, bad)
    assert bad_resp.status_code in {400, 422}, bad_resp.text
    detail = bad_resp.json().get("detail")
    assert isinstance(detail, dict)
    assert detail.get("code") == "world_size_mismatch"

    # Consistent: 4 = 2 * 2
    good = _valid_job_body(world_size=4, nnodes=2, nproc_per_node=2)
    good_resp = await _post_job(job_client, good)
    assert good_resp.status_code == 200, good_resp.text
    assert _job_id(good_resp.json())


# ----- VAL-JOB-004: unauthenticated reject -----------------------------------


@pytest.mark.asyncio
async def test_unauthenticated_job_create_rejected(job_client: AsyncClient) -> None:
    """VAL-JOB-004: missing/forged auth fails closed; no job rows created."""

    body = _valid_job_body()
    raw = json.dumps(body).encode()

    # No headers at all.
    resp = await job_client.post(
        "/v1/jobs",
        content=raw,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code in {401, 403}, resp.text

    # Hotkey only, no signature.
    resp = await job_client.post(
        "/v1/jobs",
        content=raw,
        headers={
            "Content-Type": "application/json",
            "X-Hotkey": SUBMITTER_HK,
        },
    )
    assert resp.status_code in {401, 403}, resp.text

    # Bad signature.
    headers = _sign(raw, hotkey=SUBMITTER_HK)
    headers["X-Signature"] = "0" * 64
    headers["Content-Type"] = "application/json"
    resp = await job_client.post("/v1/jobs", content=raw, headers=headers)
    assert resp.status_code in {401, 403}, resp.text

    # Replay same nonce (first succeeds; second is replay).
    body2 = _valid_job_body(client_request_id="auth-replay-probe")
    raw2 = json.dumps(body2).encode()
    good_headers = _sign(raw2, hotkey=SUBMITTER_HK, nonce="job-nonce-once-only")
    good_headers["Content-Type"] = "application/json"
    first = await job_client.post("/v1/jobs", content=raw2, headers=good_headers)
    assert first.status_code == 200, first.text
    job_id = _job_id(first.json())

    # Replayed nonce with same body headers → 401.
    replay = await job_client.post("/v1/jobs", content=raw2, headers=good_headers)
    assert replay.status_code in {401, 403}, replay.text
    detail = replay.json().get("detail")
    if isinstance(detail, dict):
        assert detail.get("code") in {"nonce_replay", "invalid_signature", "missing_auth_headers"}

    listed = await job_client.get("/v1/jobs", headers={"X-Hotkey": SUBMITTER_HK})
    assert listed.status_code == 200
    items = listed.json().get("items") or []
    # Only the one successful create; unsigned/forged never added extras.
    assert len(items) == 1
    assert (items[0].get("id") or items[0].get("job_id")) == job_id


# ----- VAL-JOB-005: idempotent create ----------------------------------------


@pytest.mark.asyncio
async def test_idempotent_create_by_hotkey_and_client_request_id(
    job_client: AsyncClient,
) -> None:
    """VAL-JOB-005: same (hotkey, client_request_id) returns same job; different keys distinct."""

    key = "idem-key-alpha-001"
    body = _valid_job_body(client_request_id=key)
    first = await _post_job(job_client, body)
    assert first.status_code == 200, first.text
    first_id = _job_id(first.json())

    # Second POST with same key + same hotkey → same id (no twin).
    second = await _post_job(job_client, body)
    assert second.status_code in {200, 409}, second.text
    second_id = _job_id(second.json())
    assert second_id == first_id
    # Idempotent responses may include a flag.
    if second.status_code == 200:
        # Either prefer original payload or explicit reused flag.
        assert second.json().get("status") in {"submitted", "admitted"}

    # Different key → new job.
    body2 = _valid_job_body(client_request_id="idem-key-beta-002")
    third = await _post_job(job_client, body2)
    assert third.status_code == 200, third.text
    third_id = _job_id(third.json())
    assert third_id != first_id

    # Same key from a different hotkey invents a separate job (key is per-hotkey).
    body3 = _valid_job_body(client_request_id=key)
    other = await _post_job(job_client, body3, hotkey=OTHER_HK)
    assert other.status_code == 200, other.text
    other_id = _job_id(other.json())
    assert other_id not in {first_id, third_id}

    listed = await job_client.get("/v1/jobs", headers={"X-Hotkey": SUBMITTER_HK})
    assert listed.status_code == 200
    items = listed.json().get("items") or []
    submitter_ids = {(i.get("id") or i.get("job_id")) for i in items}
    assert first_id in submitter_ids
    assert third_id in submitter_ids
    assert other_id not in submitter_ids
    # Exactly two jobs for submitter (not three from double-post of same key).
    assert len(submitter_ids) == 2


@pytest.mark.asyncio
async def test_list_jobs_without_hotkey_is_empty(job_client: AsyncClient) -> None:
    """Fail-closed list identity scope (aligns with marketplace leases pattern)."""

    body = _valid_job_body(client_request_id="list-scope-probe")
    created = await _post_job(job_client, body)
    assert created.status_code == 200, created.text

    bare = await job_client.get("/v1/jobs")
    assert bare.status_code == 200
    assert bare.json().get("items") == []
