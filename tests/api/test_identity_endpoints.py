"""VAL-SCAF identity endpoints: /health /ready /version (GET/HEAD, contracts)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from base.challenge_sdk.schemas import HealthResponse, VersionResponse
from base.challenge_sdk.version import (
    API_VERSION,
    ARTIFACT_VERSION,
    DISTRIBUTION_NAME,
    RELEASE_ID,
    SDK_CONTRACT_VERSION,
)
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


def _assert_health_contract(body: dict[str, Any], *, expect_ready: bool | None = None) -> None:
    """Validate Base HealthResponse wire shape and hypercluster identity."""

    parsed = HealthResponse.model_validate(body)
    assert parsed.slug == "hypercluster"
    assert parsed.role == "challenge"
    assert parsed.version  # non-empty
    assert parsed.status in {"ok", "degraded", "unhealthy"}
    assert isinstance(parsed.ready, bool)
    assert len(parsed.capabilities) == len(set(parsed.capabilities))
    assert all(isinstance(c, str) for c in parsed.capabilities)
    for check in parsed.checks:
        assert check.name
        assert check.status in {"ok", "degraded", "unhealthy"}
        assert isinstance(check.required, bool)
    if expect_ready is not None:
        assert parsed.ready is expect_ready
    # ready must match required checks
    required_ok = all(c.status == "ok" for c in parsed.checks if c.required)
    assert parsed.ready is required_ok


def _assert_version_contract(body: dict[str, Any]) -> VersionResponse:
    parsed = VersionResponse.model_validate(body)
    assert parsed.distribution_name == DISTRIBUTION_NAME
    assert parsed.artifact_version == ARTIFACT_VERSION
    assert parsed.release_id == RELEASE_ID
    assert parsed.api_version == API_VERSION
    assert parsed.challenge_slug == "hypercluster"
    assert parsed.challenge_version
    assert parsed.sdk_contract_version == SDK_CONTRACT_VERSION
    assert parsed.sdk_version == SDK_CONTRACT_VERSION
    assert parsed.role == "challenge"
    assert len(parsed.capabilities) == len(set(parsed.capabilities))
    return parsed


@pytest.mark.asyncio
async def test_get_health_returns_200_with_base_identity(app_client: AsyncClient) -> None:
    """VAL-SCAF-001 / VAL-SCAF-034 / VAL-SCAF-038: healthy identity HealthResponse."""

    response = await app_client.get("/health")
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")
    body = response.json()
    _assert_health_contract(body, expect_ready=True)
    assert body["status"] == "ok"
    assert body["slug"] == "hypercluster"
    assert body["role"] == "challenge"


@pytest.mark.asyncio
async def test_get_health_ready_true_and_status_ok(app_client: AsyncClient) -> None:
    """VAL-SCAF-002: fully started instance reports ready=true and status=ok."""

    response = await app_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["status"] == "ok"
    required = [c for c in body["checks"] if c["required"]]
    assert required, "expected at least one required check (database)"
    assert all(c["status"] == "ok" for c in required)


@pytest.mark.asyncio
async def test_get_health_always_200_when_not_ready(settings_factory) -> None:
    """VAL-SCAF-003: /health stays 200 even when readiness probes fail."""

    from hypercluster.app import create_app

    app = create_app(settings_factory())
    async with app.router.lifespan_context(app):
        # Force database probe failure while process still serves.
        app.state.database._closed = True  # noqa: SLF001 — test-only probe force
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            health = await client.get("/health")
            assert health.status_code == 200
            body = health.json()
            _assert_health_contract(body, expect_ready=False)
            assert body["ready"] is False
            assert body["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_head_health_allowed(app_client: AsyncClient) -> None:
    """VAL-SCAF-004: HEAD /health is allowed (not 405)."""

    response = await app_client.head("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_ready_200_when_ready(app_client: AsyncClient) -> None:
    """VAL-SCAF-005: /ready 200 + ready=true when probes pass."""

    response = await app_client.get("/ready")
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")
    body = response.json()
    _assert_health_contract(body, expect_ready=True)
    assert body["ready"] is True


@pytest.mark.asyncio
async def test_get_ready_503_when_not_ready(settings_factory) -> None:
    """VAL-SCAF-006: /ready 503 + HealthResponse body when not ready."""

    from hypercluster.app import create_app

    app = create_app(settings_factory())
    async with app.router.lifespan_context(app):
        app.state.database._closed = True  # noqa: SLF001
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            ready = await client.get("/ready")
            assert ready.status_code == 503
            body = ready.json()
            _assert_health_contract(body, expect_ready=False)
            assert body["ready"] is False
            assert body["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_head_ready_mirrors_get_status_codes(settings_factory) -> None:
    """VAL-SCAF-007: HEAD /ready mirrors GET readiness status codes."""

    from hypercluster.app import create_app

    app = create_app(settings_factory())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            get_ready = await client.get("/ready")
            head_ready = await client.head("/ready")
            assert get_ready.status_code == 200
            assert head_ready.status_code == 200

            app.state.database._closed = True  # noqa: SLF001
            get_unready = await client.get("/ready")
            head_unready = await client.head("/ready")
            assert get_unready.status_code == 503
            assert head_unready.status_code == 503


@pytest.mark.asyncio
async def test_get_version_returns_version_response(app_client: AsyncClient) -> None:
    """VAL-SCAF-008 / VAL-SCAF-034: VersionResponse identity fields."""

    response = await app_client.get("/version")
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")
    body = response.json()
    _assert_version_contract(body)


@pytest.mark.asyncio
async def test_version_challenge_version_matches_health(app_client: AsyncClient) -> None:
    """VAL-SCAF-009: /version.challenge_version == /health.version (+ capabilities)."""

    health = await app_client.get("/health")
    version = await app_client.get("/version")
    hbody = health.json()
    vbody = version.json()
    assert vbody["challenge_version"] == hbody["version"]
    assert sorted(vbody["capabilities"]) == sorted(hbody["capabilities"])
    assert vbody["role"] == hbody["role"] == "challenge"


@pytest.mark.asyncio
async def test_head_version_allowed(app_client: AsyncClient) -> None:
    """VAL-SCAF-010: HEAD /version is allowed."""

    response = await app_client.head("/version")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_checks_include_database(app_client: AsyncClient) -> None:
    """VAL-SCAF-011: checks include required database probe."""

    response = await app_client.get("/health")
    body = response.json()
    by_name = {c["name"]: c for c in body["checks"]}
    assert "database" in by_name
    assert by_name["database"]["required"] is True
    assert by_name["database"]["status"] == "ok"

    ready = await app_client.get("/ready")
    r_by_name = {c["name"]: c for c in ready.json()["checks"]}
    assert "database" in r_by_name
    assert r_by_name["database"]["required"] is True


def test_worker_readiness_probe_when_background_tasks_configured(
    settings_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-SCAF-012 / VAL-SCAF-029: worker probe present and fails after exit."""

    from hypercluster.app import create_app

    # Avoid process SIGTERM when background task exits in tests.
    monkeypatch.setattr(
        "base.challenge_sdk.app_factory.signal.raise_signal",
        lambda _signal: None,
    )

    started = asyncio.Event()
    release = asyncio.Event()

    async def worker(_app: FastAPI) -> None:
        started.set()
        await release.wait()

    app = create_app(settings_factory(), background_tasks=(worker,))
    with TestClient(app) as client:
        assert started.is_set()
        ready = client.get("/ready")
        assert ready.status_code == 200
        checks = {c["name"]: c for c in ready.json()["checks"]}
        assert "worker" in checks
        assert checks["worker"]["required"] is True
        assert checks["worker"]["status"] == "ok"

        release.set()
        # Wait until factory observes worker exit and readiness flips.
        status = 200
        for _ in range(100):
            response = client.get("/ready")
            status = response.status_code
            if status == 503:
                body = response.json()
                assert body["ready"] is False
                w = next(c for c in body["checks"] if c["name"] == "worker")
                assert w["status"] == "unhealthy"
                assert w["required"] is True
                break
        assert status == 503


def test_default_app_omits_worker_probe_without_background_tasks(settings_factory) -> None:
    """VAL-SCAF-029 (default): without combined worker, no worker probe."""

    from hypercluster.app import create_app

    app = create_app(settings_factory())
    with TestClient(app) as client:
        body = client.get("/health").json()
        names = {c["name"] for c in body["checks"]}
        assert "worker" not in names
        assert "database" in names


@pytest.mark.asyncio
async def test_mutations_return_503_while_not_ready(settings_factory) -> None:
    """VAL-SCAF-013: non-safe methods get runtime_not_ready while unready."""

    from hypercluster.app import create_app

    # Register a real POST route so when ready the mutation is accepted.
    probe_router = APIRouter()

    @probe_router.post("/v1/probe-mutation")
    async def probe_mutation() -> dict[str, str]:
        return {"ok": "yes"}

    app = create_app(settings_factory(), public_router=probe_router)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Healthy path accepts mutation.
            ok = await client.post("/v1/probe-mutation", json={})
            assert ok.status_code == 200
            assert ok.json()["ok"] == "yes"

            # Force not-ready; identity still works, mutations 503.
            app.state.database._closed = True  # noqa: SLF001
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["ready"] is False
            ready = await client.get("/ready")
            assert ready.status_code == 503

            mutation = await client.post("/v1/probe-mutation", json={})
            assert mutation.status_code == 503
            detail = mutation.json()["detail"]
            assert detail["code"] == "runtime_not_ready"


@pytest.mark.asyncio
async def test_identity_routes_need_no_auth(app_client: AsyncClient) -> None:
    """VAL-SCAF-026: identity triangle works without auth headers."""

    for path in ("/health", "/ready", "/version"):
        response = await app_client.get(path)
        # No Authorization / miner signature headers on app_client.
        assert response.status_code == 200, path
        assert "application/json" in response.headers.get("content-type", ""), path


@pytest.mark.asyncio
async def test_identity_content_type_json(app_client: AsyncClient) -> None:
    """VAL-SCAF-038: GET identity responses are application/json."""

    for path in ("/health", "/ready", "/version"):
        response = await app_client.get(path)
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type
        # Must be parseable JSON object.
        body = response.json()
        assert isinstance(body, dict)


@pytest.mark.asyncio
async def test_role_always_challenge_on_identity(app_client: AsyncClient) -> None:
    """VAL-SCAF-034: role is always challenge on /health and /version."""

    health = (await app_client.get("/health")).json()
    version = (await app_client.get("/version")).json()
    assert health["role"] == "challenge"
    assert version["role"] == "challenge"
    # Never claim master/validator/worker for this service.
    assert health["role"] not in {"master", "validator", "worker"}
    assert version["role"] not in {"master", "validator", "worker"}


def test_hyper_combined_worker_env_registers_worker_probe(
    settings_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-SCAF-029: HYPER_COMBINED_WORKER=true wires background worker + probe."""

    from hypercluster.app import create_app

    monkeypatch.setenv("HYPER_COMBINED_WORKER", "true")
    # Avoid SIGTERM if worker somehow exits.
    monkeypatch.setattr(
        "base.challenge_sdk.app_factory.signal.raise_signal",
        lambda _signal: None,
    )

    app = create_app(settings_factory())
    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        checks = {c["name"]: c for c in ready.json()["checks"]}
        assert "worker" in checks
        assert checks["worker"]["status"] == "ok"
        assert checks["worker"]["required"] is True
