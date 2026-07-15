"""VAL-SCAF SQLite defaults, CHALLENGE_*/HYPER_* config, public vs internal isolation."""

from __future__ import annotations

import inspect
import stat
from pathlib import Path
from typing import Any

import pytest
from base.challenge_sdk import is_public_route, public_route
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


def _walk_routes(routes: Any) -> list[Any]:
    """Flatten FastAPI route trees (including nested include_router nodes).

    Recent FastAPI versions nest included routers as `_IncludedRouter` with the
    original APIRouter under `.original_router` (not `.routes`).
    """

    found: list[Any] = []
    for route in routes:
        found.append(route)
        nested = getattr(route, "routes", None)
        if nested is not None:
            found.extend(_walk_routes(nested))
        original = getattr(route, "original_router", None)
        if original is not None:
            original_routes = getattr(original, "routes", None)
            if original_routes is not None:
                found.extend(_walk_routes(original_routes))
    return found


def _iter_route_endpoints(app: FastAPI) -> list[Any]:
    """Collect endpoint callables from the FastAPI app (including nested)."""

    endpoints: list[Any] = []
    for route in _walk_routes(app.routes):
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None:
            endpoints.append(endpoint)
    return endpoints


def _public_route_paths(app: FastAPI) -> set[str]:
    paths: set[str] = set()
    for route in _walk_routes(app.routes):
        endpoint = getattr(route, "endpoint", None)
        path = getattr(route, "path", None)
        if endpoint is not None and path is not None and is_public_route(endpoint):
            paths.add(path)
    return paths


def test_default_database_url_is_data_sqlite() -> None:
    """VAL-SCAF-021: default CHALLENGE_DATABASE_URL points at /data/challenge.sqlite3."""

    from hypercluster.settings import Settings, clear_settings_cache

    clear_settings_cache()
    # Construct without env override so Field default wins.
    settings = Settings(
        shared_token="unit-token",
        shared_token_file=None,
        database_url="sqlite+aiosqlite:////data/challenge.sqlite3",
    )
    # Explicit contract string (four leading slashes → absolute /data/...).
    assert settings.database_url == "sqlite+aiosqlite:////data/challenge.sqlite3"
    assert "/data/challenge.sqlite3" in settings.database_url
    # Default on the model also documents the same path for no-override runs.
    field_default = Settings.model_fields["database_url"].default
    assert field_default == "sqlite+aiosqlite:////data/challenge.sqlite3"


def test_settings_default_from_env_without_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-SCAF-021: Settings() resolves default to /data when env unset."""

    from hypercluster.settings import Settings, clear_settings_cache

    monkeypatch.delenv("CHALLENGE_DATABASE_URL", raising=False)
    clear_settings_cache()
    settings = Settings(shared_token="unit-token", shared_token_file=None)
    assert settings.database_url == "sqlite+aiosqlite:////data/challenge.sqlite3"


def test_service_never_requires_base_database_url(
    settings_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-SCAF-022: ready with only CHALLENGE_* DB + token; no BASE_DATABASE_URL."""

    from hypercluster.app import create_app

    monkeypatch.delenv("BASE_DATABASE_URL", raising=False)
    # Ensure product code does not depend on BASE_DATABASE_URL at ready time.
    settings = settings_factory()
    assert not hasattr(settings, "base_database_url")
    assert "BASE_DATABASE_URL" not in (settings.model_dump())

    app = create_app(settings)
    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["ready"] is True
        assert ready.json()["slug"] == "hypercluster"


def test_challenge_env_controls_host_port_database(
    settings_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-SCAF-027: CHALLENGE_HOST / PORT / DATABASE_URL drive settings."""

    from hypercluster.settings import Settings, clear_settings_cache

    alt_db = tmp_path / "alt" / "challenge.sqlite3"
    monkeypatch.setenv(
        "CHALLENGE_DATABASE_URL",
        f"sqlite+aiosqlite:///{alt_db}",
    )
    monkeypatch.setenv("CHALLENGE_HOST", "127.0.0.1")
    monkeypatch.setenv("CHALLENGE_PORT", "3200")
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", "env-token")
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN_FILE", "")
    monkeypatch.delenv("CHALLENGE_SLUG", raising=False)
    clear_settings_cache()

    settings = Settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 3200
    assert settings.database_url.endswith(str(alt_db)) or str(alt_db) in settings.database_url
    assert settings.shared_token == "env-token"


@pytest.mark.asyncio
async def test_challenge_database_url_override_reaches_ready(
    settings_factory,
    tmp_path: Path,
) -> None:
    """VAL-SCAF-027: alternate writable sqlite path still reaches ready."""

    from hypercluster.app import create_app

    alt = tmp_path / "override-data" / "challenge.sqlite3"
    settings = settings_factory(database_url=f"sqlite+aiosqlite:///{alt}")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            ready = await client.get("/ready")
            assert ready.status_code == 200
            assert ready.json()["ready"] is True
            # File (or WAL siblings) should exist under the overridden path parent.
            assert alt.parent.is_dir()
            assert alt.exists() or any(alt.parent.glob("challenge.sqlite3*"))


def test_public_routes_decorated_internal_not_public(settings_factory) -> None:
    """VAL-SCAF-024: sample public routes marked; /internal/* not public set."""

    from hypercluster.app import create_app

    app = create_app(settings_factory())
    public_paths = _public_route_paths(app)

    # Scaffold public surface must advertise marketplace/jobs placeholders.
    assert any(p.endswith("/offers") or p == "/v1/offers" for p in public_paths), public_paths
    assert any(p.endswith("/jobs") or p == "/v1/jobs" for p in public_paths), public_paths

    # Internal weights is present on the app but never public-decorated.
    route_paths = {getattr(r, "path", None) for r in _walk_routes(app.routes)}
    assert "/internal/v1/get_weights" in route_paths
    assert not any(
        path.startswith("/internal") for path in public_paths
    ), f"internal paths leaked into public set: {public_paths}"

    # Explicit: get_weights endpoint callable is not public.
    for route in _walk_routes(app.routes):
        if getattr(route, "path", None) == "/internal/v1/get_weights":
            endpoint = getattr(route, "endpoint", None)
            assert endpoint is not None
            assert is_public_route(endpoint) is False


@pytest.mark.asyncio
async def test_unauthenticated_get_weights_rejected(
    app_client: AsyncClient,
) -> None:
    """VAL-SCAF-025: unauthenticated GET /internal/v1/get_weights is not 200."""

    response = await app_client.get("/internal/v1/get_weights")
    assert response.status_code in {401, 403}
    assert response.status_code != 200


@pytest.mark.asyncio
async def test_authenticated_get_weights_ok(
    app_client: AsyncClient,
    internal_headers: dict[str, str],
) -> None:
    """Positive control for VAL-SCAF-025: bearer + slug header yields weights."""

    response = await app_client.get(
        "/internal/v1/get_weights",
        headers=internal_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["challenge_slug"] == "hypercluster"
    assert isinstance(body["weights"], dict)


@pytest.mark.asyncio
async def test_hyper_knobs_do_not_break_identity(
    settings_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-SCAF-028: HYPER_* flags leave /health /ready /version intact."""

    from hypercluster.app import create_app
    from hypercluster.settings import clear_settings_cache, get_hyper_settings

    monkeypatch.setenv("HYPER_COMBINED_WORKER", "true")
    monkeypatch.setenv("HYPER_TEE_LIVE", "true")
    monkeypatch.setenv("HYPER_TEE_BONUS_TDX", "1.08")
    monkeypatch.setenv("HYPER_WEIGHT_PUSH_INTERVAL_S", "90")
    monkeypatch.setenv("HYPER_SCORE_WINDOW_ATTEMPTS", "25")
    # Avoid process SIGTERM if loop cancelled unpredictably.
    monkeypatch.setattr(
        "base.challenge_sdk.app_factory.signal.raise_signal",
        lambda _signal: None,
    )
    clear_settings_cache()

    # Hyper knobs resolve from env.
    hyper = get_hyper_settings()
    assert hyper.combined_worker is True
    assert hyper.tee_live is True
    assert hyper.tee_bonus_tdx == pytest.approx(1.08)
    assert hyper.weight_push_interval_s == pytest.approx(90.0)
    assert hyper.score_window_attempts == 25

    app = create_app(settings_factory())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            for path in ("/health", "/ready", "/version"):
                response = await client.get(path)
                assert response.status_code == 200, path
                body = response.json()
                assert isinstance(body, dict)
            health = (await client.get("/health")).json()
            version = (await client.get("/version")).json()
            assert health["slug"] == "hypercluster"
            assert health["role"] == "challenge"
            assert version["challenge_slug"] == "hypercluster"
            assert version["role"] == "challenge"
            # Combined worker may add worker probe but must keep required fields.
            assert "status" in health and "ready" in health and "capabilities" in health
            assert "capabilities" in version


@pytest.mark.asyncio
async def test_identity_stable_without_hyper_combined(
    settings_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-SCAF-028: default HYPER off still serves full identity triangle."""

    from hypercluster.app import create_app

    monkeypatch.delenv("HYPER_COMBINED_WORKER", raising=False)
    monkeypatch.delenv("HYPER_TEE_LIVE", raising=False)
    app = create_app(settings_factory())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            health = await client.get("/health")
            ready = await client.get("/ready")
            version = await client.get("/version")
            assert health.status_code == 200
            assert ready.status_code == 200
            assert version.status_code == 200
            # Without combined worker, only database probe is required.
            names = {c["name"] for c in health.json()["checks"]}
            assert "worker" not in names
            assert "database" in names


@pytest.mark.asyncio
async def test_capability_tokens_challenge_role_only(
    app_client: AsyncClient,
) -> None:
    """VAL-SCAF-033: capabilities are unique challenge.* tokens only."""

    health = (await app_client.get("/health")).json()
    version = (await app_client.get("/version")).json()
    for caps in (health["capabilities"], version["capabilities"]):
        assert caps
        assert len(caps) == len(set(caps)), "duplicate capability tokens"
        for token in caps:
            assert token.startswith("challenge."), token
            assert not token.startswith("master.")
            assert not token.startswith("validator.")
            assert not token.startswith("worker.")
    assert sorted(health["capabilities"]) == sorted(version["capabilities"])


@pytest.mark.asyncio
async def test_unwritable_data_path_fails_readiness(
    settings_factory,
    tmp_path: Path,
) -> None:
    """VAL-SCAF-035: unwritable DB parent → database unhealthy + /ready 503."""

    from hypercluster.app import create_app
    from hypercluster.db.database import Database

    locked = tmp_path / "locked-data"
    locked.mkdir()
    db_file = locked / "challenge.sqlite3"
    # Start with a writable path to complete lifespan init, then lock the dir.
    settings = settings_factory(database_url=f"sqlite+aiosqlite:///{db_file}")
    app = create_app(settings)
    try:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                ready_ok = await client.get("/ready")
                assert ready_ok.status_code == 200

                # Pollute: revoke write on /data-equivalent parent.
                locked.chmod(stat.S_IRUSR | stat.S_IXUSR)  # 0o500
                # Drop engine connect path so path-gate is the fail-closed signal.
                database: Database = app.state.database
                assert await database.healthcheck() is False

                ready = await client.get("/ready")
                assert ready.status_code == 503
                body = ready.json()
                assert body["ready"] is False
                db_checks = [c for c in body["checks"] if c["name"] == "database"]
                assert db_checks
                assert db_checks[0]["status"] == "unhealthy"
                assert db_checks[0]["required"] is True
    finally:
        locked.chmod(stat.S_IRWXU)


def test_database_healthcheck_rejects_unwritable_parent(tmp_path: Path) -> None:
    """VAL-SCAF-035 unit: Database.healthcheck fails when parent is unwritable."""

    import asyncio

    from hypercluster.db.database import Database

    locked = tmp_path / "locked"
    locked.mkdir()
    db_file = locked / "challenge.sqlite3"
    db = Database(f"sqlite+aiosqlite:///{db_file}")

    async def _run() -> None:
        await db.init()
        assert await db.healthcheck() is True
        locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            assert await db.healthcheck() is False
        finally:
            locked.chmod(stat.S_IRWXU)
            await db.close()

    asyncio.run(_run())


def test_no_base_database_url_in_source_tree() -> None:
    """VAL-SCAF-022 (static): product sources must not require BASE_DATABASE_URL."""

    root = Path(__file__).resolve().parents[2] / "src" / "hypercluster"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # Mentions only allowed as a reject/forbidden comment without env reads.
        if "BASE_DATABASE_URL" in text:
            # Fail if code reads or sets the env var pattern actually used.
            if (
                'os.environ.get("BASE_DATABASE_URL"' in text
                or "os.environ['BASE_DATABASE_URL']" in text
                or "BASE_DATABASE_URL" in text
                and ("getenv" in text or "Field(" in text or "env=" in text)
            ):
                offenders.append(str(path))
            # Bare string mention in comments/docs of forbidden name is OK if not wired.
            # But product must not define settings fields named this either.
            if "base_database_url" in text.lower() and "class " in text:
                offenders.append(str(path))
    assert offenders == []

    # Behavioral: Settings has no such field.
    from hypercluster.settings import Settings

    assert "base_database_url" not in Settings.model_fields
    assert "BASE_DATABASE_URL" not in {
        f"CHALLENGE_{name.upper()}" for name in Settings.model_fields
    }


def test_scaffold_public_routes_use_public_route_decorator() -> None:
    """VAL-SCAF-024: public.py endpoints carry @public_route marker."""

    from hypercluster.api import public as public_mod

    # Endpoints defined on the module must use the Base marker.
    offers = getattr(public_mod, "list_offers", None)
    jobs = getattr(public_mod, "list_jobs", None)
    assert offers is not None and jobs is not None
    assert is_public_route(offers)
    assert is_public_route(jobs)
    # Decorator helper is the SDK one (marker attribute set).
    assert getattr(offers, "__base_public_route__", False) is True


def test_public_route_decorator_is_base_sdk() -> None:
    """Sanity: we use Base public_route (not a forked marker)."""

    assert callable(public_route)
    src = inspect.getsource(public_route)
    assert "__base_public_route__" in src
