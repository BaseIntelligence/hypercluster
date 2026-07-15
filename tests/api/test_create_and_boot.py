"""VAL-SCAF-014..017: create_challenge_app wiring and startup guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from base.challenge_sdk.app_factory import create_challenge_app as sdk_create_challenge_app
from base.challenge_sdk.version import API_VERSION, SDK_CONTRACT_VERSION
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


def test_create_app_uses_base_create_challenge_app(settings_factory) -> None:
    """VAL-SCAF-014: app is constructed via Base SDK create_challenge_app."""

    from hypercluster.app import create_app

    real_factory = sdk_create_challenge_app
    calls: list[dict[str, Any]] = []

    def _spy(**kwargs: Any) -> FastAPI:
        calls.append(kwargs)
        return real_factory(**kwargs)

    with patch("hypercluster.app.create_challenge_app", side_effect=_spy) as mocked:
        app = create_app(settings_factory())

    assert mocked.called
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["settings"].slug == "hypercluster"
    assert kwargs["database"] is not None
    assert kwargs["public_router"] is not None
    assert callable(kwargs["get_weights_fn"])
    assert isinstance(app, FastAPI)
    # SDK identity routes installed by factory.
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/health" in paths
    assert "/ready" in paths
    assert "/version" in paths


@pytest.mark.asyncio
async def test_app_boots_and_serves_identity(app_client: AsyncClient) -> None:
    """VAL-SCAF-014: process serves SDK identity routes with hypercluster slug."""

    health = await app_client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["slug"] == "hypercluster"
    assert body["role"] == "challenge"
    assert "status" in body
    assert "version" in body
    assert "ready" in body
    assert isinstance(body.get("capabilities"), list)
    assert isinstance(body.get("checks"), list)

    ready = await app_client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True

    version = await app_client.get("/version")
    assert version.status_code == 200
    vbody = version.json()
    assert vbody["challenge_slug"] == "hypercluster"
    assert vbody["role"] == "challenge"
    assert vbody["api_version"] == API_VERSION
    assert vbody["sdk_version"] == SDK_CONTRACT_VERSION


def test_startup_rejects_missing_shared_token(settings_factory, tmp_path: Path) -> None:
    """VAL-SCAF-015: missing shared token refuses start (lifespan RuntimeError)."""

    from hypercluster.app import create_app

    # Neither env token nor readable secret file.
    settings = settings_factory(shared_token=None, shared_token_file=str(tmp_path / "missing"))
    # Settings model requires token or file path non-empty; we pass a file path that
    # does not exist so load_shared_token returns None at lifespan (not settings ctor).
    app = create_app(settings)

    with pytest.raises(RuntimeError, match="authentication secret is missing or empty"):
        with TestClient(app):
            pass  # entering context runs lifespan


def test_startup_rejects_empty_token_file(settings_factory, tmp_path: Path) -> None:
    """VAL-SCAF-015: empty token file also refuses start."""

    from hypercluster.app import create_app

    empty = tmp_path / "empty_token"
    empty.write_text("", encoding="utf-8")
    settings = settings_factory(shared_token=None, shared_token_file=str(empty))
    app = create_app(settings)

    with pytest.raises(RuntimeError, match="authentication secret is missing or empty"):
        with TestClient(app):
            pass


def test_startup_enforces_api_version_match(settings_factory) -> None:
    """VAL-SCAF-016: api_version mismatch raises ValueError before bind."""

    from hypercluster.settings import Settings

    with pytest.raises(ValueError, match="Incompatible API version"):
        Settings(
            slug="hypercluster",
            name="Hypercluster",
            version="0.1.0",
            api_version="9.9",  # deliberate mismatch vs Base package
            sdk_version=SDK_CONTRACT_VERSION,
            shared_token="secret",
            shared_token_file=None,
        )


def test_startup_enforces_sdk_version_match(settings_factory) -> None:
    """VAL-SCAF-016: sdk_version mismatch raises ValueError before bind."""

    from hypercluster.settings import Settings

    with pytest.raises(ValueError, match="Incompatible SDK version"):
        Settings(
            slug="hypercluster",
            name="Hypercluster",
            version="0.1.0",
            api_version=API_VERSION,
            sdk_version="9.9.9",
            shared_token="secret",
            shared_token_file=None,
        )


def test_create_challenge_app_rejects_version_mismatch_directly(settings_factory) -> None:
    """VAL-SCAF-016: factory hard-checks versions even if settings bypassed.

    Constructing Settings already validates; also assert the factory raises when
    given mismatched fields on a monkeypatched settings object.
    """

    from hypercluster.api.public import router
    from hypercluster.db.database import Database
    from hypercluster.weights import get_weights

    settings = settings_factory()
    # Mutate after construction to simulate desync and hit factory guards.
    object.__setattr__(settings, "api_version", "0.0")

    with pytest.raises(ValueError, match="Incompatible challenge API version"):
        sdk_create_challenge_app(
            settings=settings,
            database=Database(settings.database_url),
            public_router=router,
            get_weights_fn=get_weights,
        )


@pytest.mark.asyncio
async def test_database_lifespan_init_then_close_ordering(
    settings_factory,
    tmp_path: Path,
) -> None:
    """VAL-SCAF-017: lifespan calls database.init before serve and close on stop."""

    from hypercluster.app import create_app
    from hypercluster.db.database import Database

    db_path = tmp_path / "data" / "challenge.sqlite3"
    settings = settings_factory(database_url=f"sqlite+aiosqlite:///{db_path}")
    app = create_app(settings)
    database: Database = app.state.database

    assert database.initialized is False
    assert database.closed is False

    # httpx ASGITransport does not run lifespan; enter it explicitly.
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Lifespan entered → init ran; ready should succeed.
            assert database.initialized is True
            assert database.closed is False
            ready = await client.get("/ready")
            assert ready.status_code == 200
            assert ready.json()["ready"] is True
            # SQLite parent path created and accepts ops post-init.
            assert db_path.parent.is_dir()
            assert await database.healthcheck() is True

    # Context exit triggers close.
    assert database.closed is True
    assert await database.healthcheck() is False


def test_database_lifespan_ordering_with_testclient(settings_factory, tmp_path: Path) -> None:
    """VAL-SCAF-017: TestClient start/stop mirrors init→serve→close."""

    from hypercluster.app import create_app
    from hypercluster.db.database import Database

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'challenge.sqlite3'}"
    )
    app = create_app(settings)
    database: Database = app.state.database

    assert database.initialized is False
    with TestClient(app) as client:
        assert database.initialized is True
        assert database.closed is False
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True
    assert database.closed is True
