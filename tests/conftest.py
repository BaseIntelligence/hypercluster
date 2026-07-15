"""Shared pytest fixtures for hypercluster challenge tests."""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

# Isolate process-level settings before importing application modules.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="hypercluster-tests-"))
_TEST_DB = _TEST_ROOT / "challenge.sqlite3"
_TEST_TOKEN = "test-challenge-shared-token"

os.environ["CHALLENGE_DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB}"
os.environ["CHALLENGE_SHARED_TOKEN"] = _TEST_TOKEN
os.environ["CHALLENGE_SHARED_TOKEN_FILE"] = ""
# Avoid picking up host secrets / env drift for version fields.
os.environ.setdefault("CHALLENGE_SLUG", "hypercluster")


@pytest.fixture
def test_token() -> str:
    return _TEST_TOKEN


@pytest.fixture
def tmp_db_url(tmp_path: Path) -> str:
    db_path = tmp_path / "challenge.sqlite3"
    return f"sqlite+aiosqlite:///{db_path}"


@pytest.fixture
def settings_factory(tmp_path: Path):
    """Build Settings with isolated temp DB / token (no process env pollution)."""

    from hypercluster.settings import Settings

    def _make(**overrides: object) -> Settings:
        defaults: dict[str, object] = {
            "slug": "hypercluster",
            "name": "Hypercluster",
            "version": "0.1.0",
            "database_url": f"sqlite+aiosqlite:///{tmp_path / 'challenge.sqlite3'}",
            "shared_token": _TEST_TOKEN,
            "shared_token_file": None,
        }
        defaults.update(overrides)
        return Settings(**defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture
async def app_client(settings_factory) -> AsyncIterator[AsyncClient]:
    """ASGI client with lifespan (init like production boot)."""

    from hypercluster.app import create_app

    app = create_app(settings_factory())
    # httpx ASGITransport does not run lifespan; enter it explicitly.
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


@pytest.fixture
def internal_headers(test_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {test_token}",
        "X-Base-Challenge-Slug": "hypercluster",
    }


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()
    yield
    clear_settings_cache()
