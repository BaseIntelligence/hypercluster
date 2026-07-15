"""FastAPI application entrypoint: Base `create_challenge_app` wiring."""

from __future__ import annotations

from typing import Any

from base.challenge_sdk.app_factory import create_challenge_app
from fastapi import FastAPI

from hypercluster.api.public import router
from hypercluster.db.database import Database
from hypercluster.settings import Settings, get_settings
from hypercluster.weights import get_weights


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the challenge FastAPI app via Base SDK factory.

    Uses `create_challenge_app` so startup enforces:
    - shared challenge token presence (lifespan `RuntimeError` if missing)
    - api_version / sdk_version match with the pinned Base package (`ValueError`)
    - database.init() before serving and database.close() on shutdown
    """

    app_settings = settings if settings is not None else get_settings()
    database = Database(app_settings.database_url)
    app = create_challenge_app(
        settings=app_settings,
        database=database,
        public_router=router,
        get_weights_fn=get_weights,
    )
    app.state.settings = app_settings
    app.state.database = database
    return app


_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Return the process app, creating settings only on first use."""

    global _app
    if _app is None:
        _app = create_app()
    return _app


def reset_app_for_tests() -> None:
    """Clear the cached process app (tests only)."""

    global _app
    _app = None


def __getattr__(name: str) -> Any:
    # Support `uvicorn hypercluster.app:app` without forcing settings at import
    # time (tests / tooling can import the module without CHALLENGE_* secrets).
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["create_app", "get_app", "reset_app_for_tests"]
