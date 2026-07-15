"""FastAPI application entrypoint: Base `create_challenge_app` wiring."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from base.challenge_sdk.app_factory import create_challenge_app
from fastapi import APIRouter, FastAPI

from hypercluster.api.public import router as default_public_router
from hypercluster.db.database import Database
from hypercluster.settings import Settings, get_settings
from hypercluster.weights import get_weights

BackgroundTaskFactory = Callable[[FastAPI], Coroutine[Any, Any, None]]


def _env_flag_true(name: str) -> bool:
    """Parse a boolean-ish environment flag (true/1/yes/on)."""

    raw = os.environ.get(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


async def _scaffold_combined_worker_loop(app: FastAPI) -> None:
    """Lightweight combined-mode loop until real queue/worker lands.

    Keeps the SDK `worker` readiness probe green while the process is up.
    Later milestones replace this with marketplace/job/weight drain loops.
    """

    interval = float(os.environ.get("HYPER_COMBINED_WORKER_INTERVAL_SECONDS", "5") or "5")
    if interval < 0.05:
        interval = 0.05
    try:
        while True:
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        raise


def create_app(
    settings: Settings | None = None,
    *,
    public_router: APIRouter | None = None,
    background_tasks: Sequence[BackgroundTaskFactory] | None = None,
) -> FastAPI:
    """Build the challenge FastAPI app via Base SDK factory.

    Uses `create_challenge_app` so startup enforces:
    - shared challenge token presence (lifespan `RuntimeError` if missing)
    - api_version / sdk_version match with the pinned Base package (`ValueError`)
    - database.init() before serving and database.close() on shutdown
    - identity routes GET/HEAD `/health` `/ready` `/version` with Base shapes
    - mutations rejected with `runtime_not_ready` while readiness fails
    - `worker` readiness probe when `background_tasks` are configured

    `HYPER_COMBINED_WORKER=true` (or explicit `background_tasks`) registers an
    in-process worker task so the factory installs the required worker probe
    (VAL-SCAF-012 / VAL-SCAF-029). Default remains no worker probe.
    """

    app_settings = settings if settings is not None else get_settings()
    database = Database(app_settings.database_url)

    tasks: list[BackgroundTaskFactory]
    if background_tasks is not None:
        tasks = list(background_tasks)
    elif _env_flag_true("HYPER_COMBINED_WORKER"):
        tasks = [_scaffold_combined_worker_loop]
    else:
        tasks = []

    app = create_challenge_app(
        settings=app_settings,
        database=database,
        public_router=public_router if public_router is not None else default_public_router,
        get_weights_fn=get_weights,
        background_tasks=tuple(tasks),
    )
    app.state.settings = app_settings
    app.state.database = database
    app.state.combined_worker_enabled = bool(tasks)
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
