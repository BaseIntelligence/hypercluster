"""FastAPI application entrypoint: Base `create_challenge_app` wiring."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from base.challenge_sdk.app_factory import create_challenge_app
from fastapi import APIRouter, FastAPI

from hypercluster.api.public import router as default_public_router
from hypercluster.db.database import Database
from hypercluster.settings import (
    HyperSettings,
    Settings,
    get_hyper_settings,
    get_settings,
)
from hypercluster.weights import get_weights

BackgroundTaskFactory = Callable[[FastAPI], Coroutine[Any, Any, None]]


async def _scaffold_combined_worker_loop(
    app: FastAPI,
    *,
    interval_seconds: float = 5.0,
) -> None:
    """Combined-mode loop: drain job lifecycle (place/launch/collect/score).

    Keeps the SDK `worker` readiness probe green while advancing admitted
    jobs under local sim (VAL-JOB-006/017). Interval and sim run sleep come
    from HyperSettings on app.state.
    """

    interval = interval_seconds if interval_seconds >= 0.05 else 0.05
    try:
        while True:
            await _drain_job_lifecycle(app)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        raise


async def _drain_job_lifecycle(app: FastAPI) -> None:
    """Run one job-lifecycle drain tick against app.state.database."""

    from hypercluster.domain.job_lifecycle import drain_jobs_once

    database = getattr(app.state, "database", None)
    if database is None:
        return
    hyper = getattr(app.state, "hyper_settings", None)
    run_sleep = 0.0
    if hyper is not None:
        run_sleep = float(getattr(hyper, "sim_job_run_sleep_s", 0.0) or 0.0)
    try:
        async with database.session() as session:
            await drain_jobs_once(session, run_sleep_s=run_sleep, limit=16)
    except Exception:  # noqa: BLE001 — never crash the worker loop
        import logging

        logging.getLogger(__name__).exception("combined worker job drain failed")


def create_app(
    settings: Settings | None = None,
    *,
    public_router: APIRouter | None = None,
    background_tasks: Sequence[BackgroundTaskFactory] | None = None,
    hyper_settings: HyperSettings | None = None,
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
    (VAL-SCAF-012 / VAL-SCAF-029). Default remains no worker probe. HYPER_*
    knobs must not break Base identity contracts (VAL-SCAF-028).
    """

    app_settings = settings if settings is not None else get_settings()
    product = hyper_settings if hyper_settings is not None else get_hyper_settings()
    database = Database(app_settings.database_url)

    tasks: list[BackgroundTaskFactory]
    if background_tasks is not None:
        tasks = list(background_tasks)
    elif product.combined_worker:
        interval = product.combined_worker_interval_seconds

        async def _combined_loop(app: FastAPI) -> None:
            await _scaffold_combined_worker_loop(app, interval_seconds=interval)

        tasks = [_combined_loop]
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
    app.state.hyper_settings = product
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
