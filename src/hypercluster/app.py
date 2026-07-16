"""FastAPI application entrypoint: Base `create_challenge_app` wiring."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from base.challenge_sdk.app_factory import create_challenge_app
from base.challenge_sdk.health import ReadinessProbe
from fastapi import APIRouter, FastAPI

from hypercluster.api.public import router as default_public_router
from hypercluster.db.database import Database
from hypercluster.settings import (
    HyperSettings,
    Settings,
    get_hyper_settings,
    get_settings,
)
from hypercluster.weights import bind_weights_runtime, get_weights

BackgroundTaskFactory = Callable[[FastAPI], Coroutine[Any, Any, None]]


def _make_drain_state() -> dict[str, bool]:
    """Shared mutable drain flag (VAL-CROSS-026).

    Typed as a plain dict so readiness probes can close over it before the
    FastAPI app instance exists, then the same object is bound on ``app.state``.
    """

    return {"draining": False}


def is_draining(app: FastAPI | None) -> bool:
    """True when the challenge has entered drain mode (ready→503, refuse admits)."""

    if app is None:
        return False
    flag = getattr(app.state, "drain_state", None)
    if isinstance(flag, dict):
        return bool(flag.get("draining"))
    return bool(getattr(app.state, "draining", False))


def set_draining(app: FastAPI, draining: bool) -> bool:
    """Enter/leave drain mode. Returns the new draining value."""

    value = bool(draining)
    flag = getattr(app.state, "drain_state", None)
    if isinstance(flag, dict):
        flag["draining"] = value
    app.state.draining = value
    return value


async def _scaffold_combined_worker_loop(
    app: FastAPI,
    *,
    interval_seconds: float = 5.0,
) -> None:
    """Combined-mode loop: drain job lifecycle + optional weight push tick.

    Keeps the SDK `worker` readiness probe green while advancing admitted
    jobs under local sim (VAL-JOB-006/017). Weight push is best-effort and
    must never block /health (VAL-SCORE-023). Interval and sim run sleep come
    from HyperSettings on app.state.
    """

    interval = interval_seconds if interval_seconds >= 0.05 else 0.05
    push_every = 0.0
    last_push = 0.0
    hyper = getattr(app.state, "hyper_settings", None)
    if hyper is not None:
        push_every = float(getattr(hyper, "weight_push_interval_s", 120.0) or 120.0)
    try:
        while True:
            await _drain_job_lifecycle(app)
            # Cooperative push tick: never await longer than the drain interval
            # in a way that freezes identity surfaces (asyncio single-thread).
            if push_every > 0:
                now = asyncio.get_event_loop().time()
                if now - last_push >= push_every:
                    await _tick_weight_push(app)
                    last_push = now
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
            await drain_jobs_once(
                session,
                run_sleep_s=run_sleep,
                limit=16,
                hyper=hyper,
            )
    except Exception:  # noqa: BLE001 — never crash the worker loop
        import logging

        logging.getLogger(__name__).exception("combined worker job drain failed")


async def _tick_weight_push(app: FastAPI) -> None:
    """One raw-weight push attempt (non-fatal). No on-chain set_weights."""

    client = getattr(app.state, "weight_push_client", None)
    if client is None:
        return
    try:
        await client.push_once()
    except Exception:  # noqa: BLE001 — never crash the worker loop
        import logging

        logging.getLogger(__name__).exception("weight push tick failed")


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

    # M11 optional catalog seed after create_all (HYPER_PRICE_SEED_ON_BOOT).
    # only_if_empty=True never clobbers admin rows (VAL-PRICE-020/021).
    if bool(getattr(product, "price_seed_on_boot", False)):

        async def _price_seed_hook(db: Database) -> None:
            from hypercluster.domain.pricing import maybe_seed_prices_on_boot

            await maybe_seed_prices_on_boot(
                db,
                price_seed_on_boot=True,
                only_if_empty=True,
                source="seed",
            )

        database.add_post_init_hook(_price_seed_hook)

    drain_state = _make_drain_state()

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

    # Bind DB for process-level get_weights (aggregation window → raw map).
    bind_weights_runtime(database, product)

    # Optional dedicated weight-push background task when master is configured
    # and combined worker is off (still non-blocking for /health; VAL-SCORE-023).
    from hypercluster.weight_push import maybe_build_push_client, run_weight_push_loop

    push_client = maybe_build_push_client(
        database=database,
        settings=app_settings,
        hyper=product,
    )
    if push_client is not None and not product.combined_worker and background_tasks is None:
        # Run push loop as its own background task so /health keeps responding.
        interval = float(product.weight_push_interval_s)

        async def _push_loop(app: FastAPI) -> None:
            await run_weight_push_loop(push_client, interval_seconds=interval)

        tasks = list(tasks) + [_push_loop]

    def _not_draining() -> bool:
        # VAL-CROSS-026: drain forces ready=false → mutations 503 /runtime_not_ready.
        return not bool(drain_state.get("draining"))

    readiness_probes = (ReadinessProbe(name="not_draining", check=_not_draining, required=True),)

    app = create_challenge_app(
        settings=app_settings,
        database=database,
        public_router=public_router if public_router is not None else default_public_router,
        get_weights_fn=get_weights,
        background_tasks=tuple(tasks),
        readiness_probes=readiness_probes,
    )
    app.state.settings = app_settings
    app.state.hyper_settings = product
    app.state.database = database
    app.state.combined_worker_enabled = bool(tasks)
    app.state.weight_push_client = push_client
    app.state.drain_state = drain_state
    app.state.draining = False

    # VAL-CROSS-026: while drained, the Base `runtime_not_ready` middleware
    # refuses normal POSTs. Allow /v1/sim/drain itself so operators can leave
    # drain and restore ready=true without restarting the process. Middleware
    # added after create_challenge_app runs first on the request path.
    from fastapi.responses import JSONResponse
    from starlette.requests import Request as StarletteRequest

    @app.middleware("http")
    async def allow_sim_drain_while_unready(
        request: StarletteRequest,
        call_next: Callable[[StarletteRequest], Coroutine[Any, Any, Any]],
    ) -> Any:
        path = request.url.path.rstrip("/") or "/"
        if path == "/v1/sim/drain":
            if request.method == "GET":
                return JSONResponse({"ok": True, "draining": bool(drain_state.get("draining"))})
            if request.method in {"POST", "PUT", "PATCH"}:
                try:
                    payload = await request.json()
                except Exception:  # noqa: BLE001 — empty body = enter drain
                    payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                new_value = bool(payload.get("draining", True))
                drain_state["draining"] = new_value
                app.state.draining = new_value
                return JSONResponse(
                    {
                        "ok": True,
                        "draining": new_value,
                        "was_draining": new_value,
                    }
                )
        return await call_next(request)

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


__all__ = [
    "create_app",
    "get_app",
    "is_draining",
    "reset_app_for_tests",
    "set_draining",
]
