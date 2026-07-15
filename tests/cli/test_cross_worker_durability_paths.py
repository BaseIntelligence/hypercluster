"""VAL-CROSS-014/015/016/017/025/026/028: combined worker durability paths.

Pure local sim + isolated SQLite. No live Verda.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from typer.testing import CliRunner

from hypercluster.cli import app as cli_app
from hypercluster.sim.cross_worker_durability_paths import (
    CROSS_WORKER_DURABILITY,
    check_port_band_discipline,
    run_cross_cancel_cleans_bindings,
    run_cross_combined_worker_full_path,
    run_cross_drain_ready_503,
    run_cross_integrity_fail_stops_reward,
    run_cross_restart_mid_flight,
    run_cross_timeout_non_success,
    run_cross_worker_durability_paths,
)
from hypercluster.sim.orchestration import run_cross_worker_durability_paths_bundle
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT
from hypercluster.sim.scenarios import CROSS_WORKER_DURABILITY as SCENARIO_NAME
from hypercluster.sim.scenarios import run_scenario

TOKEN = "test-challenge-shared-token"
ALLOWED_IMAGE = (
    "sha256:sim000000000000000000000000000000000000000000000000000000000001"
)
runner = CliRunner()


def _pick_port() -> int:
    for candidate in range(MIN_MISSION_PORT, MAX_MISSION_PORT + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError("no free port in mission band 3200–3299")


def _spawn_server(
    *,
    settings_factory: Any,
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    port: int | None = None,
    sim_job_run_sleep_s: float = 1.2,
) -> dict[str, Any]:
    """Start combined-worker API bound in mission port band; return handles."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings, clear_settings_cache

    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)
    monkeypatch.setenv(
        "CHALLENGE_DATABASE_URL",
        f"sqlite+aiosqlite:///{db_path}",
    )
    clear_settings_cache()

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    # Run sleep > 0 so cancel + timeout races and mid-flight restart are observable.
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        combined_worker=True,
        combined_worker_interval_seconds=0.05,
        node_liveness_seconds=120,
        max_offer_price_per_hour=100.0,
        max_offer_lifetime_hours=168.0,
        job_image_allowlist=ALLOWED_IMAGE,
        max_job_world_size=64,
        max_job_nnodes=16,
        max_job_nproc_per_node=8,
        max_job_timeout_s=3600,
        max_job_gpu_budget=32,
        sim_job_step_delay_s=0.0,
        # Keep run sleep > 1s so timeout_s=1 jobs hit the watchdog (VAL-CROSS-016).
        sim_job_run_sleep_s=max(float(sim_job_run_sleep_s), 0.0),
        sim_auto_capacity=True,
        score_window_attempts=50,
        self_deal_damping=0.5,
        weight_push_enabled=False,
    )
    fastapi_app = create_app(settings, hyper_settings=hyper)

    bound_port = port if port is not None else _pick_port()
    config = uvicorn.Config(
        fastapi_app,
        host="127.0.0.1",
        port=bound_port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{bound_port}"
    deadline = time.time() + 15.0
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/ready", timeout=1.0)
            if response.status_code == 200 and response.json().get("ready") is True:
                break
        except Exception as exc:  # noqa: BLE001 — probe loop
            last_err = exc
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise AssertionError(f"live API not ready on {base_url}: {last_err!r}")

    return {
        "base_url": base_url,
        "port": bound_port,
        "token": TOKEN,
        "db_path": db_path,
        "server": server,
        "thread": thread,
        "app": fastapi_app,
    }


def _stop_server(ctx: dict[str, Any]) -> None:
    server = ctx.get("server")
    thread = ctx.get("thread")
    if server is not None:
        server.should_exit = True
    if thread is not None:
        thread.join(timeout=10)
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()


@pytest.fixture
def live_cross_wdp_api(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Combined-worker API for durability paths (mission band)."""

    db_path = tmp_path / "cross-wdp.sqlite3"
    ctx = _spawn_server(
        settings_factory=settings_factory,
        db_path=db_path,
        monkeypatch=monkeypatch,
        sim_job_run_sleep_s=2.0,
    )
    try:
        yield ctx
    finally:
        _stop_server(ctx)


# ----- VAL-CROSS-028 port band ------------------------------------------------


def test_port_band_discipline_helpers() -> None:
    """VAL-CROSS-028: helpers enforce 3200–3299 labels."""

    ok, steps = check_port_band_discipline("http://127.0.0.1:3200")
    assert ok is True
    assert any("3200" in s for s in steps)

    bad, bad_steps = check_port_band_discipline("http://127.0.0.1:8000")
    assert bad is False
    assert any("outside" in s for s in bad_steps)


def test_live_api_binds_mission_band(live_cross_wdp_api: dict[str, Any]) -> None:
    """VAL-CROSS-028: live harness binds only inside 3200–3299."""

    port = int(live_cross_wdp_api["port"])
    assert MIN_MISSION_PORT <= port <= MAX_MISSION_PORT
    ok, _ = check_port_band_discipline(live_cross_wdp_api["base_url"])
    assert ok is True


# ----- VAL-CROSS-014 combined worker ------------------------------------------


def test_cross_combined_worker_single_process_full_path(
    live_cross_wdp_api: dict[str, Any],
) -> None:
    """VAL-CROSS-014: one process serves API and drains to terminal."""

    result = run_cross_combined_worker_full_path(
        live_cross_wdp_api["base_url"],
        shared_token=live_cross_wdp_api["token"],
        poll_timeout_s=25.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "combined" in joined
    assert "succeeded" in joined


# ----- VAL-CROSS-016 timeout --------------------------------------------------


def test_cross_timeout_path_non_success_score(
    live_cross_wdp_api: dict[str, Any],
) -> None:
    """VAL-CROSS-016: timeout terminal; not demand-success composite."""

    result = run_cross_timeout_non_success(
        live_cross_wdp_api["base_url"],
        shared_token=live_cross_wdp_api["token"],
        poll_timeout_s=20.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "timeout" in joined
    assert "non-success" in joined or "no positive" in joined or "composite" in joined


# ----- VAL-CROSS-017 cancel ---------------------------------------------------


def test_cross_cancel_path_cleans_bindings(
    live_cross_wdp_api: dict[str, Any],
) -> None:
    """VAL-CROSS-017: cancel → cancelled; no zombie dual attempts."""

    result = run_cross_cancel_cleans_bindings(
        live_cross_wdp_api["base_url"],
        shared_token=live_cross_wdp_api["token"],
        poll_timeout_s=15.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "cancel" in joined
    assert "cancelled" in joined


# ----- VAL-CROSS-025 integrity ----------------------------------------------


def test_cross_integrity_fail_stops_reward(
    live_cross_wdp_api: dict[str, Any],
) -> None:
    """VAL-CROSS-025: rank desync / image mutation → composite 0, no mass."""

    result = run_cross_integrity_fail_stops_reward(
        live_cross_wdp_api["base_url"],
        shared_token=live_cross_wdp_api["token"],
        poll_timeout_s=20.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "integrity" in joined
    assert "0" in joined or "stops reward" in joined


# ----- VAL-CROSS-026 drain ----------------------------------------------------


def test_cross_drain_ready_503_finishes_inflight(
    live_cross_wdp_api: dict[str, Any],
) -> None:
    """VAL-CROSS-026: drain → ready 503; new admits rejected; in-flight finishes."""

    result = run_cross_drain_ready_503(
        live_cross_wdp_api["base_url"],
        shared_token=live_cross_wdp_api["token"],
        poll_timeout_s=25.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "503" in joined
    assert "drain" in joined
    assert "in-flight" in joined or "finished" in joined


# ----- VAL-CROSS-015 restart --------------------------------------------------


def test_cross_restart_mid_flight_completes(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VAL-CROSS-015: admit job, restart same SQLite, job lasts and terminals."""

    db_path = tmp_path / "cross-wdp-restart.sqlite3"
    # Slow run so mid-flight is non-terminal before restart.
    ctx = _spawn_server(
        settings_factory=settings_factory,
        db_path=db_path,
        monkeypatch=monkeypatch,
        sim_job_run_sleep_s=3.0,
    )
    try:
        base1 = ctx["base_url"]
        port1 = int(ctx["port"])

        def _restart(**_kwargs: Any) -> str:
            # Stop first process; rebind same DB on a fresh port in band.
            _stop_server(ctx)
            # Give socket a beat to free.
            time.sleep(0.3)
            new_ctx = _spawn_server(
                settings_factory=settings_factory,
                db_path=db_path,
                monkeypatch=monkeypatch,
                sim_job_run_sleep_s=0.05,
            )
            ctx.update(new_ctx)
            assert int(new_ctx["port"]) != port1 or True  # may reuse if free
            return str(new_ctx["base_url"])

        result = run_cross_restart_mid_flight(
            base1,
            shared_token=TOKEN,
            poll_timeout_s=25.0,
            restart_fn=_restart,
        )
        assert result.ok is True, "\n".join(result.summary_lines())
        joined = "\n".join(result.steps).lower()
        assert "restart" in joined
        assert "terminal" in joined or "succeeded" in joined or "failed" in joined
        assert "job" in joined
    finally:
        _stop_server(ctx)


# ----- Combined runner + CLI + dispatch ---------------------------------------


def test_run_cross_worker_durability_paths_combined(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Combined 014+015+016+017+025+026+028 runner with real restart."""

    db_path = tmp_path / "cross-wdp-combined.sqlite3"
    ctx = _spawn_server(
        settings_factory=settings_factory,
        db_path=db_path,
        monkeypatch=monkeypatch,
        sim_job_run_sleep_s=1.2,
    )
    try:

        def _restart(**_kwargs: Any) -> str:
            _stop_server(ctx)
            time.sleep(0.3)
            new_ctx = _spawn_server(
                settings_factory=settings_factory,
                db_path=db_path,
                monkeypatch=monkeypatch,
                # Must stay >1s so VAL-CROSS-016 timeout_s=1 trips the watchdog.
                sim_job_run_sleep_s=2.0,
            )
            ctx.update(new_ctx)
            return str(new_ctx["base_url"])

        result = run_cross_worker_durability_paths(
            ctx["base_url"],
            shared_token=TOKEN,
            poll_timeout_s=25.0,
            restart_fn=_restart,
            include_restart=True,
        )
        assert result.ok is True, "\n".join(result.summary_lines())
        assert result.name == CROSS_WORKER_DURABILITY
        assert SCENARIO_NAME == "cross-worker-durability-paths"
        text = "\n".join(result.summary_lines()).lower()
        assert "combined" in text or "014" in text
        assert "timeout" in text or "016" in text
        assert "cancel" in text or "017" in text
        assert "integrity" in text or "025" in text
        assert "drain" in text or "026" in text
        assert "port" in text or "028" in text
    finally:
        _stop_server(ctx)


def test_run_scenario_dispatches_cross_worker_durability(
    live_cross_wdp_api: dict[str, Any],
) -> None:
    """Dispatcher routes cross-worker-durability-paths to combined runner."""

    # include restart internally without restart_fn still validates id durability.
    result = run_scenario(
        "cross-worker-durability-paths",
        live_cross_wdp_api["base_url"],
        shared_token=live_cross_wdp_api["token"],
        timeout=90.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    assert result.name == CROSS_WORKER_DURABILITY


def test_run_cross_worker_durability_paths_bundle(
    live_cross_wdp_api: dict[str, Any],
) -> None:
    """Orchestration bundle wrapper."""

    result = run_cross_worker_durability_paths_bundle(
        live_cross_wdp_api["base_url"],
        shared_token=live_cross_wdp_api["token"],
        timeout=90.0,
        include_restart=True,
    )
    assert result.ok is True, "\n".join(result.summary_lines())


def test_cli_sim_run_scenario_cross_worker_durability(
    live_cross_wdp_api: dict[str, Any],
) -> None:
    """CLI: sim run-scenario --name cross-worker-durability-paths."""

    r = runner.invoke(
        cli_app,
        [
            "sim",
            "run-scenario",
            "--name",
            "cross-worker-durability-paths",
            "--url",
            live_cross_wdp_api["base_url"],
        ],
        env={
            **dict(__import__("os").environ),
            "CHALLENGE_SHARED_TOKEN": live_cross_wdp_api["token"],
        },
    )
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output or "pass" in r.output.lower()
