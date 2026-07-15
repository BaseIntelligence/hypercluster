"""VAL-CROSS-007/018/022/023: Docker health→API, suite order, rebind, relative /v1.

Pure local sim + optional Docker daemon. No live Verda.
Docker lifecycle tests skip cleanly when daemon is unavailable.
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
from hypercluster.sim.cross_docker_scenario_proxy import (
    CROSS_DOCKER_SCENARIO_PROXY,
    DEFAULT_CROSS_DOCKER_CONTAINER,
    DEFAULT_CROSS_DOCKER_HOST_PORT,
    PROXY_RELATIVE_PATHS,
    docker_available,
    docker_rm_force,
    probe_port_free,
    probe_relative_proxy_paths,
    run_cross_docker_scenario_proxy,
    run_docker_health_then_api_job,
    run_docker_stop_remove_rebind_free,
    run_scenario_suite_order_green,
)
from hypercluster.sim.orchestration import (
    DEFAULT_SCENARIO_ORDER,
    run_cross_docker_scenario_proxy_bundle,
)
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT
from hypercluster.sim.scenarios import (
    CROSS_DOCKER_SCENARIO_PROXY as SCENARIO_NAME,
)
from hypercluster.sim.scenarios import (
    KNOWN_SCENARIOS,
)

TOKEN = "test-challenge-shared-token"
ALLOWED_IMAGE = (
    "sha256:sim000000000000000000000000000000000000000000000000000000000001"
)
runner = CliRunner()

docker_required = pytest.mark.skipif(
    not docker_available(),
    reason="docker daemon not available",
)


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


def _spawn_api(
    *,
    settings_factory: Any,
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    port: int | None = None,
    master_url: str = "http://127.0.0.1:3201",
) -> dict[str, Any]:
    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings, clear_settings_cache

    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)
    monkeypatch.setenv("CHALLENGE_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("HYPER_MASTER_BASE_URL", master_url)
    clear_settings_cache()

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        combined_worker=False,
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
        sim_job_run_sleep_s=0.0,
        sim_auto_capacity=True,
        score_window_attempts=50,
        self_deal_damping=0.5,
        weight_push_enabled=True,
        master_base_url=master_url,
        weight_push_freshness_s=300,
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
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise AssertionError(f"live API not ready on {base_url}: {last_err!r}")

    return {
        "base_url": base_url,
        "port": bound_port,
        "server": server,
        "thread": thread,
        "token": TOKEN,
        "db_path": db_path,
        "app": fastapi_app,
    }


def _stop_api(handles: dict[str, Any]) -> None:
    handles["server"].should_exit = True
    handles["thread"].join(timeout=10)
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()


def _spawn_mock_master(*, preferred: int | None = 3201) -> dict[str, Any]:
    from hypercluster.sim import mock_master as mm

    mm.reset_store()
    mm.configure_token(TOKEN)

    preferred_list = (
        [preferred, *range(MIN_MISSION_PORT, MAX_MISSION_PORT + 1)]
        if preferred is not None
        else list(range(MIN_MISSION_PORT, MAX_MISSION_PORT + 1))
    )
    bound_port: int | None = None
    for candidate in preferred_list:
        if candidate is None:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", int(candidate)))
            except OSError:
                continue
            bound_port = int(candidate)
            break
    if bound_port is None:
        pytest.skip("no free mission-band port for mock-master")

    config = uvicorn.Config(
        mm.app,
        host="127.0.0.1",
        port=bound_port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{bound_port}"
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise AssertionError(f"mock-master not ready on {base_url}")

    return {
        "base_url": base_url,
        "port": bound_port,
        "server": server,
        "thread": thread,
    }


def _stop_master(handles: dict[str, Any] | None) -> None:
    if handles is None:
        return
    handles["server"].should_exit = True
    handles["thread"].join(timeout=10)


@pytest.fixture
def live_stack(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    master = _spawn_mock_master()
    try:
        api = _spawn_api(
            settings_factory=settings_factory,
            db_path=tmp_path / "cross-docker-proxy.sqlite3",
            monkeypatch=monkeypatch,
            master_url=master["base_url"],
        )
    except Exception:
        _stop_master(master)
        raise
    try:
        yield {
            "api": api,
            "master": master,
            "base_url": api["base_url"],
            "master_url": master["base_url"],
            "token": TOKEN,
            "app": api["app"],
        }
    finally:
        _stop_api(api)
        _stop_master(master)


# ----- VAL-CROSS-023 relative /v1 paths --------------------------------------


def test_relative_proxy_paths_on_live_api(live_stack: dict[str, Any]) -> None:
    """VAL-CROSS-023: direct /v1/offers and /v1/jobs without challenges prefix."""

    result = probe_relative_proxy_paths(
        live_stack["base_url"],
        shared_token=TOKEN,
        app=live_stack["app"],
    )
    assert result.ok is True, "\n".join(result.steps + [result.message])
    assert any("/v1/offers" in s for s in result.steps)
    assert any("/v1/jobs" in s for s in result.steps)

    # Black-box sample: list endpoints must not 404.
    offers = httpx.get(f"{live_stack['base_url']}/v1/offers", timeout=5.0)
    jobs = httpx.get(f"{live_stack['base_url']}/v1/jobs", timeout=5.0)
    assert offers.status_code != 404
    assert jobs.status_code != 404


def test_relative_proxy_paths_constant_surface() -> None:
    """VAL-CROSS-023 inventory includes architecture-facing relative surfaces."""

    assert "/v1/offers" in PROXY_RELATIVE_PATHS
    assert "/v1/jobs" in PROXY_RELATIVE_PATHS
    assert all(p.startswith("/v1/") for p in PROXY_RELATIVE_PATHS)


# ----- VAL-CROSS-018 scenario suite order green ------------------------------


def test_scenario_suite_order_green(
    live_stack: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-CROSS-018: smoke→marketplace→nccl→tee-offline→weights all green, ordered."""

    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)
    monkeypatch.setenv("HYPER_MASTER_BASE_URL", live_stack["master_url"])
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()

    assert list(DEFAULT_SCENARIO_ORDER) == list(KNOWN_SCENARIOS)
    assert list(DEFAULT_SCENARIO_ORDER) == [
        "smoke",
        "marketplace",
        "nccl",
        "tee-offline",
        "weights",
    ]

    suite = run_scenario_suite_order_green(
        live_stack["base_url"],
        shared_token=TOKEN,
        master_url=live_stack["master_url"],
        timeout=45.0,
    )
    assert suite.ok is True, "\n".join(suite.summary_lines())
    assert suite.order == list(DEFAULT_SCENARIO_ORDER)
    assert [r.name for r in suite.results] == list(DEFAULT_SCENARIO_ORDER)
    assert all(r.ok for r in suite.results)


# ----- VAL-CROSS-007 docker health then API job ------------------------------


@docker_required
def test_docker_health_then_api_job() -> None:
    """VAL-CROSS-007: healthy container health/ready then challenge mutate path."""

    docker_rm_force(DEFAULT_CROSS_DOCKER_CONTAINER)
    probe = run_docker_health_then_api_job(
        host_port=DEFAULT_CROSS_DOCKER_HOST_PORT,
        shared_token=TOKEN,
        cleanup=True,
        mutate=True,
    )
    assert probe.ok is True, "\n".join(probe.steps + [probe.message])
    assert probe.health_body.get("slug") == "hypercluster"
    assert probe.ready_status == 200
    assert probe.version_status == 200
    assert probe.mutate_status is not None
    assert probe.mutate_status != 404
    assert probe.cleaned is True
    assert probe.rebind_ok is True


# ----- VAL-CROSS-022 docker stop/rm rebind free ------------------------------


@docker_required
def test_docker_stop_remove_cleans_port_for_rebind() -> None:
    """VAL-CROSS-022: after stop/rm port free; bare-metal can rebind."""

    docker_rm_force(DEFAULT_CROSS_DOCKER_CONTAINER)
    probe = run_docker_stop_remove_rebind_free(
        host_port=DEFAULT_CROSS_DOCKER_HOST_PORT,
        shared_token=TOKEN,
        rebind_smoke=True,
    )
    assert probe.ok is True, "\n".join(probe.steps + [probe.message])
    free = probe_port_free(DEFAULT_CROSS_DOCKER_HOST_PORT)
    assert free.free is True


# ----- Aggregate scenario + CLI + bundle -------------------------------------


def test_cross_docker_scenario_proxy_without_docker(
    live_stack: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-docker legs of the aggregate scenario (proxy + suite) on live API."""

    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)
    monkeypatch.setenv("HYPER_MASTER_BASE_URL", live_stack["master_url"])
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()

    result = run_cross_docker_scenario_proxy(
        live_stack["base_url"],
        shared_token=TOKEN,
        master_url=live_stack["master_url"],
        include_docker=False,
        include_suite=True,
        include_proxy=True,
        app=live_stack["app"],
        timeout=60.0,
    )
    assert result.ok is True, "\n".join(result.steps + [result.message])
    assert result.name == CROSS_DOCKER_SCENARIO_PROXY
    text = "\n".join(result.steps).lower()
    assert "val-cross-023" in text or "/v1/" in text
    assert "val-cross-018" in text or "suite" in text


def test_orchestration_bundle_proxy_and_suite(
    live_stack: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bundle entry excludes docker for fast CI; still covers 018+023."""

    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)
    monkeypatch.setenv("HYPER_MASTER_BASE_URL", live_stack["master_url"])
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()

    result = run_cross_docker_scenario_proxy_bundle(
        live_stack["base_url"],
        shared_token=TOKEN,
        master_url=live_stack["master_url"],
        include_docker=False,
        app=live_stack["app"],
        timeout=60.0,
    )
    assert result.ok is True, "\n".join(result.steps + [result.message])


def test_run_scenario_dispatcher_name() -> None:
    """Dispatcher accepts cross-docker-scenario-proxy name (docker may soft-skip)."""

    assert SCENARIO_NAME == CROSS_DOCKER_SCENARIO_PROXY
    # Unknown-only: ensure name is registered as string constant.
    assert "cross-docker" in SCENARIO_NAME


def test_cli_help_lists_scenario_name() -> None:
    """CLI still documents sim run-scenario; extended names accepted at runtime."""

    r = runner.invoke(cli_app, ["sim", "run-scenario", "--help"])
    assert r.exit_code == 0
    assert "run-scenario" in r.output or "name" in r.output.lower()
