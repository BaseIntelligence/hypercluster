"""VAL-CROSS-001/002/003/008/009/013: marketplace→rent→job→score→weights happy path.

Pure local sim + SQLite. No live Verda. Exercises baseline identity, causal
hotkey/weight 1:1, demand+provider auth continuity, and egress cleanliness.
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
from hypercluster.sim.cross_happy_path import (
    DEMAND_HK,
    FOREIGN_HK,
    PROVIDER_HK,
    EgressTrace,
    capture_httpx_egress,
    probe_baseline_identity,
    run_cross_happy_path,
)
from hypercluster.sim.orchestration import run_cross_happy_path_bundle
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT
from hypercluster.sim.scenarios import CROSS_HAPPY_PATH, run_scenario

TOKEN = "test-challenge-shared-token"
ALLOWED_IMAGE = (
    "sha256:sim000000000000000000000000000000000000000000000000000000000001"
)
runner = CliRunner()


@pytest.fixture
def live_cross_api(
    settings_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Combined-worker API for full marketplace→job→score chain (mission band)."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings, clear_settings_cache

    db_path = tmp_path / "cross-happy.sqlite3"
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
        sim_job_run_sleep_s=0.0,
        sim_auto_capacity=True,
        score_window_attempts=50,
        self_deal_damping=0.5,
        weight_push_enabled=False,
    )
    fastapi_app = create_app(settings, hyper_settings=hyper)

    bound_port: int | None = None
    for candidate in range(MIN_MISSION_PORT, MAX_MISSION_PORT + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            bound_port = candidate
            break
    if bound_port is None:
        pytest.skip("no free port in mission band 3200–3299")

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

    try:
        yield {
            "base_url": base_url,
            "port": bound_port,
            "token": TOKEN,
            "db_path": db_path,
        }
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        clear_settings_cache()


# ----- VAL-CROSS-001 baseline identity ---------------------------------------


def test_baseline_identity_health_version_ready(
    live_cross_api: dict[str, Any],
) -> None:
    """VAL-CROSS-001: /health /version /ready all 200 before drives."""

    report, steps, codes = probe_baseline_identity(live_cross_api["base_url"])
    assert report.ok is True, "\n".join(steps + report.errors)
    assert codes.get("health") == 200
    assert codes.get("version") == 200
    assert codes.get("ready") == 200


# ----- VAL-CROSS-002/003 happy path + causal 1:1 ------------------------------


def test_cross_happy_path_marketplace_job_score_weights(
    live_cross_api: dict[str, Any],
) -> None:
    """VAL-CROSS-002/003: full chain with causal demand weight only."""

    result = run_cross_happy_path(
        live_cross_api["base_url"],
        shared_token=live_cross_api["token"],
        poll_timeout_s=25.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    text = "\n".join(result.summary_lines()).lower()
    assert "timeline" in text
    assert "provider_id=" in text
    assert "node_id=" in text
    assert "offer_id=" in text
    assert "lease_id=" in text
    assert "job_id=" in text
    assert "score_id=" in text or "composite=" in text
    assert "weights" in text
    assert DEMAND_HK.lower() in text or "demand" in text
    # Causal: no foreign weight ball.
    assert "foreign hotkey gained" not in text


def test_run_scenario_dispatches_cross_happy_path(
    live_cross_api: dict[str, Any],
) -> None:
    """Dispatcher routes cross-happy-path to full causal runner."""

    result = run_scenario(
        CROSS_HAPPY_PATH,
        live_cross_api["base_url"],
        shared_token=live_cross_api["token"],
        timeout=45.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    assert result.name == "cross-happy-path"


def test_run_cross_happy_path_bundle(
    live_cross_api: dict[str, Any],
) -> None:
    """Orchestration bundle wrapper returns green ScenarioResult."""

    result = run_cross_happy_path_bundle(
        live_cross_api["base_url"],
        shared_token=live_cross_api["token"],
        timeout=45.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())


# ----- VAL-CROSS-008/009 auth continuity -------------------------------------


def test_auth_continuity_messages_in_happy_path(
    live_cross_api: dict[str, Any],
) -> None:
    """VAL-CROSS-008/009: foreign terminate/cancel/results refused; demand listed."""

    result = run_cross_happy_path(
        live_cross_api["base_url"],
        shared_token=live_cross_api["token"],
        poll_timeout_s=25.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "foreign terminate refused" in joined
    assert "foreign cancel refused" in joined
    assert "foreign results refused" in joined
    assert "demand list isolation" in joined
    assert "provider results ok" in joined or "provider results" in joined
    assert PROVIDER_HK  # hotkey constants remain stable for black-box evidence
    assert FOREIGN_HK
    assert DEMAND_HK


# ----- VAL-CROSS-013 no Verda egress ----------------------------------------


def test_no_verda_egress_during_pure_sim(
    live_cross_api: dict[str, Any],
) -> None:
    """VAL-CROSS-013: httpx egress trace clean of Verda hosts."""

    result = run_cross_happy_path(
        live_cross_api["base_url"],
        shared_token=live_cross_api["token"],
        poll_timeout_s=25.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    text = "\n".join(result.summary_lines()).lower()
    assert "no verda" in text
    assert "verda.com" not in text or "verda_hits=0" in text


def test_egress_trace_detects_verda_host() -> None:
    """Unit: EgressTrace flags verda hosts when injected."""

    trace = EgressTrace()
    with capture_httpx_egress(trace):
        # Do not actually hit Verda — craft a failed request to a verda-shaped URL
        # that never leaves the client if DNS does not resolve; still record URL.
        try:
            httpx.get("https://api.verda.com/v1/instances", timeout=0.01)
        except Exception:  # noqa: BLE001 — expected network failure
            pass
    assert not trace.verda_clean
    assert any("verda" in r.host or "verda" in r.url.lower() for r in trace.requests)


# ----- CLI surface ------------------------------------------------------------


def test_cli_sim_run_scenario_cross_happy_path(
    live_cross_api: dict[str, Any],
) -> None:
    """CLI: sim run-scenario --name cross-happy-path exits 0 under combined worker."""

    r = runner.invoke(
        cli_app,
        [
            "sim",
            "run-scenario",
            "--name",
            "cross-happy-path",
            "--url",
            live_cross_api["base_url"],
        ],
        env={
            **dict(__import__("os").environ),
            "CHALLENGE_SHARED_TOKEN": live_cross_api["token"],
        },
    )
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output or "pass" in r.output.lower()
