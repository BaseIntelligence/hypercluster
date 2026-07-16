"""VAL-CROSS-010/011/024: double-rent recover, idle protection, nonce replay.

Cross-area market resilience under pure local sim + SQLite. No live Verda.
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
from hypercluster.sim.cross_market_resilience_auth import (
    CROSS_MARKET_RESILIENCE,
    RENTER2_HK,
    RENTER_HK,
    run_cross_double_rent_recover,
    run_cross_idle_rental_protection,
    run_cross_market_resilience_auth,
    run_cross_nonce_replay_refuse,
)
from hypercluster.sim.orchestration import run_cross_market_resilience_auth_bundle
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT
from hypercluster.sim.scenarios import CROSS_MARKET_RESILIENCE as SCENARIO_NAME
from hypercluster.sim.scenarios import run_scenario

TOKEN = "test-challenge-shared-token"
ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"
runner = CliRunner()


def _spawn_live_api(
    *,
    settings_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    db_name: str = "cross-mra.sqlite3",
) -> Any:
    """Start combined-worker API in mission port band; yield context dict."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings, clear_settings_cache

    db_path = tmp_path / db_name
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


@pytest.fixture
def live_cross_mra_api(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Resilience/auth path API (double-rent, idle, nonce)."""

    yield from _spawn_live_api(
        settings_factory=settings_factory,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        db_name="cross-mra.sqlite3",
    )


# ----- VAL-CROSS-010 double-rent recover --------------------------------------


def test_cross_double_rent_reject_then_second_offer_path(
    live_cross_mra_api: dict[str, Any],
) -> None:
    """VAL-CROSS-010: rent1 2xx, rent2 4xx, after terminate + re-list rent3 2xx."""

    result = run_cross_double_rent_recover(
        live_cross_mra_api["base_url"],
        shared_token=live_cross_mra_api["token"],
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "double-rent" in joined
    assert "reject" in joined or "4xx" in joined or "http 4" in joined
    assert "second rent" in joined or "rent3" in joined or "re-rent" in joined
    assert "terminate" in joined


# ----- VAL-CROSS-011 idle reclaim protection ----------------------------------


def test_cross_active_rental_survives_idle_sweep(
    live_cross_mra_api: dict[str, Any],
) -> None:
    """VAL-CROSS-011: active lease/pod survives idle-only health reclaim tick."""

    result = run_cross_idle_rental_protection(
        live_cross_mra_api["base_url"],
        shared_token=live_cross_mra_api["token"],
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "idle" in joined or "reclaim" in joined
    assert "active" in joined
    assert "survived" in joined or "protected" in joined or "short-circuit" in joined


# ----- VAL-CROSS-024 nonce replay refuse --------------------------------------


def test_cross_nonce_replay_cannot_double_create_or_rent(
    live_cross_mra_api: dict[str, Any],
) -> None:
    """VAL-CROSS-024: replay identical nonce rejects; stable job/lease counts."""

    result = run_cross_nonce_replay_refuse(
        live_cross_mra_api["base_url"],
        shared_token=live_cross_mra_api["token"],
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "nonce" in joined
    assert "replay" in joined
    assert "job" in joined
    assert "rent" in joined or "lease" in joined
    # Resource counts remain stable (exact wording from runner steps).
    assert "stable" in joined or "count" in joined


# ----- Combined runner + CLI + dispatch ---------------------------------------


def test_run_cross_market_resilience_auth_combined(
    live_cross_mra_api: dict[str, Any],
) -> None:
    """Combined 010+011+024 runner."""

    result = run_cross_market_resilience_auth(
        live_cross_mra_api["base_url"],
        shared_token=live_cross_mra_api["token"],
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    assert result.name == CROSS_MARKET_RESILIENCE
    assert SCENARIO_NAME == "cross-market-resilience-auth"
    text = "\n".join(result.summary_lines()).lower()
    assert "double-rent" in text or "010" in text or "recover" in text
    assert "idle" in text or "011" in text
    assert "nonce" in text or "024" in text


def test_run_scenario_dispatches_cross_market_resilience(
    live_cross_mra_api: dict[str, Any],
) -> None:
    """Dispatcher routes cross-market-resilience-auth to combined runner."""

    result = run_scenario(
        "cross-market-resilience-auth",
        live_cross_mra_api["base_url"],
        shared_token=live_cross_mra_api["token"],
        timeout=60.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    assert result.name == CROSS_MARKET_RESILIENCE


def test_run_cross_market_resilience_auth_bundle(
    live_cross_mra_api: dict[str, Any],
) -> None:
    """Orchestration bundle wrapper."""

    result = run_cross_market_resilience_auth_bundle(
        live_cross_mra_api["base_url"],
        shared_token=live_cross_mra_api["token"],
        timeout=60.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())


def test_cli_sim_run_scenario_cross_market_resilience(
    live_cross_mra_api: dict[str, Any],
) -> None:
    """CLI: sim run-scenario --name cross-market-resilience-auth."""

    r = runner.invoke(
        cli_app,
        [
            "sim",
            "run-scenario",
            "--name",
            "cross-market-resilience-auth",
            "--url",
            live_cross_mra_api["base_url"],
        ],
        env={
            **dict(__import__("os").environ),
            "CHALLENGE_SHARED_TOKEN": live_cross_mra_api["token"],
        },
    )
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output or "pass" in r.output.lower()


def test_hotkeys_distinct_for_double_rent_path() -> None:
    """Sanity: two capitalist renters + provider are distinct identities."""

    assert RENTER_HK != RENTER2_HK
    assert "renter" in RENTER_HK
    assert "renter2" in RENTER2_HK or "renter-2" in RENTER2_HK or "2" in RENTER2_HK
