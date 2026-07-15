"""VAL-MKT-030: marketplace sim scenario covers offer/rent/terminate + double-rent."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

from hypercluster.cli import app as cli_app
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT
from hypercluster.sim.scenarios import (
    MARKETPLACE,
    run_marketplace_scenario,
    run_scenario,
)

TOKEN = "test-challenge-shared-token"
runner = CliRunner()


@pytest.fixture
def live_marketplace_api(settings_factory, tmp_path: Path) -> Any:
    """Start uvicorn on a free mission-band port with insecure signatures enabled."""

    import socket

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'mkt-sim.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        node_liveness_seconds=120,
        max_offer_price_per_hour=100.0,
        max_offer_lifetime_hours=168.0,
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

    deadline = time.time() + 15.0
    import httpx

    base_url = f"http://127.0.0.1:{bound_port}"
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
        yield {"base_url": base_url, "port": bound_port, "token": TOKEN}
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_run_marketplace_scenario_library(live_marketplace_api: dict[str, Any]) -> None:
    """VAL-MKT-030 library path: offer/rent/double-rent/terminate against live API."""

    result = run_marketplace_scenario(
        live_marketplace_api["base_url"],
        shared_token=live_marketplace_api["token"],
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    assert result.name == MARKETPLACE
    text = "\n".join(result.summary_lines()).lower()
    assert "offer create" in text
    assert "rent" in text
    assert "double-rent" in text
    assert "terminate" in text
    assert "list offers" in text
    assert "pass" in text


def test_run_scenario_dispatcher_marketplace(
    live_marketplace_api: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatcher name 'marketplace' routes to the full local sim flow."""

    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", live_marketplace_api["token"])
    result = run_scenario(
        "marketplace",
        live_marketplace_api["base_url"],
        shared_token=live_marketplace_api["token"],
    )
    assert result.ok is True, "\n".join(result.summary_lines())


def test_cli_sim_run_scenario_marketplace(
    live_marketplace_api: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-MKT-030 CLI: hypercluster sim run-scenario --name marketplace."""

    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", live_marketplace_api["token"])
    base_url = live_marketplace_api["base_url"]
    result = runner.invoke(
        cli_app,
        ["sim", "run-scenario", "--name", "marketplace", "--url", base_url],
    )
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "marketplace" in out
    assert "pass" in out or "result=pass" in out
    assert "double-rent" in out
    assert "terminate" in out
    assert base_url in result.output or "127.0.0.1" in result.output


def test_marketplace_scenario_fails_when_api_down() -> None:
    """Marketplace scenario must not pass against an unreachable API."""

    result = run_marketplace_scenario(
        "http://127.0.0.1:3294",
        timeout=0.5,
        shared_token=TOKEN,
    )
    assert result.ok is False
    assert result.name == MARKETPLACE
