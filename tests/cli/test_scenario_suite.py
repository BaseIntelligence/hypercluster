"""VAL-CLI-015..019: scenario suite smoke/marketplace/nccl/tee-offline/weights.

All scenarios must exit green under local sim (no live IB / No live Verda).
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

from hypercluster.cli import app as cli_app
from hypercluster.sim.orchestration import (
    DEFAULT_SCENARIO_ORDER,
    SuiteResult,
    run_scenario_suite,
)
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT
from hypercluster.sim.scenarios import (
    KNOWN_SCENARIOS,
    MARKETPLACE,
    NCCL,
    SMOKE,
    TEE_OFFLINE,
    WEIGHTS,
    run_nccl_scenario,
    run_scenario,
    run_smoke_scenario,
    run_tee_offline_scenario,
)

TOKEN = "test-challenge-shared-token"
runner = CliRunner()


@pytest.fixture
def live_scenario_api(settings_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Mission-band API with insecure signature mode for marketplace path."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    db_path = tmp_path / "scenario-suite.sqlite3"
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)
    monkeypatch.setenv(
        "CHALLENGE_DATABASE_URL",
        f"sqlite+aiosqlite:///{db_path}",
    )
    # Clear settings cache so get_settings() matches the live process DB.
    from hypercluster import settings as settings_mod

    if hasattr(settings_mod, "get_settings"):
        cache_clear = getattr(settings_mod.get_settings, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        node_liveness_seconds=120,
        max_offer_price_per_hour=100.0,
        max_offer_lifetime_hours=168.0,
        master_base_url="http://127.0.0.1:3201",
        weight_push_enabled=True,
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
        yield {"base_url": base_url, "port": bound_port, "token": TOKEN, "db_path": db_path}
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if hasattr(settings_mod, "get_settings"):
            cache_clear = getattr(settings_mod.get_settings, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()


@pytest.fixture
def mock_master(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Start mock-master on free mission port (prefer 3201)."""

    from hypercluster.sim import mock_master as mm

    mm.reset_store()
    mm.configure_token(TOKEN)

    bound_port: int | None = None
    preferred = [3201, *range(MIN_MISSION_PORT, MAX_MISSION_PORT + 1)]
    for candidate in preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            bound_port = candidate
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

    deadline = time.time() + 10.0
    import httpx

    base = f"http://127.0.0.1:{bound_port}"
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=0.5).status_code == 200:
                break
        except Exception:  # noqa: BLE001
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise AssertionError(f"mock-master not healthy on {base}")

    monkeypatch.setenv("HYPER_MASTER_BASE_URL", base)
    try:
        yield {"base_url": base, "port": bound_port, "token": TOKEN}
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# ----- VAL-CLI-015 smoke -----------------------------------------------------


def test_smoke_scenario_green_with_empty_weights(
    live_scenario_api: dict[str, Any],
) -> None:
    """VAL-CLI-015: smoke → health/ready + empty weights burn-safe."""

    result = run_smoke_scenario(live_scenario_api["base_url"])
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = " ".join(result.steps).lower() + " " + result.message.lower()
    assert "identity" in joined
    assert "burn" in joined or "weight" in joined


def test_cli_smoke_scenario_exit_0(live_scenario_api: dict[str, Any]) -> None:
    """VAL-CLI-015 CLI: sim run-scenario --name smoke exits 0."""

    r = runner.invoke(
        cli_app,
        ["sim", "run-scenario", "--name", "smoke", "--url", live_scenario_api["base_url"]],
    )
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output or "pass" in r.output.lower()


# ----- VAL-CLI-016 marketplace -----------------------------------------------


def test_marketplace_scenario_includes_double_rent(
    live_scenario_api: dict[str, Any],
) -> None:
    """VAL-CLI-016: marketplace covers offer/rent/terminate + double-rent reject."""

    result = run_scenario(
        MARKETPLACE,
        live_scenario_api["base_url"],
        shared_token=live_scenario_api["token"],
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    text = "\n".join(result.summary_lines()).lower()
    assert "double-rent" in text
    assert "terminate" in text
    assert "offer" in text


# ----- VAL-CLI-017 nccl ------------------------------------------------------


def test_nccl_scenario_pack_spread_and_fail_inject() -> None:
    """VAL-CLI-017: multi-node pack/spread + fabric_gate fail inject (local sim)."""

    result = run_nccl_scenario("http://127.0.0.1:3200")
    assert result.ok is True, "\n".join(result.summary_lines())
    assert result.name == NCCL
    text = "\n".join(result.summary_lines()).lower()
    assert "pack" in text
    assert "spread" in text
    assert "fabric_gate" in text.replace(" ", "_") or "fabric_gate" in text
    assert "fail" in text or "inject" in text or "spoof" in text or "zero" in text


def test_run_scenario_dispatcher_nccl() -> None:
    """Dispatcher routes name=nccl to full offline sim (no real IB)."""

    result = run_scenario(NCCL, "http://127.0.0.1:9")
    assert result.ok is True, result.message
    assert "not implemented" not in result.message.lower()


def test_cli_nccl_scenario_exit_0() -> None:
    """VAL-CLI-017 CLI: sim run-scenario --name nccl exits 0 offline."""

    r = runner.invoke(
        cli_app,
        ["sim", "run-scenario", "--name", "nccl", "--url", "http://127.0.0.1:3200"],
    )
    assert r.exit_code == 0, r.output
    assert "nccl" in r.output.lower()


# ----- VAL-CLI-018 tee-offline -----------------------------------------------


def test_tee_offline_scenario_green_offline() -> None:
    """VAL-CLI-018: tee-offline positive/negative + bonus offline paths."""

    result = run_tee_offline_scenario("http://127.0.0.1:3200")
    assert result.ok is True, "\n".join(result.summary_lines())
    text = " ".join(result.steps).lower()
    assert "positive" in text or "verify" in text
    assert "reject" in text or "mutat" in text or "bonus" in text


# ----- VAL-CLI-019 weights ---------------------------------------------------


def test_weights_scenario_push_ack(
    live_scenario_api: dict[str, Any],
    mock_master: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-CLI-019: multi-hotkey composites → push ack / idempotency."""

    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", live_scenario_api["token"])
    monkeypatch.setenv("HYPER_MASTER_BASE_URL", mock_master["base_url"])
    from hypercluster import settings as settings_mod

    cache_clear = getattr(settings_mod.get_settings, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()

    result = run_scenario(
        WEIGHTS,
        live_scenario_api["base_url"],
        shared_token=live_scenario_api["token"],
        timeout=45.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    text = "\n".join(result.summary_lines()).lower()
    assert "push" in text
    assert "ack" in text or "acknowledged" in text or "sim" in text


# ----- Reusable suite orchestration ------------------------------------------


def test_known_scenarios_order_matches_architecture() -> None:
    """Architecture §12.3 names must all be registered."""

    assert set(KNOWN_SCENARIOS) == {
        SMOKE,
        MARKETPLACE,
        NCCL,
        TEE_OFFLINE,
        WEIGHTS,
    }
    assert list(DEFAULT_SCENARIO_ORDER) == list(KNOWN_SCENARIOS)


def test_run_scenario_suite_all_green(
    live_scenario_api: dict[str, Any],
    mock_master: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-feature reusable suite runner: all five names pass under local sim."""

    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", live_scenario_api["token"])
    monkeypatch.setenv("HYPER_MASTER_BASE_URL", mock_master["base_url"])
    from hypercluster import settings as settings_mod

    cache_clear = getattr(settings_mod.get_settings, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()

    suite = run_scenario_suite(
        base_url=live_scenario_api["base_url"],
        shared_token=live_scenario_api["token"],
        master_url=mock_master["base_url"],
    )
    assert isinstance(suite, SuiteResult)
    assert suite.ok is True, "\n".join(suite.summary_lines())
    assert {r.name for r in suite.results} == set(KNOWN_SCENARIOS)
    assert all(r.ok for r in suite.results)
