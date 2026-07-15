"""Scaffold CLI health/version + sim doctor/smoke identity gates.

Covers VAL-SCAF-030, VAL-SCAF-031, VAL-SCAF-032, VAL-SCAF-036.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

from hypercluster import __version__ as pkg_version
from hypercluster.cli import app as cli_app
from hypercluster.sim.ports import (
    DEFAULT_BAREMETAL_PORT,
    MAX_MISSION_PORT,
    MIN_MISSION_PORT,
    assert_mission_port,
    mission_port_band,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


@pytest.fixture
def live_api(settings_factory, tmp_path: Path) -> Any:
    """Start a short uvicorn instance on a free mission-band port (3200–3299)."""

    import socket

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'live.sqlite3'}",
        shared_token="cli-scaffold-token",
        shared_token_file=None,
    )
    fastapi_app = create_app(settings)

    # Prefer 3200 then fall through band so bare-metal default stays primary.
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
        yield {"base_url": base_url, "port": bound_port}
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_mission_port_band_constants() -> None:
    """VAL-SCAF-030: bare-metal default lives in the mission port band."""

    assert MIN_MISSION_PORT == 3200
    assert MAX_MISSION_PORT == 3299
    assert DEFAULT_BAREMETAL_PORT == 3200
    assert mission_port_band() == (3200, 3299)
    assert_mission_port(3200)
    assert_mission_port(3250)
    assert_mission_port(3299)
    with pytest.raises(ValueError, match="3200"):
        assert_mission_port(3180)
    with pytest.raises(ValueError, match="3200"):
        assert_mission_port(8000)
    with pytest.raises(ValueError, match="3200"):
        assert_mission_port(3300)


def test_cli_version_offline_reports_package_identity() -> None:
    """VAL-SCAF-032 offline: package print is challenge version, product name hypercluster."""

    result = runner.invoke(cli_app, ["version"])
    assert result.exit_code == 0, result.output
    assert pkg_version in result.output
    # Do not invent unrelated product names.
    assert "lium" not in result.output.lower()
    assert "verda" not in result.output.lower()


def test_cli_health_talks_to_live_base_url(live_api: dict[str, Any]) -> None:
    """VAL-SCAF-031: health command exercises live /health and exits 0 when ok."""

    base_url = live_api["base_url"]
    result = runner.invoke(cli_app, ["health", "--url", base_url])
    assert result.exit_code == 0, result.output
    # JSON identity fragments must appear in CLI output.
    assert "hypercluster" in result.output
    assert "ok" in result.output
    assert '"slug"' in result.output or "slug" in result.output


def test_cli_health_nonzero_when_unreachable() -> None:
    """VAL-SCAF-031: CLI must not false-positive when API is down."""

    # Port inside band but nothing listening (avoid binding foreign services).
    result = runner.invoke(cli_app, ["health", "--url", "http://127.0.0.1:3298"])
    assert result.exit_code != 0
    assert result.exit_code == 1 or result.exit_code == 2


def test_cli_health_nonzero_when_status_unhealthy() -> None:
    """VAL-SCAF-031: non-zero exit when /health status is not ok (policy)."""

    from hypercluster.cli import _evaluate_health_payload

    # Direct unit of CLI policy used by the command.
    assert _evaluate_health_payload({"status": "ok", "slug": "hypercluster", "ready": True}) == 0
    assert (
        _evaluate_health_payload({"status": "unhealthy", "slug": "hypercluster", "ready": False})
        != 0
    )
    assert (
        _evaluate_health_payload({"status": "degraded", "slug": "hypercluster", "ready": True})
        != 0
    )


def test_cli_version_live_matches_http_version(live_api: dict[str, Any]) -> None:
    """VAL-SCAF-032 live: CLI version --url matches curl /version challenge_version."""

    import httpx

    base_url = live_api["base_url"]
    http_body = httpx.get(f"{base_url}/version", timeout=5.0).json()
    result = runner.invoke(cli_app, ["version", "--url", base_url])
    assert result.exit_code == 0, result.output
    challenge_version = http_body["challenge_version"]
    assert challenge_version in result.output
    assert challenge_version == pkg_version or challenge_version in result.output
    assert "hypercluster" in result.output.lower() or http_body["challenge_slug"] in result.output


def test_cli_health_defaults_to_port_band_default() -> None:
    """VAL-SCAF-030: default bare-metal base URL uses port 3200."""

    from hypercluster.cli import DEFAULT_BASE_URL, default_base_url

    assert DEFAULT_BASE_URL == "http://127.0.0.1:3200"
    assert default_base_url() == "http://127.0.0.1:3200"
    assert ":3200" in default_base_url()


def test_sim_doctor_passes_when_identity_green(live_api: dict[str, Any]) -> None:
    """VAL-SCAF-036: sim doctor requires health+ready green."""

    base_url = live_api["base_url"]
    result = runner.invoke(cli_app, ["sim", "doctor", "--url", base_url])
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "health" in out or "ready" in out or "ok" in out


def test_sim_doctor_fails_when_api_down() -> None:
    """VAL-SCAF-036: doctor fails closed when identity not green."""

    result = runner.invoke(cli_app, ["sim", "doctor", "--url", "http://127.0.0.1:3297"])
    assert result.exit_code != 0


def test_sim_run_scenario_smoke_requires_identity(live_api: dict[str, Any]) -> None:
    """VAL-SCAF-036: smoke scenario presupposes health/ready green and passes when so."""

    base_url = live_api["base_url"]
    result = runner.invoke(
        cli_app,
        ["sim", "run-scenario", "--name", "smoke", "--url", base_url],
    )
    assert result.exit_code == 0, result.output
    assert "smoke" in result.output.lower() or "pass" in result.output.lower()
    # Must document which base URL was hit.
    assert base_url in result.output or "127.0.0.1" in result.output


def test_sim_run_scenario_smoke_fails_when_ready_not_green() -> None:
    """VAL-SCAF-036: smoke must fail if identity is not reachable/gready."""

    result = runner.invoke(
        cli_app,
        ["sim", "run-scenario", "--name", "smoke", "--url", "http://127.0.0.1:3296"],
    )
    assert result.exit_code != 0


def test_sim_library_probe_identity_unit(live_api: dict[str, Any]) -> None:
    """Library-level identity gate used by doctor/smoke."""

    from hypercluster.sim.identity import probe_identity_gates

    report = probe_identity_gates(live_api["base_url"])
    assert report.ok is True
    assert report.health_status == "ok"
    assert report.ready is True
    assert report.slug == "hypercluster"
    assert report.base_url == live_api["base_url"].rstrip("/")


def test_sim_library_probe_fails_closed() -> None:
    from hypercluster.sim.identity import probe_identity_gates

    report = probe_identity_gates("http://127.0.0.1:3295", timeout=0.5)
    assert report.ok is False
