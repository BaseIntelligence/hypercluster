"""CLI packaging + core surface (M7).

Covers:
  VAL-CLI-001  --help lists major groups
  VAL-CLI-002  version package string
  VAL-CLI-003  health --url up/down
  VAL-CLI-004  db init schema under data path (idempotent)
  VAL-CLI-020  serve binds configured port /health
  VAL-CLI-021  incomplete auth flags reject mutate
  VAL-CLI-022  unknown scenario fails with valid names
  VAL-CLI-023  non-zero on connection / down API for list+status
  VAL-CLI-024  --json machine mode parseable
  VAL-CLI-025  packaging entrypoint hypercluster
  VAL-CLI-026  missing required args fail closed with usage
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from typer.testing import CliRunner

from hypercluster import __version__ as pkg_version
from hypercluster.cli import app as cli_app
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT

REPO_ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()

REQUIRED_TOP_LEVEL = (
    "serve",
    "version",
    "health",
    "db",
    "marketplace",
    "nodes",
    "jobs",
    "fabric",
    "attest",
    "score",
    "weights",
    "sim",
)


@pytest.fixture
def live_api(settings_factory, tmp_path: Path) -> Any:
    """Short-lived uvicorn on a free mission-band port."""

    from hypercluster.app import create_app

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'live.sqlite3'}",
        shared_token="cli-packaging-token",
        shared_token_file=None,
    )
    fastapi_app = create_app(settings)

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
        yield {"base_url": base_url, "port": bound_port}
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_help_lists_major_command_groups() -> None:
    """VAL-CLI-001: hypercluster --help documents architecture groups."""

    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0, result.output
    lower = result.output.lower()
    missing = [name for name in REQUIRED_TOP_LEVEL if name not in lower]
    assert not missing, f"help missing groups {missing}:\n{result.output}"


def test_version_prints_package_string() -> None:
    """VAL-CLI-002: offline version exits 0 with non-empty package version."""

    result = runner.invoke(cli_app, ["version"])
    assert result.exit_code == 0, result.output
    assert pkg_version in result.output
    assert result.output.strip()


def test_health_url_up_and_down(live_api: dict[str, Any]) -> None:
    """VAL-CLI-003: health --url exits 0 when ok, non-zero when refused."""

    base_url = live_api["base_url"]
    up = runner.invoke(cli_app, ["health", "--url", base_url])
    assert up.exit_code == 0, up.output

    down = runner.invoke(cli_app, ["health", "--url", "http://127.0.0.1:3298"])
    assert down.exit_code != 0
    assert "fail" in down.output.lower() or "error" in down.output.lower() or down.exit_code == 1


def test_db_init_creates_schema_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """VAL-CLI-004: db init creates providers/jobs tables; second run keeps data."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "challenge.sqlite3"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    monkeypatch.setenv("CHALLENGE_DATABASE_URL", db_url)
    # Force fresh settings cache.
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()

    first = runner.invoke(cli_app, ["db", "init"])
    assert first.exit_code == 0, first.output
    assert db_path.exists(), "sqlite file must appear under data dir"

    import asyncio

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    async def _tables() -> set[str]:
        engine = create_async_engine(db_url)
        async with engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            )
            names = {r[0] for r in rows.fetchall()}
        await engine.dispose()
        return names

    tables = asyncio.run(_tables())
    for expected in ("providers", "jobs", "offers", "leases", "nodes"):
        assert expected in tables, f"missing table {expected} in {tables}"

    # Seed a row then re-init; must not wipe.
    async def _seed() -> None:
        engine = create_async_engine(db_url)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO providers "
                    "(id, hotkey, display_name, status, created_at, updated_at) "
                    "VALUES ('p1', 'hk-seed', 'seed', 'active', "
                    "'2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
                )
            )
        await engine.dispose()

    asyncio.run(_seed())

    second = runner.invoke(cli_app, ["db", "init"])
    assert second.exit_code == 0, second.output

    async def _count() -> int:
        engine = create_async_engine(db_url)
        async with engine.connect() as conn:
            row = await conn.execute(text("SELECT COUNT(*) FROM providers WHERE id='p1'"))
            value = int(row.scalar_one())
        await engine.dispose()
        return value

    assert asyncio.run(_count()) == 1


def test_serve_binds_configured_port_and_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VAL-CLI-020: serve binds mission port and answers /health."""

    # Find free mission port.
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
        pytest.skip("no free mission port")

    db_path = tmp_path / "serve.sqlite3"
    monkeypatch.setenv("CHALLENGE_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", "cli-serve-token")
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN_FILE", "")
    monkeypatch.setenv("CHALLENGE_PORT", str(bound_port))
    monkeypatch.setenv("CHALLENGE_HOST", "127.0.0.1")
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()

    # Launch serve via subprocess so it is the real CLI packaging path.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from hypercluster.cli import app; "
                f"app(['serve', '--host', '127.0.0.1', '--port', '{bound_port}', "
                f"'--no-reload'])"
            ),
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{bound_port}"
        deadline = time.time() + 20.0
        last: Exception | None = None
        body: dict[str, Any] | None = None
        while time.time() < deadline:
            try:
                response = httpx.get(f"{base}/health", timeout=1.0)
                if response.status_code == 200:
                    body = response.json()
                    if body.get("status") == "ok" and body.get("slug") == "hypercluster":
                        break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.15)
        else:
            out = ""
            if proc.stdout:
                try:
                    out = proc.stdout.read() if proc.poll() is not None else ""
                except Exception:  # noqa: BLE001
                    out = ""
            raise AssertionError(f"serve never healthy on {base}: {last!r} out={out[:500]!r}")
        assert body is not None
        assert body["status"] == "ok"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_incomplete_auth_flags_rejected_on_mutate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-CLI-021: mutate without --hotkey/wallet fails clear non-zero."""

    # Ensure no env hotkey/token silently supplies identity.
    monkeypatch.delenv("HYPER_HOTKEY", raising=False)
    monkeypatch.delenv("CHALLENGE_HOTKEY", raising=False)
    monkeypatch.delenv("CHALLENGE_SHARED_TOKEN", raising=False)
    monkeypatch.delenv("HYPER_SHARED_TOKEN", raising=False)

    cases = [
        ["marketplace", "offer", "create", "--node-ids", "n1", "--price", "1", "--lifetime", "1"],
        ["marketplace", "rent", "--offer-id", "offer-x"],
        ["marketplace", "terminate", "--lease-id", "lease-x"],
        ["nodes", "register", "--ssh", "h:22", "--gpus", "1"],
        ["jobs", "submit", "--spec", "jobs/train.yaml"],
        ["jobs", "cancel", "--id", "job-x"],
    ]
    for argv in cases:
        result = runner.invoke(cli_app, argv)
        assert result.exit_code != 0, f"expected non-zero for {argv}: {result.output}"
        out = (result.output + (result.stderr or "")).lower()
        # Clear usage / auth message — not traceback-only silence.
        assert (
            "hotkey" in out
            or "auth" in out
            or "required" in out
            or "token" in out
            or "usage" in out
            or "missing" in out
            or "wallet" in out
        ), f"unclear error for {argv}: {result.output!r}"
        assert "traceback" not in out


def test_unknown_scenario_fails_with_valid_names() -> None:
    """VAL-CLI-022: sim run-scenario unknown name non-zero + list known."""

    result = runner.invoke(
        cli_app,
        ["sim", "run-scenario", "--name", "does-not-exist", "--url", "http://127.0.0.1:3200"],
    )
    assert result.exit_code != 0
    out = result.output.lower()
    assert "does-not-exist" in out or "unknown" in out
    # Valid names must appear so operators know options.
    for name in ("smoke", "marketplace", "tee-offline", "weights"):
        assert name in out


def test_cli_nonzero_on_connection_errors() -> None:
    """VAL-CLI-023: marketplace list / jobs status non-zero when API down."""

    dead = "http://127.0.0.1:3294"
    list_result = runner.invoke(
        cli_app,
        ["marketplace", "offers", "list", "--url", dead],
    )
    assert list_result.exit_code != 0, list_result.output
    assert list_result.output.strip()  # must not blank-success

    status_result = runner.invoke(
        cli_app,
        ["jobs", "status", "--id", "job-x", "--url", dead],
    )
    assert status_result.exit_code != 0, status_result.output


def test_json_machine_mode_parseable(live_api: dict[str, Any]) -> None:
    """VAL-CLI-024: --json list stdout is parseable JSON only (no rich noise)."""

    base = live_api["base_url"]
    result = runner.invoke(
        cli_app,
        ["marketplace", "offers", "list", "--url", base, "--json"],
    )
    assert result.exit_code == 0, result.output
    # Stdout must be pure JSON (array or object).
    text = result.stdout if result.stdout is not None else result.output
    parsed = json.loads(text)
    assert isinstance(parsed, (list, dict))


def test_packaging_entrypoint_resolves() -> None:
    """VAL-CLI-025: project scripts entry hypercluster on PATH via venv/uv."""

    # Prefer installed console script; fall back to module form the package documents.
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "hypercluster",
    ]
    invoked = False
    for path in candidates:
        if path.exists():
            proc = subprocess.run(
                [str(path), "--help"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            assert proc.returncode == 0, proc.stdout + proc.stderr
            out = (proc.stdout + proc.stderr).lower()
            for name in REQUIRED_TOP_LEVEL:
                assert name in out, f"{name} missing from packaged help"
            invoked = True
            break
    if not invoked:
        proc = subprocess.run(
            [sys.executable, "-m", "hypercluster.cli", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


def test_missing_required_args_fail_closed() -> None:
    """VAL-CLI-026: missing required options → non-zero + usage, not traceback silence."""

    matrix: list[list[str]] = [
        ["jobs", "status"],
        ["jobs", "cancel"],
        ["marketplace", "rent"],
        ["marketplace", "terminate"],
        ["nodes", "fabric-scan"],
        ["sim", "run-scenario"],
        ["marketplace", "lease", "show"],
        ["jobs", "logs"],
    ]
    for argv in matrix:
        result = runner.invoke(cli_app, argv)
        assert result.exit_code != 0, f"expected non-zero for {argv}: {result.output!r}"
        combined = (result.output + (result.stderr or "")).lower()
        # Typer/Click usage or named missing option; never silent success.
        assert (
            "usage" in combined
            or "missing" in combined
            or "required" in combined
            or "error" in combined
            or "--id" in combined
            or "--offer-id" in combined
            or "--lease-id" in combined
            or "--node-id" in combined
            or "--name" in combined
            or "option" in combined
        ), f"unclear usage for {argv}: {result.output!r}"
        assert "traceback (most recent call last)" not in combined
