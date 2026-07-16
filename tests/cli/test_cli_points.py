"""VAL-WGT-021: CLI points balance|list (+ optional history).

Pass: exit 0 and agree with API for seeded data; no secrets echoed.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from typer.testing import CliRunner

from hypercluster.cli import app as cli_app
from hypercluster.db.models import PointsLedger, utc_now
from hypercluster.domain.points import REASON_SCORE_EARN, compute_score_earn_delta
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT

TOKEN = "points-cli-token-SUPERSECRET-do-not-echo"
HOTKEY_A = "points-cli-hotkey-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
HOTKEY_B = "points-cli-hotkey-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

runner = CliRunner()


@pytest.fixture
def live_api(settings_factory: Any, tmp_path: Path) -> Any:
    """Short-lived uvicorn for points CLI vs live API (mission ports)."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'cli-points.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        points_enabled=True,
        points_scale=1.0,
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
        pytest.skip("no free mission-band port")

    config = uvicorn.Config(
        fastapi_app,
        host="127.0.0.1",
        port=bound_port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{bound_port}"
    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=0.5)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("points CLI live API failed to become healthy")

    # Seed two balances via domain session on the running app.
    database = fastapi_app.state.database

    async def _seed() -> None:
        from hypercluster.domain.points import _upsert_balance

        async with database.session() as session:
            for hk, comp in ((HOTKEY_A, 5.0), (HOTKEY_B, 2.5)):
                delta = compute_score_earn_delta(comp, scale=1.0)
                bal_after = await _upsert_balance(session, hotkey=hk, delta=delta)
                session.add(
                    PointsLedger(
                        id=str(uuid.uuid4()),
                        hotkey=hk,
                        role="demand",
                        delta=float(delta),
                        balance_after=float(bal_after),
                        reason=REASON_SCORE_EARN,
                        score_id=str(uuid.uuid4()),
                        attempt_id=str(uuid.uuid4()),
                        details_json=json.dumps({"composite": comp}),
                        created_at=utc_now(),
                    )
                )
            await session.commit()

    import asyncio

    asyncio.run(_seed())

    try:
        yield {"base": base, "port": bound_port, "app": fastapi_app}
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_cli_points_help_lists_balance_and_list() -> None:
    """VAL-WGT-021: points group documents balance + list."""

    top = runner.invoke(cli_app, ["--help"])
    assert top.exit_code == 0, top.output
    assert "points" in top.output.lower()

    sub = runner.invoke(cli_app, ["points", "--help"])
    assert sub.exit_code == 0, sub.output
    lower = sub.output.lower()
    assert "balance" in lower
    assert "list" in lower


def test_cli_points_balance_agrees_with_api(live_api: dict[str, Any]) -> None:
    """VAL-WGT-021: points balance exit 0 and matches GET /v1/points/{hotkey}."""

    base = live_api["base"]
    api = httpx.get(f"{base}/v1/points/{HOTKEY_A}", timeout=5.0)
    assert api.status_code == 200, api.text
    api_balance = float(api.json()["balance"])
    assert api_balance == pytest.approx(5.0)

    result = runner.invoke(
        cli_app,
        ["points", "balance", "--hotkey", HOTKEY_A, "--url", base],
    )
    assert result.exit_code == 0, result.output
    assert TOKEN not in result.output
    assert "set_weights" not in result.output
    body = json.loads(result.output)
    assert body["hotkey"] == HOTKEY_A
    assert float(body["balance"]) == pytest.approx(api_balance)


def test_cli_points_list_agrees_with_api(live_api: dict[str, Any]) -> None:
    """VAL-WGT-021: points list exit 0 and matches GET /v1/points."""

    base = live_api["base"]
    api = httpx.get(f"{base}/v1/points", timeout=5.0)
    assert api.status_code == 200, api.text
    api_body = api.json()
    api_map = {r["hotkey"]: float(r["balance"]) for r in api_body["items"]}

    result = runner.invoke(cli_app, ["points", "list", "--url", base])
    assert result.exit_code == 0, result.output
    assert TOKEN not in result.output
    body = json.loads(result.output)
    assert body["count"] == api_body["count"]
    cli_map = {r["hotkey"]: float(r["balance"]) for r in body["items"]}
    assert cli_map[HOTKEY_A] == pytest.approx(api_map[HOTKEY_A])
    assert cli_map[HOTKEY_B] == pytest.approx(api_map[HOTKEY_B])


def test_cli_points_history_optional(live_api: dict[str, Any]) -> None:
    """Optional history subcommand agrees with API when present."""

    base = live_api["base"]
    result = runner.invoke(
        cli_app,
        ["points", "history", "--hotkey", HOTKEY_A, "--url", base],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["hotkey"] == HOTKEY_A
    assert body["count"] >= 1
    assert body["items"][0]["reason"] == REASON_SCORE_EARN
    assert TOKEN not in result.output


def test_cli_points_balance_never_seen_zero(live_api: dict[str, Any]) -> None:
    """CLI balance for unseen hotkey is empty-safe 0."""

    base = live_api["base"]
    never = "points-cli-hotkey-never-seen-zzzzzzzzzzzzzzzzzzzz"
    result = runner.invoke(
        cli_app,
        ["points", "balance", "--hotkey", never, "--url", base],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert float(body["balance"]) == 0.0
