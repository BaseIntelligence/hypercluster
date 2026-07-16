"""VAL-PRICE-040 / VAL-PRICE-041: CLI prices list|get|set|disable|history.

Parity with public/admin GPU price catalog HTTP routes.
Shared token via env/secret; NEVER print token; nonzero exit on 4xx/5xx.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import httpx
import pytest
import uvicorn
from typer.testing import CliRunner

from hypercluster.cli import app as cli_app
from hypercluster.domain.pricing import disable_catalog_price, upsert_catalog_price
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT

TOKEN = "prices-cli-token-SUPERSECRET-do-not-echo-ever"

MODEL_ACTIVE = "H100_80GB"
MODEL_A100 = "A100_40GB"
MODEL_DISABLED = "RTX4090_24GB_DISABLED"

runner = CliRunner()


@pytest.fixture
def live_api(settings_factory: Any, tmp_path: Any) -> Any:
    """Short-lived uvicorn for prices CLI vs live API (mission ports)."""

    from pathlib import Path

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    tmp = Path(tmp_path)
    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp / 'cli-prices.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        price_seed_on_boot=False,
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
        pytest.fail("prices CLI live API failed to become healthy")

    database = fastapi_app.state.database

    async def _seed() -> None:
        async with database.session() as session:
            await upsert_catalog_price(
                session,
                model_key=MODEL_ACTIVE,
                price_per_hour=2.49,
                family="h100",
                display_name="NVIDIA H100 80GB",
                notes="secret-operator-note-do-not-leak",
                source="admin",
                changed_by="admin",
                reason="seed active",
            )
            await upsert_catalog_price(
                session,
                model_key=MODEL_A100,
                price_per_hour=1.29,
                family="a100",
                display_name="NVIDIA A100 40GB",
                notes="another-secret-note",
                source="admin",
                changed_by="admin",
                reason="seed a100",
            )
            await upsert_catalog_price(
                session,
                model_key=MODEL_DISABLED,
                price_per_hour=0.45,
                family="rtx4090",
                display_name="RTX 4090 disabled test",
                notes="disabled-secret",
                active=True,
                source="admin",
                changed_by="admin",
                reason="seed then disable",
            )
            await disable_catalog_price(
                session,
                model_key=MODEL_DISABLED,
                changed_by="admin",
                reason="disable for list filter",
            )
            await session.commit()

    import asyncio

    asyncio.run(_seed())

    try:
        yield {"base": base, "port": bound_port, "app": fastapi_app}
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _assert_token_not_leaked(*chunks: str) -> None:
    """Never print the shared challenge token (VAL-PRICE-041)."""

    for chunk in chunks:
        assert TOKEN not in chunk, f"token leaked in output: {chunk[:400]!r}"


# ---------------------------------------------------------------------------
# VAL-PRICE-040: list / get agree with public API
# ---------------------------------------------------------------------------


def test_cli_prices_help_lists_subcommands() -> None:
    """VAL-PRICE-040: prices group documents list|get|set|disable|history."""

    top = runner.invoke(cli_app, ["--help"])
    assert top.exit_code == 0, top.output
    assert "prices" in top.output.lower()

    sub = runner.invoke(cli_app, ["prices", "--help"])
    assert sub.exit_code == 0, sub.output
    lower = sub.output.lower()
    for name in ("list", "get", "set", "disable", "history"):
        assert name in lower, f"missing prices subcommand {name!r} in help"


def test_cli_prices_list_matches_public_api(live_api: dict[str, Any]) -> None:
    """VAL-PRICE-040: prices list exit 0, keys match GET /v1/gpu-prices."""

    base = live_api["base"]
    api = httpx.get(f"{base}/v1/gpu-prices", timeout=5.0)
    assert api.status_code == 200, api.text
    api_body = api.json()
    api_keys = {item["model_key"] for item in api_body["items"]}
    assert MODEL_ACTIVE in api_keys
    assert MODEL_A100 in api_keys
    assert MODEL_DISABLED not in api_keys

    result = runner.invoke(cli_app, ["prices", "list", "--url", base])
    assert result.exit_code == 0, result.output
    _assert_token_not_leaked(result.output, result.stdout or "", result.stderr or "")
    body = json.loads(result.output)
    cli_keys = {item["model_key"] for item in body["items"]}
    assert cli_keys == api_keys
    assert body["count"] == api_body["count"]
    # public shape: no operator notes
    for item in body["items"]:
        assert "notes" not in item


def test_cli_prices_get_matches_public_api(live_api: dict[str, Any]) -> None:
    """VAL-PRICE-040: prices get MODEL_KEY agrees with public GET when active."""

    base = live_api["base"]
    api = httpx.get(f"{base}/v1/gpu-prices/{MODEL_ACTIVE}", timeout=5.0)
    assert api.status_code == 200, api.text
    api_body = api.json()

    result = runner.invoke(
        cli_app,
        ["prices", "get", MODEL_ACTIVE, "--url", base],
    )
    assert result.exit_code == 0, result.output
    _assert_token_not_leaked(result.output)
    body = json.loads(result.output)
    assert body["model_key"] == MODEL_ACTIVE
    assert float(body["price_per_hour"]) == pytest.approx(
        float(api_body["price_per_hour"])
    )
    assert "notes" not in body


def test_cli_prices_list_all_includes_inactive_with_token(
    live_api: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-PRICE-040: optional --all uses admin token for inactive rows."""

    base = live_api["base"]
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)

    result = runner.invoke(cli_app, ["prices", "list", "--all", "--url", base])
    assert result.exit_code == 0, result.output
    _assert_token_not_leaked(result.output, result.stdout or "", result.stderr or "")
    body = json.loads(result.output)
    keys = {item["model_key"] for item in body["items"]}
    assert MODEL_ACTIVE in keys
    assert MODEL_DISABLED in keys


# ---------------------------------------------------------------------------
# VAL-PRICE-041: set / disable / history + no token leak + error exits
# ---------------------------------------------------------------------------


def test_cli_prices_set_changes_public_get(
    live_api: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-PRICE-041: prices set updates catalog; public GET shows new price."""

    base = live_api["base"]
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)
    new_price = 3.33

    before = httpx.get(f"{base}/v1/gpu-prices/{MODEL_ACTIVE}", timeout=5.0)
    assert before.status_code == 200
    assert float(before.json()["price_per_hour"]) != pytest.approx(new_price)

    result = runner.invoke(
        cli_app,
        [
            "prices",
            "set",
            MODEL_ACTIVE,
            "--price",
            str(new_price),
            "--reason",
            "ops reprice",
            "--url",
            base,
        ],
    )
    assert result.exit_code == 0, result.output
    _assert_token_not_leaked(result.output, result.stdout or "", result.stderr or "")
    set_body = json.loads(result.output)
    catalog = set_body.get("catalog") or set_body
    assert float(catalog["price_per_hour"]) == pytest.approx(new_price)

    after = httpx.get(f"{base}/v1/gpu-prices/{MODEL_ACTIVE}", timeout=5.0)
    assert after.status_code == 200, after.text
    assert float(after.json()["price_per_hour"]) == pytest.approx(new_price)

    # CLI list also shows new price
    listed = runner.invoke(cli_app, ["prices", "list", "--url", base])
    assert listed.exit_code == 0, listed.output
    list_body = json.loads(listed.output)
    match = next(i for i in list_body["items"] if i["model_key"] == MODEL_ACTIVE)
    assert float(match["price_per_hour"]) == pytest.approx(new_price)
    _assert_token_not_leaked(listed.output)


def test_cli_prices_disable_and_history(
    live_api: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-PRICE-041: disable digs active=0; history prints audit rows."""

    base = live_api["base"]
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)

    # Disable a still-active model
    result = runner.invoke(
        cli_app,
        [
            "prices",
            "disable",
            MODEL_A100,
            "--reason",
            "cli disable test",
            "--url",
            base,
        ],
    )
    assert result.exit_code == 0, result.output
    _assert_token_not_leaked(result.output)
    body = json.loads(result.output)
    catalog = body.get("catalog") or body
    assert int(catalog.get("active", 1)) == 0 or catalog.get("active") is False

    # Public GET now 404
    pub = httpx.get(f"{base}/v1/gpu-prices/{MODEL_A100}", timeout=5.0)
    assert pub.status_code == 404

    # History has rows
    hist = runner.invoke(
        cli_app,
        ["prices", "history", MODEL_A100, "--url", base],
    )
    assert hist.exit_code == 0, hist.output
    _assert_token_not_leaked(hist.output)
    hist_body = json.loads(hist.output)
    assert hist_body["model_key"] == MODEL_A100
    assert hist_body["count"] >= 1
    assert isinstance(hist_body["items"], list)
    assert len(hist_body["items"]) >= 1


def test_cli_prices_set_rejects_missing_token(
    live_api: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-PRICE-041: set without token fails closed (exit non-zero), no leak."""

    base = live_api["base"]
    monkeypatch.delenv("CHALLENGE_SHARED_TOKEN", raising=False)
    monkeypatch.delenv("HYPER_SHARED_TOKEN", raising=False)

    result = runner.invoke(
        cli_app,
        [
            "prices",
            "set",
            MODEL_ACTIVE,
            "--price",
            "9.99",
            "--url",
            base,
        ],
    )
    assert result.exit_code != 0, result.output
    _assert_token_not_leaked(result.output, result.stdout or "", result.stderr or "")


def test_cli_prices_get_unknown_exits_nonzero(live_api: dict[str, Any]) -> None:
    """VAL-PRICE-041 / fail-closed: get missing active key exits non-zero (4xx)."""

    base = live_api["base"]
    result = runner.invoke(
        cli_app,
        ["prices", "get", "DOES_NOT_EXIST_GPU", "--url", base],
    )
    assert result.exit_code != 0, result.output
    _assert_token_not_leaked(result.output)


def test_cli_prices_list_connection_error_nonzero() -> None:
    """Nonzero exit when API is down (connection error)."""

    # Port that is almost certainly closed in mission band
    dead = "http://127.0.0.1:3298"
    result = runner.invoke(cli_app, ["prices", "list", "--url", dead])
    assert result.exit_code != 0
