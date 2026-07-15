"""VAL-CROSS-012/019/020/027: weight push ack, multi-miner LB, self-deal, master chaos.

Pure local sim + isolated SQLite + mock-master. No live Verda.
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
from hypercluster.sim.cross_weights_leaderboard_selfdeal import (
    CROSS_WEIGHTS_LEADERBOARD,
    MINER_A,
    MINER_B,
    MINER_C,
    SELF_DEAL_HK,
    TWIN_HONEST_HK,
    run_cross_leaderboard_weights_agree,
    run_cross_mock_master_down_resilience,
    run_cross_self_deal_finite_damped,
    run_cross_weight_push_ack,
    run_cross_weights_leaderboard_selfdeal,
)
from hypercluster.sim.orchestration import run_cross_weights_leaderboard_selfdeal_bundle
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT
from hypercluster.sim.scenarios import (
    CROSS_WEIGHTS_LEADERBOARD as SCENARIO_NAME,
)
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


def _spawn_api(
    *,
    settings_factory: Any,
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    port: int | None = None,
    master_url: str = "http://127.0.0.1:3201",
) -> dict[str, Any]:
    """Start combined-worker challenge API bound in mission port band."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings, clear_settings_cache

    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)
    monkeypatch.setenv(
        "CHALLENGE_DATABASE_URL",
        f"sqlite+aiosqlite:///{db_path}",
    )
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
        weight_push_enabled=False,  # explicit scenario/CLI push only
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
    }


def _stop_api(handles: dict[str, Any]) -> None:
    handles["server"].should_exit = True
    handles["thread"].join(timeout=10)
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()


def _spawn_mock_master(
    *,
    preferred: int | None = 3201,
) -> dict[str, Any]:
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
            r = httpx.get(f"{base_url}/health", timeout=1.0)
            if r.status_code == 200:
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
    """API + mock-master both on mission ports with shared SQLite."""

    master = _spawn_mock_master()
    try:
        api = _spawn_api(
            settings_factory=settings_factory,
            db_path=tmp_path / "cross-wls.sqlite3",
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
            "db_path": api["db_path"],
        }
    finally:
        _stop_api(api)
        _stop_master(master)


# ----- VAL-CROSS-012 weight push ack after scored chain ----------------------


def test_weight_push_after_scored_chain_acks_mock_master(
    live_stack: dict[str, Any],
) -> None:
    """VAL-CROSS-012: push after scores → mock-master ack + finite map."""

    result = run_cross_weight_push_ack(
        live_stack["base_url"],
        shared_token=TOKEN,
        master_url=live_stack["master_url"],
        timeout=45.0,
    )
    assert result.ok is True, "\n".join(result.steps + [result.message])
    assert any("push_status=acked" in s or "push status=acknowledged" in s for s in result.steps)


def test_weight_preview_reflects_push_map(live_stack: dict[str, Any]) -> None:
    """VAL-CROSS-012 surface: weight-preview finite after ack."""

    r = run_cross_weight_push_ack(
        live_stack["base_url"],
        shared_token=TOKEN,
        master_url=live_stack["master_url"],
    )
    assert r.ok is True, "\n".join(r.steps + [r.message])
    preview = httpx.get(f"{live_stack['base_url']}/v1/weight-preview", timeout=5.0)
    assert preview.status_code == 200
    wmap = preview.json().get("weights") or {}
    assert MINER_A in wmap and float(wmap[MINER_A]) > 0
    snap = preview.json().get("snapshot") or {}
    if snap:
        assert snap.get("push_status") in {"acked", "sim", "pending"}


# ----- VAL-CROSS-019 multi-miner leaderboard ↔ weights -----------------------


def test_leaderboard_scores_weights_agree_multi_miner(
    live_stack: dict[str, Any],
) -> None:
    """VAL-CROSS-019: three hotkeys rank order matches weight mass order."""

    # Seed via push-ack runner then assert agreement (no second seed needed).
    seed = run_cross_weight_push_ack(
        live_stack["base_url"],
        shared_token=TOKEN,
        master_url=live_stack["master_url"],
    )
    assert seed.ok is True, "\n".join(seed.steps + [seed.message])

    result = run_cross_leaderboard_weights_agree(
        live_stack["base_url"],
        shared_token=TOKEN,
        master_url=live_stack["master_url"],
        ensure_seeded=False,
    )
    assert result.ok is True, "\n".join(result.steps + [result.message])

    board = httpx.get(f"{live_stack['base_url']}/v1/leaderboard", timeout=5.0).json()
    items = board.get("items") or []
    our = [
        row
        for row in items
        if isinstance(row, dict) and row.get("hotkey") in {MINER_A, MINER_B, MINER_C}
    ]
    assert len(our) == 3
    order = [row["hotkey"] for row in our]
    assert order.index(MINER_A) < order.index(MINER_B) < order.index(MINER_C)
    assert float(our[0]["aggregate"]) > float(our[-1]["aggregate"])

    wmap = httpx.get(
        f"{live_stack['base_url']}/v1/weight-preview", timeout=5.0
    ).json().get("weights") or {}
    assert float(wmap[MINER_A]) > float(wmap[MINER_B]) > float(wmap[MINER_C]) > 0


# ----- VAL-CROSS-020 self-deal dual-role finite + damped ---------------------


def test_self_deal_demand_supply_finite_and_damped(
    live_stack: dict[str, Any],
) -> None:
    """VAL-CROSS-020: same hotkey demand+supply scores finite; soft damping applied."""

    seed = run_cross_weight_push_ack(
        live_stack["base_url"],
        shared_token=TOKEN,
        master_url=live_stack["master_url"],
    )
    assert seed.ok is True, "\n".join(seed.steps + [seed.message])

    result = run_cross_self_deal_finite_damped(
        live_stack["base_url"],
        shared_token=TOKEN,
        master_url=live_stack["master_url"],
        ensure_seeded=False,
    )
    assert result.ok is True, "\n".join(result.steps + [result.message])

    scores = httpx.get(
        f"{live_stack['base_url']}/v1/scores/{SELF_DEAL_HK}", timeout=5.0
    ).json()
    items = scores.get("items") or []
    roles = {i.get("role") for i in items}
    assert "demand" in roles and "supply" in roles
    for item in items:
        assert float(item["composite"]) >= 0.0

    wmap = httpx.get(
        f"{live_stack['base_url']}/v1/weight-preview", timeout=5.0
    ).json().get("weights") or {}
    self_w = float(wmap[SELF_DEAL_HK])
    twin_w = float(wmap.get(TWIN_HONEST_HK) or 0.0)
    assert self_w >= 0.0
    # damped expected ~6.0; undamped would be 12.0
    assert self_w < 12.0 - 1e-6
    assert abs(self_w - 6.0) < 0.25
    assert twin_w == pytest.approx(8.0, abs=0.25)


# ----- VAL-CROSS-027 mock-master down resilience -----------------------------


def test_mock_master_down_keeps_scores_then_retries_acked(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VAL-CROSS-027: master stop keeps scores; pending/failed; recover → acked."""

    master_handles: dict[str, Any] | None = _spawn_mock_master()
    master_url_box: dict[str, str] = {"url": master_handles["base_url"]}

    api = _spawn_api(
        settings_factory=settings_factory,
        db_path=tmp_path / "cross-wls-chaos.sqlite3",
        monkeypatch=monkeypatch,
        master_url=master_url_box["url"],
    )
    try:
        # Seed with master up so we have durable scores.
        seed = run_cross_weight_push_ack(
            api["base_url"],
            shared_token=TOKEN,
            master_url=master_url_box["url"],
        )
        assert seed.ok is True, "\n".join(seed.steps + [seed.message])

        def stop_master() -> None:
            nonlocal master_handles
            _stop_master(master_handles)
            master_handles = None

        def start_master() -> str:
            nonlocal master_handles
            # Prefer same port when free; otherwise any band port.
            preferred_port = int(
                master_url_box["url"].rsplit(":", 1)[-1]
            )
            master_handles = _spawn_mock_master(preferred=preferred_port)
            master_url_box["url"] = master_handles["base_url"]
            return master_handles["base_url"]

        result = run_cross_mock_master_down_resilience(
            api["base_url"],
            shared_token=TOKEN,
            master_url=master_url_box["url"],
            stop_master_fn=stop_master,
            start_master_fn=start_master,
            ensure_seeded=False,
            timeout=60.0,
        )
        assert result.ok is True, "\n".join(result.steps + [result.message])
        assert any("scores durable" in s for s in result.steps)
        assert any(
            "eventual acked" in s or "recovery push status=acknowledged" in s
            for s in result.steps
        )

        # Scores still present for all three miners.
        for hk in (MINER_A, MINER_B, MINER_C):
            sc = httpx.get(f"{api['base_url']}/v1/scores/{hk}", timeout=5.0)
            assert sc.status_code == 200
            assert len(sc.json().get("items") or []) >= 1
    finally:
        _stop_api(api)
        _stop_master(master_handles)


# ----- Bundle + CLI dispatch -------------------------------------------------


def test_bundle_all_four_assertions(live_stack: dict[str, Any]) -> None:
    """Full bundle VAL-CROSS-012/019/020; chaos without restart inject is soft."""

    # Bundle without injectors still covers 012/019/020 (027 soft when no injectors).
    result = run_cross_weights_leaderboard_selfdeal_bundle(
        live_stack["base_url"],
        shared_token=TOKEN,
        master_url=live_stack["master_url"],
        include_master_chaos=False,
    )
    assert result.ok is True, "\n".join(result.steps + [result.message])
    assert result.name == CROSS_WEIGHTS_LEADERBOARD


def test_bundle_with_chaos_injectors(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bundle with master chaos injectors succeeds end-to-end."""

    master_handles: dict[str, Any] | None = _spawn_mock_master()
    master_url_box: dict[str, str] = {"url": master_handles["base_url"]}
    api = _spawn_api(
        settings_factory=settings_factory,
        db_path=tmp_path / "cross-wls-bundle.sqlite3",
        monkeypatch=monkeypatch,
        master_url=master_url_box["url"],
    )
    try:

        def stop_master() -> None:
            nonlocal master_handles
            _stop_master(master_handles)
            master_handles = None

        def start_master() -> str:
            nonlocal master_handles
            preferred_port = int(master_url_box["url"].rsplit(":", 1)[-1])
            master_handles = _spawn_mock_master(preferred=preferred_port)
            master_url_box["url"] = master_handles["base_url"]
            return master_handles["base_url"]

        result = run_cross_weights_leaderboard_selfdeal(
            api["base_url"],
            shared_token=TOKEN,
            master_url=master_url_box["url"],
            stop_master_fn=stop_master,
            start_master_fn=start_master,
            include_master_chaos=True,
            timeout=90.0,
        )
        assert result.ok is True, "\n".join(result.steps + [result.message])
        assert any("VAL-CROSS-012" in s for s in result.steps)
        assert any("VAL-CROSS-019" in s for s in result.steps)
        assert any("VAL-CROSS-020" in s for s in result.steps)
        assert any("VAL-CROSS-027" in s for s in result.steps)
    finally:
        _stop_api(api)
        _stop_master(master_handles)


def test_run_scenario_dispatch_name(live_stack: dict[str, Any]) -> None:
    """scenarios.run_scenario name dispatches to this feature without chaos injects."""

    result = run_scenario(
        SCENARIO_NAME,
        live_stack["base_url"],
        shared_token=TOKEN,
        master_url=live_stack["master_url"],
        timeout=60.0,
    )
    # Chaos without injectors still verifies durable scores on soft path.
    assert result.ok is True, "\n".join(result.steps + [result.message])


def test_cli_sim_run_scenario_name(
    live_stack: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Typer CLI exposes --name cross-weights-leaderboard-selfdeal."""

    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)
    monkeypatch.setenv("HYPER_MASTER_BASE_URL", live_stack["master_url"])
    invoked = runner.invoke(
        cli_app,
        [
            "sim",
            "run-scenario",
            "--name",
            CROSS_WEIGHTS_LEADERBOARD,
            "--url",
            live_stack["base_url"],
        ],
    )
    # CLI path may soft-pass 027 without chaos injectors.
    assert invoked.exit_code == 0, invoked.output
    assert "PASS" in invoked.output or "passed" in invoked.output.lower()
