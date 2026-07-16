"""VAL-CROSS-004/005/006/021: multi-node fabric success/fail + TEE offline bonus.

CPU-only sim + SQLite. No live Verda. Two live API fixtures:
  - live_cross_mn_api: clean fabric path (success + TEE twin)
  - live_cross_mn_fail_api: HYPER_SIM_ETH_FALLBACK for fabric_gate fail inject
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
from hypercluster.sim.cross_multinode_fabric_tee import (
    CROSS_MULTINODE,
    DEMAND_HK,
    run_cross_multinode_fabric_fail,
    run_cross_multinode_fabric_tee,
    run_cross_multinode_success,
    run_cross_tee_offline_bonus,
)
from hypercluster.sim.orchestration import run_cross_multinode_fabric_tee_bundle
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT
from hypercluster.sim.scenarios import CROSS_MULTINODE as SCENARIO_NAME
from hypercluster.sim.scenarios import run_scenario

TOKEN = "test-challenge-shared-token"
ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"
runner = CliRunner()


def _spawn_live_api(
    *,
    settings_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sim_eth_fallback: bool = False,
    db_name: str = "cross-mn.sqlite3",
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
    if sim_eth_fallback:
        monkeypatch.setenv("HYPER_SIM_ETH_FALLBACK", "true")
    else:
        monkeypatch.delenv("HYPER_SIM_ETH_FALLBACK", raising=False)
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
        sim_eth_fallback=sim_eth_fallback,
        score_window_attempts=50,
        self_deal_damping=0.5,
        weight_push_enabled=False,
        tee_bonus_tdx=1.08,
        tee_bonus_tdx_gpu=1.20,
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
            "sim_eth_fallback": sim_eth_fallback,
        }
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        clear_settings_cache()


@pytest.fixture
def live_cross_mn_api(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Clean multi-node fabric path (success + TEE bonus)."""

    yield from _spawn_live_api(
        settings_factory=settings_factory,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sim_eth_fallback=False,
        db_name="cross-mn-success.sqlite3",
    )


@pytest.fixture
def live_cross_mn_fail_api(
    settings_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """ETH fallback honesty inject API (VAL-CROSS-005)."""

    yield from _spawn_live_api(
        settings_factory=settings_factory,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sim_eth_fallback=True,
        db_name="cross-mn-fail.sqlite3",
    )


# ----- VAL-CROSS-004 multi-node pack + fabric_gate 1 ---------------------------


def test_cross_multinode_success_pack_fabric_gate(
    live_cross_mn_api: dict[str, Any],
) -> None:
    """VAL-CROSS-004: cluster require_ib → world_size≥2 pack → fabric_gate=1."""

    result = run_cross_multinode_success(
        live_cross_mn_api["base_url"],
        shared_token=live_cross_mn_api["token"],
        poll_timeout_s=30.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    text = "\n".join(result.summary_lines()).lower()
    assert "pack placement" in text or "ranks=" in text
    assert "fabric_gate=1" in text or "fabric_gate=1.0" in text
    assert "composite" in text
    assert "nodes=" in text


# ----- VAL-CROSS-021 digest chain ---------------------------------------------


def test_cross_multinode_digest_chain_identity(
    live_cross_mn_api: dict[str, Any],
) -> None:
    """VAL-CROSS-021: plan graph_digest + launcher artifact + report digests chain."""

    result = run_cross_multinode_success(
        live_cross_mn_api["base_url"],
        shared_token=live_cross_mn_api["token"],
        poll_timeout_s=30.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "graph_digest" in joined
    assert "digest chain" in joined
    assert "fabric_report_digest" in joined

    # Black-box: re-fetch job and assert placement graph links to attempt metrics.
    # Parse job_id from steps.
    job_id = None
    for step in result.steps:
        if step.startswith("job_id="):
            job_id = step.split("=", 1)[1].strip()
            break
    assert job_id, f"job_id missing in steps: {result.steps}"
    detail = httpx.get(f"{live_cross_mn_api['base_url']}/v1/jobs/{job_id}", timeout=5.0)
    assert detail.status_code == 200
    placement = detail.json().get("placement") or {}
    graph = placement.get("graph_digest")
    rankmap = placement.get("rankmap") or []
    assert graph
    assert len(rankmap) >= 2
    unique = {b.get("node_id") for b in rankmap if isinstance(b, dict)}
    assert len(unique) >= 2

    attempt = httpx.get(
        f"{live_cross_mn_api['base_url']}/v1/jobs/{job_id}/attempts/1",
        timeout=5.0,
    )
    assert attempt.status_code == 200
    metrics = attempt.json().get("metrics") or {}
    assert metrics.get("fabric_artifact_digest")
    assert attempt.json().get("fabric_report_digest")

    fab = httpx.get(
        f"{live_cross_mn_api['base_url']}/v1/jobs/{job_id}/fabric-report",
        timeout=5.0,
    )
    assert fab.status_code == 200
    assert fab.json().get("report_digest") == attempt.json().get("fabric_report_digest")

    scores = httpx.get(
        f"{live_cross_mn_api['base_url']}/v1/scores/{DEMAND_HK}",
        timeout=5.0,
    )
    assert scores.status_code == 200
    items = scores.json().get("items") or []
    assert items
    score = items[0]
    assert float(score.get("fabric_gate") or 0) == 1.0
    details = score.get("details") or {}
    extra = details.get("extra") if isinstance(details, dict) else None
    # Score details extra should carry graph + artifact digests (chain identity).
    if isinstance(extra, dict):
        assert extra.get("graph_digest") == graph
        assert extra.get("fabric_artifact_digest") == metrics.get("fabric_artifact_digest")
        assert extra.get("fabric_report_digest") == attempt.json().get("fabric_report_digest")


# ----- VAL-CROSS-005 fabric_gate fail inject ----------------------------------


def test_cross_multinode_fabric_fail_zeros_composite_and_weight(
    live_cross_mn_fail_api: dict[str, Any],
) -> None:
    """VAL-CROSS-005: eth fallback inject → fabric_gate=0 composite=0 no weight."""

    assert live_cross_mn_fail_api["sim_eth_fallback"] is True
    result = run_cross_multinode_fabric_fail(
        live_cross_mn_fail_api["base_url"],
        shared_token=live_cross_mn_fail_api["token"],
        poll_timeout_s=30.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "fabric_gate=0" in joined or "composite=0" in joined
    assert "weight mass not inflated" in joined


# ----- VAL-CROSS-006 TEE offline bonus twin -----------------------------------


def test_cross_tee_offline_bonus_multiplies_composite(
    live_cross_mn_api: dict[str, Any],
) -> None:
    """VAL-CROSS-006: offline TDX bonus > tee=none twin on marketplace jobs."""

    result = run_cross_tee_offline_bonus(
        live_cross_mn_api["base_url"],
        shared_token=live_cross_mn_api["token"],
        poll_timeout_s=30.0,
        tee_bonus_tdx=1.08,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    joined = "\n".join(result.steps).lower()
    assert "tee=none" in joined
    assert "tee=tdx" in joined or "tee_bonus" in joined
    assert "composite order" in joined or "bonus" in joined


# ----- Combined runner + CLI --------------------------------------------------


def test_run_cross_multinode_fabric_tee_combined(
    live_cross_mn_api: dict[str, Any],
) -> None:
    """Combined success + TEE (CLI default, no fail inject on clean path)."""

    result = run_cross_multinode_fabric_tee(
        live_cross_mn_api["base_url"],
        shared_token=live_cross_mn_api["token"],
        poll_timeout_s=30.0,
        include_fail_inject=False,
        include_tee_bonus=True,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    assert result.name == CROSS_MULTINODE
    assert SCENARIO_NAME == "cross-multinode-fabric-tee"


def test_run_scenario_dispatches_cross_multinode(
    live_cross_mn_api: dict[str, Any],
) -> None:
    """Dispatcher routes cross-multinode-fabric-tee to combined runner."""

    result = run_scenario(
        "cross-multinode-fabric-tee",
        live_cross_mn_api["base_url"],
        shared_token=live_cross_mn_api["token"],
        timeout=90.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())
    assert result.name == CROSS_MULTINODE


def test_run_cross_multinode_fabric_tee_bundle(
    live_cross_mn_api: dict[str, Any],
) -> None:
    """Orchestration bundle wrapper."""

    result = run_cross_multinode_fabric_tee_bundle(
        live_cross_mn_api["base_url"],
        shared_token=live_cross_mn_api["token"],
        timeout=90.0,
    )
    assert result.ok is True, "\n".join(result.summary_lines())


def test_cli_sim_run_scenario_cross_multinode(
    live_cross_mn_api: dict[str, Any],
) -> None:
    """CLI: sim run-scenario --name cross-multinode-fabric-tee."""

    r = runner.invoke(
        cli_app,
        [
            "sim",
            "run-scenario",
            "--name",
            "cross-multinode-fabric-tee",
            "--url",
            live_cross_mn_api["base_url"],
        ],
        env={
            **dict(__import__("os").environ),
            "CHALLENGE_SHARED_TOKEN": live_cross_mn_api["token"],
        },
    )
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output or "pass" in r.output.lower()
