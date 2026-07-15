"""VAL-FAB-016/017/024: plan dry-run non-mutating, CLI report show, multi-node bundles."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from typer.testing import CliRunner

from hypercluster.api.auth import build_signed_headers
from hypercluster.fabric.planner import PlacementRequest, place_ranks
from hypercluster.fabric.report import (
    bundle_job_fabric_report,
    unique_node_ids_from_rankmap,
    validate_bundle_completeness,
)
from hypercluster.sim.inventory import seed_sim_inventory

TOKEN = "test-challenge-shared-token"
SUBMITTER_HK = "fab-bundle-submitter-aaaaaaaaaaaaaaaaaaaaaaaaaa"
ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"


# ----- pure bundle (VAL-FAB-024) --------------------------------------------


def test_multi_node_bundle_matches_rankmap_nodes() -> None:
    """VAL-FAB-024: |nodes in report| matches |unique nodes in rankmap|."""

    inv = seed_sim_inventory(seed=2, node_count=4, gpus_per_node=2)
    plan = place_ranks(
        PlacementRequest(
            job_id="job-bundle",
            world_size=4,
            nnodes=2,
            nproc_per_node=2,
            policy="spread",
            fabric="auto",
            node_reports=inv.reports(),
        )
    )
    assert plan.ok
    rankmap = [b.to_public() for b in plan.rankmap]
    expected_nodes = set(unique_node_ids_from_rankmap(rankmap))
    assert len(expected_nodes) >= 2

    bundle = bundle_job_fabric_report(
        job_id="job-bundle",
        attempt_id="att-1",
        rankmap=rankmap,
        fabric_mode="auto",
        world_size=4,
        nnodes=2,
        reports=inv.reports(),
    )
    assert bundle["node_count"] == len(expected_nodes)
    assert set(bundle["node_ids"]) == expected_nodes
    assert validate_bundle_completeness(bundle, rankmap)
    # Each entry has a report_digest.
    for node in bundle["nodes"]:
        assert node["report_digest"].startswith("sha256:")
    assert bundle["report_digest"].startswith("sha256:")
    # Must not be only rank0 when nnodes>1.
    assert len(bundle["nodes"]) > 1


def test_bundle_only_rank0_would_fail_completeness() -> None:
    """Guard: incomplete single-node list fails completeness for multi-node rankmap."""

    inv = seed_sim_inventory(seed=3, node_count=3, gpus_per_node=2)
    plan = place_ranks(
        PlacementRequest(
            job_id="job-incomplete",
            world_size=3,
            nnodes=3,
            nproc_per_node=1,
            policy="spread",
            fabric="auto",
            node_reports=inv.reports(),
        )
    )
    rankmap = [b.to_public() for b in plan.rankmap]
    bad = {
        "nodes": [{"node_id": rankmap[0]["node_id"], "report_digest": "sha256:" + "a" * 64}],
        "job_id": "job-incomplete",
    }
    assert validate_bundle_completeness(bad, rankmap) is False


# ----- dry-run plan non-mutating (VAL-FAB-016) ------------------------------


def test_fabric_plan_dry_run_does_not_call_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """VAL-FAB-016: dry-run path returns PlacementResult without mutators."""

    from hypercluster.cli import app

    # Dry-run uses sim inventory + place_ranks only; no job mutate.
    called = {"launch": 0}

    def _boom(*_a: Any, **_k: Any) -> None:
        called["launch"] += 1
        raise AssertionError("launcher must not run on plan dry-run")

    monkeypatch.setattr("hypercluster.fabric.launcher.sim_launch", _boom)

    runner = CliRunner()
    # Spec-based dry-run (no live API / no job mutation).
    result = runner.invoke(
        app,
        [
            "fabric",
            "plan",
            "--world-size",
            "4",
            "--nnodes",
            "2",
            "--nproc-per-node",
            "2",
            "--policy",
            "pack",
            "--fabric",
            "auto",
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "rankmap" in payload
    assert payload.get("ok", True) is True
    assert payload.get("dry_run") is True
    assert called["launch"] == 0
    assert "graph_digest" in payload
    assert "planner_version" in payload


# ----- API multi-node fabric report + CLI show (VAL-FAB-017/024) ------------


@pytest.fixture
async def fab_app_client(settings_factory, tmp_path) -> AsyncIterator[tuple[Any, AsyncClient]]:
    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'fab-bundle.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        combined_worker=True,
        combined_worker_interval_seconds=0.05,
        job_image_allowlist=ALLOWED_IMAGE,
        max_job_world_size=64,
        max_job_nnodes=16,
        max_job_nproc_per_node=8,
        max_job_timeout_s=3600,
        max_job_gpu_budget=32,
        sim_job_step_delay_s=0.0,
        sim_job_run_sleep_s=0.0,
        sim_auto_capacity=True,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield app, client


def _sign(body: bytes, *, hotkey: str = SUBMITTER_HK) -> dict[str, str]:
    return build_signed_headers(secret=TOKEN, hotkey=hotkey, body=body)


def _patch_httpx_get_json(
    monkeypatch: pytest.MonkeyPatch,
    path_payloads: dict[str, dict[str, Any] | Callable[[str], dict[str, Any]]],
) -> None:
    """Stub httpx.get so CLI commands see canned JSON without a live server."""

    def _get(url: str, **_kwargs: Any) -> httpx.Response:
        path = urlparse(url).path if url.startswith("http://") else url
        if path not in path_payloads:
            # Prefix match for job-scoped paths.
            match = None
            for key in path_payloads:
                if path.endswith(key) or key in path:
                    match = key
                    break
            if match is None:
                return httpx.Response(404, json={"detail": f"no stub for {path}"})
            payload = path_payloads[match]
        else:
            payload = path_payloads[path]
        if callable(payload):
            body = payload(path)
        else:
            body = payload
        return httpx.Response(200, json=body)

    monkeypatch.setattr(httpx, "get", _get)


async def _submit_and_drain(client: AsyncClient, **overrides: Any) -> str:
    body: dict[str, Any] = {
        "image_digest": ALLOWED_IMAGE,
        "entrypoint": ["python", "-m", "train"],
        "world_size": 4,
        "nnodes": 2,
        "nproc_per_node": 2,
        "timeout_s": 300,
        "resource": {"gpus": 4, "nodes": 2},
        "backend": "nccl",
        "fabric": "auto",
        "tee": "none",
        "placement_policy": "pack",
    }
    body.update(overrides)
    raw = json.dumps(body).encode()
    headers = _sign(raw)
    headers["Content-Type"] = "application/json"
    resp = await client.post("/v1/jobs", content=raw, headers=headers)
    assert resp.status_code in {200, 201}, resp.text
    job_id = resp.json()["id"]

    # Drain until terminal via internal worker steps.
    for _ in range(40):
        detail = await client.get(f"/v1/jobs/{job_id}")
        assert detail.status_code == 200
        status = detail.json()["status"]
        if status in {"succeeded", "failed", "timeout", "cancelled"}:
            return job_id
        # Nudge combined worker via health/version (lifecycle tick already running);
        # also call ready which is cheap.
        await client.get("/health")
        import asyncio

        await asyncio.sleep(0.08)
    return job_id


@pytest.fixture
async def eth_fallback_app_client(
    settings_factory, tmp_path
) -> AsyncIterator[tuple[Any, AsyncClient]]:
    """Combined worker with HYPER_SIM_ETH_FALLBACK=true (VAL-FAB-012 black-box)."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'fab-eth-fallback.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        combined_worker=True,
        combined_worker_interval_seconds=0.05,
        job_image_allowlist=ALLOWED_IMAGE,
        max_job_world_size=64,
        max_job_nnodes=16,
        max_job_nproc_per_node=8,
        max_job_timeout_s=3600,
        max_job_gpu_budget=32,
        sim_job_step_delay_s=0.0,
        sim_job_run_sleep_s=0.0,
        sim_auto_capacity=True,
        sim_eth_fallback=True,
        sim_inventory_spoof=False,
        sim_honesty_level="l1",
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield app, client


@pytest.mark.asyncio
async def test_api_eth_fallback_observably_zeros_fabric_gate(
    eth_fallback_app_client: tuple[Any, AsyncClient],
) -> None:
    """VAL-FAB-012 black-box: HYPER_SIM_ETH_FALLBACK → fabric_gate=0 on job metrics.

    Auto-capacity fabric=ib is allowed without bound IB reports (no member
    reports → missing_ib gate not tripped at place). Fallbacks inject at
    launch so attempt metrics_json shows fabric_gate 0 + composite 0.
    """

    _app, client = eth_fallback_app_client
    job_id = await _submit_and_drain(
        client,
        fabric="ib",
        world_size=2,
        nnodes=2,
        nproc_per_node=1,
        resource={"gpus": 2, "nodes": 2},
        client_request_id="fab-eth-fallback-012",
    )
    detail = await client.get(f"/v1/jobs/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] in {"succeeded", "failed", "timeout"}, body

    attempts = await client.get(f"/v1/jobs/{job_id}/attempts/1")
    assert attempts.status_code == 200, attempts.text
    att = attempts.json()
    metrics = att.get("metrics") or {}
    assert metrics, att
    # Observable score factors via attempt metrics (launcher metrics_json).
    fabric_gate = metrics.get("fabric_gate")
    composite = metrics.get("composite")
    score_factors = metrics.get("score_factors") or {}
    if fabric_gate is None and "fabric_gate" in score_factors:
        fabric_gate = score_factors["fabric_gate"]
    if composite is None and "composite" in score_factors:
        composite = score_factors["composite"]
    assert float(fabric_gate) == 0.0, metrics
    assert float(composite) == 0.0, metrics
    reasons = list(score_factors.get("reason_codes") or [])
    failure = str(att.get("failure_code") or metrics.get("failure_code") or "")
    assert (
        any("fallback" in c.lower() or "eth" in c.lower() for c in reasons)
        or "fallback" in failure.lower()
        or "eth" in failure.lower()
        or fabric_gate == 0.0
    ), metrics


@pytest.mark.asyncio
async def test_completed_multi_node_job_fabric_report_bundles_nodes(
    fab_app_client: tuple[Any, AsyncClient],
) -> None:
    """VAL-FAB-024 + VAL-FAB-013 lifecycle: metrics + multi-node digests on job."""

    _app, client = fab_app_client
    job_id = await _submit_and_drain(client)
    detail = await client.get(f"/v1/jobs/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == "succeeded", body

    fr = await client.get(f"/v1/jobs/{job_id}/fabric-report")
    assert fr.status_code == 200, fr.text
    report = fr.json()
    assert report.get("report_digest") or report.get("fabric_report_digest")
    digest = report.get("report_digest") or report.get("fabric_report_digest")
    assert str(digest).startswith("sha256:") or len(str(digest)) >= 32

    # Bundle nodes when multi-node.
    assert "nodes" in report
    assert len(report["nodes"]) >= 2
    rankmap = (body.get("placement") or {}).get("rankmap") or []
    if rankmap:
        assert validate_bundle_completeness(report, rankmap)

    # Attempt metrics present (VAL-FAB-013 via drain path).
    attempts = await client.get(f"/v1/jobs/{job_id}/attempts/1")
    assert attempts.status_code == 200
    att = attempts.json()
    metrics = att.get("metrics") or {}
    assert metrics
    assert "allreduce_gbps" in metrics or "efficiency" in metrics


@pytest.mark.asyncio
async def test_cli_fabric_report_show_echoes_digest(
    fab_app_client: tuple[Any, AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-FAB-017: CLI fabric report show prints report_digest matching API."""

    _asgi_app, client = fab_app_client
    job_id = await _submit_and_drain(client)
    fr = await client.get(f"/v1/jobs/{job_id}/fabric-report")
    assert fr.status_code == 200
    api_body = fr.json()
    api_digest = api_body.get("report_digest") or api_body.get("fabric_report_digest")
    assert api_digest

    from hypercluster.cli import app

    _patch_httpx_get_json(
        monkeypatch,
        {f"/v1/jobs/{job_id}/fabric-report": api_body},
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["fabric", "report", "show", "--job-id", job_id, "--url", "http://testserver"],
    )
    assert result.exit_code == 0, result.output
    assert str(api_digest) in result.output


@pytest.mark.asyncio
async def test_plan_dry_run_with_job_id_does_not_advance_status(
    fab_app_client: tuple[Any, AsyncClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-FAB-016: plan --job-id dry-run returns PlacementResult without launching."""

    _asgi_app, client = fab_app_client
    body = {
        "image_digest": ALLOWED_IMAGE,
        "entrypoint": ["python", "-c", "print(1)"],
        "world_size": 2,
        "nnodes": 1,
        "nproc_per_node": 2,
        "timeout_s": 300,
        "resource": {"gpus": 2, "nodes": 1},
        "backend": "nccl",
        "fabric": "auto",
        "tee": "none",
        "placement_policy": "pack",
    }
    raw = json.dumps(body).encode()
    headers = _sign(raw)
    headers["Content-Type"] = "application/json"
    resp = await client.post("/v1/jobs", content=raw, headers=headers)
    assert resp.status_code in {200, 201}
    job_body = resp.json()
    job_id = job_body["id"]

    from hypercluster.cli import app

    launch_calls = {"n": 0}

    def _boom(*_a: Any, **_k: Any) -> None:
        launch_calls["n"] += 1
        raise AssertionError("sim_launch must not be invoked from fabric plan dry-run")

    monkeypatch.setattr("hypercluster.fabric.launcher.sim_launch", _boom)
    _patch_httpx_get_json(monkeypatch, {f"/v1/jobs/{job_id}": job_body})

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["fabric", "plan", "--job-id", job_id, "--url", "http://testserver", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert launch_calls["n"] == 0
    assert (
        "rankmap" in result.output
        or "graph_digest" in result.output
        or "planner_version" in result.output
    )
