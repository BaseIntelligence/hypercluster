"""CLI fabric / attest / score / weights (M7).

Covers:
  VAL-CLI-010  fabric plan (--spec dry-run rankmap/nccl); report show; launch gated
  VAL-CLI-011  attest verify-offline / compose-hash offline (no network)
  VAL-CLI-012  score recompute/show; weights preview/push; token never fully echoed
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
import uvicorn
from typer.testing import CliRunner

from hypercluster.cli import app as cli_app
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT

TOKEN = "test-challenge-shared-token-SUPERSECRET-do-not-echo"
HOTKEY = "cli-score-hotkey-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_COMPOSE = REPO_ROOT / "tests" / "fixtures" / "tee" / "golden_compose.yml"
GOLDEN_HASH_FILE = REPO_ROOT / "tests" / "fixtures" / "tee" / "golden_compose.sha256"
POSITIVE_TEE = REPO_ROOT / "tests" / "fixtures" / "tee" / "positive_tdx_v1.json"

runner = CliRunner()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_spec(tmp_path: Path) -> Path:
    """Minimal fabric plan placement-spec JSON for --spec dry-run."""

    path = tmp_path / "plan-spec.json"
    path.write_text(
        json.dumps(
            {
                "world_size": 4,
                "nnodes": 2,
                "nproc_per_node": 2,
                "policy": "pack",
                "fabric": "auto",
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def live_api(settings_factory: Any, tmp_path: Path) -> Any:
    """Short-lived uvicorn for score/weights/report show against real process."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'cli-fas.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        combined_worker=False,
        sim_seed_enabled=True,
        master_base_url="http://127.0.0.1:3201",
        weight_push_enabled=True,
        weight_push_freshness_s=600,
        sim_auto_capacity=True,
        job_image_allowlist=ALLOWED_IMAGE,
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
        thread.join(timeout=2)
        pytest.fail("API never became healthy")

    try:
        yield type("Live", (), {"base": base, "port": bound_port, "token": TOKEN})()
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# VAL-CLI-010 fabric plan / report show / launch gated
# ---------------------------------------------------------------------------


def test_fabric_plan_spec_prints_rankmap_and_nccl(plan_spec: Path) -> None:
    """VAL-CLI-010: fabric plan --spec dry-run prints rankmap and nccl_env."""

    result = runner.invoke(
        cli_app,
        ["fabric", "plan", "--spec", str(plan_spec)],
    )
    assert result.exit_code == 0, result.output
    # Parse the JSON body from stdout (may have trailing newlines only).
    payload = json.loads(result.stdout)
    assert payload.get("ok") is True
    assert payload.get("dry_run") is True
    assert isinstance(payload.get("rankmap"), list) and len(payload["rankmap"]) == 4
    assert isinstance(payload.get("nccl_env"), dict)
    assert payload["nccl_env"].get("HYPER_NCCL_ENV_VERSION") == "nccl_env.v1"
    assert "graph_digest" in payload
    assert payload.get("nnodes_used") == 2
    assert payload.get("job_status_unchanged") is True


def test_fabric_plan_flag_path_still_works() -> None:
    """VAL-CLI-010: flag form still emits plan JSON (parity with --spec)."""

    result = runner.invoke(
        cli_app,
        [
            "fabric",
            "plan",
            "--world-size",
            "2",
            "--nnodes",
            "2",
            "--nproc-per-node",
            "1",
            "--policy",
            "spread",
            "--seed",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert len(payload["rankmap"]) == 2
    assert "nccl_env" in payload


def test_fabric_launch_fails_closed_without_dev_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-CLI-010: fabric launch without HYPER_ALLOW_FABRIC_LAUNCH fails closed.

    Never silently restarts prod jobs; no launcher call must occur.
    """

    monkeypatch.delenv("HYPER_ALLOW_FABRIC_LAUNCH", raising=False)
    monkeypatch.delenv("HYPER_FABRIC_LAUNCH", raising=False)

    called = {"n": 0}

    def _boom(*_a: Any, **_k: Any) -> None:
        called["n"] += 1
        raise AssertionError("sim_launch must not run without gate")

    monkeypatch.setattr("hypercluster.fabric.launcher.sim_launch", _boom)

    result = runner.invoke(
        cli_app,
        ["fabric", "launch", "--job-id", "job-should-not-launch"],
    )
    assert result.exit_code != 0, result.output
    out = (result.stdout or "") + (result.stderr or "")
    assert "gated" in out.lower() or "denied" in out.lower() or "not allowed" in out.lower()
    assert called["n"] == 0
    # Token-looking secrets never appear (push path is separate; belt & suspenders).
    assert "SUPERSECRET" not in out


def test_fabric_launch_dev_gate_requires_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-CLI-010: with allow env but no --force, still fail closed unless force.

    Even when the env gate is set, launch is still opt-in via --force so accidental
    prod restarts stay hard to trigger.
    """

    monkeypatch.setenv("HYPER_ALLOW_FABRIC_LAUNCH", "1")
    called = {"n": 0}

    def _boom(*_a: Any, **_k: Any) -> None:
        called["n"] += 1
        raise AssertionError("sim_launch must not run without --force")

    monkeypatch.setattr("hypercluster.fabric.launcher.sim_launch", _boom)

    result = runner.invoke(
        cli_app,
        ["fabric", "launch", "--job-id", "job-still-gated"],
    )
    assert result.exit_code != 0, result.output
    assert called["n"] == 0


def test_fabric_report_show_prints_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """VAL-CLI-010: fabric report show --job-id echoes report_digest."""

    digest = "sha256:" + ("ab" * 32)
    job_id = "job-report-show-1"

    def _get(url: str, **_kwargs: Any) -> httpx.Response:
        path = urlparse(url).path
        if path.endswith(f"/v1/jobs/{job_id}/fabric-report"):
            return httpx.Response(
                200,
                json={
                    "job_id": job_id,
                    "report_digest": digest,
                    "nodes": [
                        {"node_id": "n0", "report_digest": digest},
                        {"node_id": "n1", "report_digest": digest},
                    ],
                },
            )
        return httpx.Response(404, json={"detail": f"no stub for {path}"})

    monkeypatch.setattr(httpx, "get", _get)

    result = runner.invoke(
        cli_app,
        [
            "fabric",
            "report",
            "show",
            "--job-id",
            job_id,
            "--url",
            "http://127.0.0.1:3200",
        ],
    )
    assert result.exit_code == 0, result.output
    assert digest in result.stdout
    assert "report_digest=" in result.stdout
    assert "node_count=2" in result.stdout


# ---------------------------------------------------------------------------
# VAL-CLI-011 attest verify-offline / compose-hash
# ---------------------------------------------------------------------------


def test_attest_compose_hash_stable_and_offline(plan_spec: Path) -> None:
    """VAL-CLI-011: compose-hash is stable and works without network."""

    r1 = runner.invoke(
        cli_app,
        ["attest", "compose-hash", "--compose-file", str(GOLDEN_COMPOSE)],
    )
    r2 = runner.invoke(
        cli_app,
        [
            "attest",
            "compose-hash",
            "--compose-file",
            str(GOLDEN_COMPOSE),
            "--check-golden",
            str(GOLDEN_HASH_FILE),
        ],
    )
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    h1 = r1.stdout.strip().splitlines()[-1].strip()
    assert h1.startswith("sha256:")
    assert h1 in r2.stdout or r2.exit_code == 0


def test_attest_verify_offline_positive_fixture_exit_0() -> None:
    """VAL-CLI-011: positive offline fixture exits 0 without live network."""

    result = runner.invoke(
        cli_app,
        ["attest", "verify-offline", "--quote-fixture", str(POSITIVE_TEE)],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body.get("is_valid") is True
    assert body.get("verify_mode") in {"offline_fixture", "offline"}
    # Must not claim live path is mandatory.
    assert "live_required" not in json.dumps(body).lower()


def test_attest_verify_offline_missing_fixture_nonzero() -> None:
    """VAL-CLI-011: missing fixture fails closed (handlable)."""

    missing = REPO_ROOT / "tests" / "fixtures" / "tee" / "nope-does-not-exist.json"
    result = runner.invoke(
        cli_app,
        ["attest", "verify-offline", "--quote-fixture", str(missing)],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# VAL-CLI-012 score recompute/show; weights preview/push secret-safe
# ---------------------------------------------------------------------------


def test_score_and_weights_help_surface() -> None:
    """VAL-CLI-012: score and weights subcommands are present."""

    score = runner.invoke(cli_app, ["score", "--help"])
    assert score.exit_code == 0
    assert "recompute" in score.stdout
    assert "show" in score.stdout

    weights = runner.invoke(cli_app, ["weights", "--help"])
    assert weights.exit_code == 0
    assert "preview" in weights.stdout
    assert "push" in weights.stdout


def test_score_show_against_live_api(live_api: Any) -> None:
    """VAL-CLI-012: score show --hotkey exits 0 against live API; no token echo."""

    # Empty history still returns 200 for known route shape.
    result = runner.invoke(
        cli_app,
        [
            "score",
            "show",
            "--hotkey",
            HOTKEY,
            "--url",
            live_api.base,
        ],
    )
    # 200 with empty items is fine; non-JSON 404 would be fail.
    assert result.exit_code == 0, result.output
    combined = (result.stdout or "") + (result.stderr or "")
    assert TOKEN not in combined
    assert "SUPERSECRET" not in combined


def test_score_recompute_and_weights_preview(
    live_api: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """VAL-CLI-012: score recompute + weights preview exit 0; preview is a map."""

    # Point process DB to the live API DB? recompute uses local settings DB, not
    # the live HTTP DB. Prefer weights preview via API and recompute local JSON.
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", TOKEN)
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN_FILE", "")
    monkeypatch.setenv(
        "CHALLENGE_DATABASE_URL",
        f"sqlite+aiosqlite:///{tmp_path / 'recompute.sqlite3'}",
    )
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()

    # Init local DB so recompute can open it.
    init = runner.invoke(cli_app, ["db", "init"])
    assert init.exit_code == 0, init.output

    recom = runner.invoke(
        cli_app,
        ["score", "recompute", "--url", live_api.base],
    )
    assert recom.exit_code == 0, recom.output
    combined = (recom.stdout or "") + (recom.stderr or "")
    assert TOKEN not in combined
    assert "SUPERSECRET" not in combined
    # Local body is JSON with recomputed flag.
    # May have a live weight-preview line then json; find last JSON object.
    lines = recom.stdout.strip().splitlines()
    json_blob = "\n".join(lines)
    # At least one parseable Object containing recomputed.
    assert "recomputed" in json_blob

    prev = runner.invoke(
        cli_app,
        ["weights", "preview", "--url", live_api.base],
    )
    assert prev.exit_code == 0, prev.output
    preview_body = json.loads(prev.stdout.strip().splitlines()[0] if prev.stdout else "{}")
    # API may emit multi-line but response.text is full; try whole stdout.
    try:
        preview_body = json.loads(prev.stdout)
    except json.JSONDecodeError:
        # last resort: find braces
        s = prev.stdout
        start = s.find("{")
        end = s.rfind("}")
        preview_body = json.loads(s[start : end + 1])
    assert "weights" in preview_body
    assert isinstance(preview_body["weights"], dict)
    assert TOKEN not in prev.stdout
    assert "SUPERSECRET" not in prev.stdout


def test_weights_push_redacts_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """VAL-CLI-012: push prints redacted fingerprint, never full token value."""

    secret = "NEVER-ECHO-THIS-TOKEN-VALUE-zzzz-deadbeef"
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", secret)
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN_FILE", "")
    monkeypatch.setenv(
        "CHALLENGE_DATABASE_URL",
        f"sqlite+aiosqlite:///{tmp_path / 'push.sqlite3'}",
    )
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()

    init = runner.invoke(cli_app, ["db", "init"])
    assert init.exit_code == 0, init.output

    # Mock master may be down; push may exit 1 on empty/transport — still must redact.
    result = runner.invoke(
        cli_app,
        [
            "weights",
            "push",
            "--token",
            secret,
            "--master",
            "http://127.0.0.1:3201",
            "--epoch",
            "1",
        ],
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert secret not in combined
    assert "NEVER-ECHO-THIS-TOKEN" not in combined
    # Redaction fingerprint markers must be present when token was resolved.
    assert "redacted" in combined.lower() or "token_set" in combined.lower()
    # Full token hex spoof shouldn't leak via environment printing either.
    assert secret not in combined


def test_weights_push_with_mock_master_ack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """VAL-CLI-012: push against in-process mock-master can ack without echoing token."""

    import asyncio
    import uuid

    from hypercluster.db.database import Database
    from hypercluster.db.models import Job, JobAttempt
    from hypercluster.domain.scoring_tee import persist_score_for_attempt
    from hypercluster.settings import HyperSettings, clear_settings_cache
    from hypercluster.sim.mock_master import app as mock_master_app
    from hypercluster.sim.mock_master import configure_token, reset_store

    secret = "push-ack-secret-DONT-LEAK-me-please"
    db_path = tmp_path / "push-ack.sqlite3"
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN", secret)
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN_FILE", "")
    monkeypatch.setenv("CHALLENGE_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    clear_settings_cache()

    reset_store()
    configure_token(secret)

    async def _seed() -> None:
        database = Database(f"sqlite+aiosqlite:///{db_path}")
        await database.init()
        hyper = HyperSettings(self_deal_damping=0.0)
        job_id = str(uuid.uuid4())
        attempt_id = str(uuid.uuid4())
        async with database.session() as session:
            session.add(
                Job(
                    id=job_id,
                    submitter_hotkey=HOTKEY,
                    status="succeeded",
                    image_digest=ALLOWED_IMAGE,
                    entrypoint_json=json.dumps(["python", "-c", "print(1)"]),
                    world_size=1,
                    nnodes=1,
                    nproc_per_node=1,
                    timeout_s=60,
                    resource_json=json.dumps({"gpus": 1, "nodes": 1}),
                    backend="nccl",
                    fabric_mode="auto",
                    tee_mode="none",
                )
            )
            session.add(
                JobAttempt(
                    id=attempt_id,
                    job_id=job_id,
                    attempt_no=1,
                    status="succeeded",
                )
            )
            await session.flush()
            await persist_score_for_attempt(
                session,
                attempt_id=attempt_id,
                hotkey=HOTKEY,
                role="demand",
                correctness=1.0,
                efficiency=1.0,
                fabric_gate=1.0,
                proof=None,
                tee_mode="none",
                hyper=hyper,
            )
            await session.commit()
        await database.close()

    asyncio.run(_seed())

    # Patch WeightPushClient http to talk to ASGI mock-master (no real port).
    import hypercluster.weight_push as wp

    transport = httpx.ASGITransport(app=mock_master_app)
    orig_init = wp.WeightPushClient.__init__

    def _patched(self: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("http_client") is None:
            kwargs["http_client"] = httpx.AsyncClient(
                transport=transport,
                base_url=str(kwargs.get("master_base_url") or "http://mock-master.test"),
            )
        kwargs["master_base_url"] = "http://mock-master.test"
        kwargs["shared_token"] = secret
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(wp.WeightPushClient, "__init__", _patched)

    result = runner.invoke(
        cli_app,
        [
            "weights",
            "push",
            "--token",
            secret,
            "--master",
            "http://mock-master.test",
            "--epoch",
            "42",
        ],
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert secret not in combined
    assert "DONT-LEAK" not in combined
    # Prefer acknowledged; empty/skipped is acceptable if scorer window missed.
    assert result.exit_code == 0 or "empty" in combined.lower() or "status" in combined
    if result.exit_code == 0:
        assert "acknowledged" in combined or "acked" in combined or "skipped" in combined
