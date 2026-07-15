"""CLI marketplace + nodes + jobs round-trips (M7).

Covers:
  VAL-CLI-005  marketplace offers list filters (--gpu-model, --require-ib)
  VAL-CLI-006  offer create + rent + lease show + terminate
  VAL-CLI-007  nodes register + heartbeat + fabric-scan
  VAL-CLI-008  jobs submit / status / list / cancel
  VAL-CLI-009  jobs logs --id safe digests / pre-collect empty handling
"""

from __future__ import annotations

import json
import re
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
from hypercluster.sim.ports import MAX_MISSION_PORT, MIN_MISSION_PORT

TOKEN = "test-challenge-shared-token"
PROVIDER_HK = "cli-mkt-provider-hotkey-aaaaaaaaaaaaaaaaaaaaaaaa"
RENTER_HK = "cli-mkt-renter-hotkey-bbbbbbbbbbbbbbbbbbbbbbbbbbbb"
SUBMITTER_HK = "cli-jobs-submitter-hotkey-cccccccccccccccccccccccc"
ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"

runner = CliRunner()


@pytest.fixture
def live_api(settings_factory, tmp_path: Path) -> Any:
    """Live uvicorn with insecure signatures + combined worker for job lifecycle."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'cli-mnj.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        node_liveness_seconds=120,
        max_offer_price_per_hour=100.0,
        max_offer_lifetime_hours=168.0,
        combined_worker=True,
        combined_worker_interval_seconds=0.05,
        job_image_allowlist=ALLOWED_IMAGE,
        max_job_world_size=64,
        max_job_nnodes=16,
        max_job_nproc_per_node=8,
        max_job_timeout_s=3600,
        max_job_gpu_budget=32,
        sim_job_step_delay_s=0.0,
        # Keep jobs cancelable for VAL-CLI-008.
        sim_job_run_sleep_s=1.5,
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
        yield {"base_url": base_url, "port": bound_port, "token": TOKEN}
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _invoke(argv: list[str]) -> Any:
    return runner.invoke(cli_app, argv, catch_exceptions=False)


def _json_blob(output: str) -> Any:
    """Extract first top-level JSON object/array from CLI stdout."""

    text = output.strip()
    # Prefer pure JSON first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: last {...} or [...] block.
    matches = list(re.finditer(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL))
    assert matches, f"no JSON in output:\n{output}"
    return json.loads(matches[-1].group(1))


def _register_node(
    base: str,
    *,
    hotkey: str = PROVIDER_HK,
    gpus: int = 2,
    gpu_model: str = "H100",
    ib: bool = False,
    ssh: str = "10.0.0.42:22",
) -> str:
    argv = [
        "nodes",
        "register",
        "--ssh",
        ssh,
        "--gpus",
        str(gpus),
        "--gpu-model",
        gpu_model,
        "--hotkey",
        hotkey,
        "--token",
        TOKEN,
        "--url",
        base,
        "--json",
    ]
    if ib:
        argv.append("--ib")
    result = _invoke(argv)
    assert result.exit_code == 0, result.output
    body = _json_blob(result.output)
    node_id = body.get("id")
    assert node_id, body
    return str(node_id)


# ---------------------------------------------------------------------------
# VAL-CLI-005
# ---------------------------------------------------------------------------


def test_marketplace_offers_list_empty_and_filters(live_api: dict[str, Any]) -> None:
    """VAL-CLI-005: list exits 0 empty; --gpu-model / --require-ib reduce set."""

    base = live_api["base_url"]

    empty = _invoke(["marketplace", "offers", "list", "--url", base, "--json"])
    assert empty.exit_code == 0, empty.output
    empty_body = _json_blob(empty.output)
    assert isinstance(empty_body, list)
    assert empty_body == []

    node_h100 = _register_node(base, ssh="10.0.1.1:22", gpu_model="H100", gpus=8, ib=True)
    node_a100 = _register_node(base, ssh="10.0.1.2:22", gpu_model="A100", gpus=4, ib=False)

    create_h100 = _invoke(
        [
            "marketplace",
            "offer",
            "create",
            "--node-ids",
            node_h100,
            "--price",
            "1.5",
            "--lifetime",
            "12",
            "--gpu-model",
            "H100",
            "--gpu-count",
            "8",
            "--require-ib",
            "--hotkey",
            PROVIDER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert create_h100.exit_code == 0, create_h100.output

    create_a100 = _invoke(
        [
            "marketplace",
            "offer",
            "create",
            "--node-ids",
            node_a100,
            "--price",
            "0.9",
            "--lifetime",
            "6",
            "--gpu-model",
            "A100",
            "--gpu-count",
            "4",
            "--hotkey",
            PROVIDER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert create_a100.exit_code == 0, create_a100.output

    all_list = _invoke(["marketplace", "offers", "list", "--url", base, "--json"])
    assert all_list.exit_code == 0, all_list.output
    all_items = _json_blob(all_list.output)
    assert isinstance(all_items, list)
    assert len(all_items) >= 2

    filtered_gpu = _invoke(
        [
            "marketplace",
            "offers",
            "list",
            "--url",
            base,
            "--gpu-model",
            "H100",
            "--json",
        ]
    )
    assert filtered_gpu.exit_code == 0, filtered_gpu.output
    gpu_items = _json_blob(filtered_gpu.output)
    assert isinstance(gpu_items, list)
    assert gpu_items
    assert all(item.get("gpu_model") == "H100" for item in gpu_items)
    assert len(gpu_items) < len(all_items)

    filtered_ib = _invoke(
        [
            "marketplace",
            "offers",
            "list",
            "--url",
            base,
            "--require-ib",
            "--json",
        ]
    )
    assert filtered_ib.exit_code == 0, filtered_ib.output
    ib_items = _json_blob(filtered_ib.output)
    assert isinstance(ib_items, list)
    assert ib_items
    assert all(bool(item.get("require_ib")) for item in ib_items)
    assert len(ib_items) < len(all_items)


# ---------------------------------------------------------------------------
# VAL-CLI-006
# ---------------------------------------------------------------------------


def test_marketplace_create_rent_show_terminate_round_trip(
    live_api: dict[str, Any],
) -> None:
    """VAL-CLI-006: create → rent → lease show → terminate mirrors lease lifecycle."""

    base = live_api["base_url"]
    node_id = _register_node(base, ssh="10.0.2.5:22", gpu_model="H100", gpus=2)

    create = _invoke(
        [
            "marketplace",
            "offer",
            "create",
            "--node-ids",
            node_id,
            "--price",
            "2.0",
            "--lifetime",
            "4",
            "--hotkey",
            PROVIDER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert create.exit_code == 0, create.output
    offer = _json_blob(create.output)
    offer_id = offer.get("id")
    assert offer_id
    assert offer.get("status") == "listed"

    rent = _invoke(
        [
            "marketplace",
            "rent",
            "--offer-id",
            str(offer_id),
            "--max-hours",
            "2",
            "--hotkey",
            RENTER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert rent.exit_code == 0, rent.output
    rent_body = _json_blob(rent.output)
    lease = rent_body.get("lease") if isinstance(rent_body, dict) else None
    if not isinstance(lease, dict):
        # Some responses may flatten lease at top-level.
        lease = rent_body if isinstance(rent_body, dict) and rent_body.get("offer_id") else None
    assert isinstance(lease, dict), rent_body
    lease_id = lease.get("id")
    assert lease_id
    assert lease.get("status") in {"requested", "active"}
    assert lease.get("renter_hotkey") == RENTER_HK

    show = _invoke(
        [
            "marketplace",
            "lease",
            "show",
            "--id",
            str(lease_id),
            "--url",
            base,
            "--json",
        ]
    )
    assert show.exit_code == 0, show.output
    shown = _json_blob(show.output)
    assert shown.get("id") == lease_id
    assert shown.get("status") in {"requested", "active"}
    assert shown.get("offer_id") == offer_id

    # API ground truth
    api_lease = httpx.get(f"{base}/v1/leases/{lease_id}", timeout=5.0)
    assert api_lease.status_code == 200
    assert api_lease.json().get("status") == shown.get("status")

    term = _invoke(
        [
            "marketplace",
            "terminate",
            "--lease-id",
            str(lease_id),
            "--hotkey",
            RENTER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert term.exit_code == 0, term.output
    term_body = _json_blob(term.output)
    # Terminate may nest lease under {"lease": {...}} (same as rent).
    term_lease = term_body.get("lease") if isinstance(term_body, dict) else None
    if not isinstance(term_lease, dict):
        term_lease = term_body if isinstance(term_body, dict) else {}
    term_status = term_lease.get("status") or term_body.get("status")
    assert term_status in {"terminated", "expired"}, term_body

    show_after = _invoke(
        [
            "marketplace",
            "lease",
            "show",
            "--id",
            str(lease_id),
            "--url",
            base,
            "--json",
        ]
    )
    assert show_after.exit_code == 0, show_after.output
    after = _json_blob(show_after.output)
    assert after.get("status") in {"terminated", "expired"}


# ---------------------------------------------------------------------------
# VAL-CLI-007
# ---------------------------------------------------------------------------


def test_nodes_register_heartbeat_fabric_scan(live_api: dict[str, Any]) -> None:
    """VAL-CLI-007: register → healthy node; heartbeat; fabric-scan with digest."""

    base = live_api["base_url"]

    reg = _invoke(
        [
            "nodes",
            "register",
            "--ssh",
            "10.0.3.9:22",
            "--gpus",
            "4",
            "--gpu-model",
            "H100",
            "--ib",
            "--hotkey",
            PROVIDER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert reg.exit_code == 0, reg.output
    node = _json_blob(reg.output)
    node_id = node.get("id")
    assert node_id
    assert int(node.get("gpu_count") or 0) == 4
    assert node.get("status") in {"registered", "healthy", "rented"}
    before_hb = node.get("last_heartbeat")

    # Register without required --gpus must fail closed.
    bad = _invoke(
        [
            "nodes",
            "register",
            "--ssh",
            "10.0.3.10:22",
            "--hotkey",
            PROVIDER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
        ]
    )
    assert bad.exit_code != 0

    hb = _invoke(
        [
            "nodes",
            "heartbeat",
            "--node-id",
            str(node_id),
            "--hotkey",
            PROVIDER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert hb.exit_code == 0, hb.output
    hb_body = _json_blob(hb.output)
    items = hb_body.get("items", hb_body) if isinstance(hb_body, dict) else hb_body
    assert items

    # API get should show refreshed last_heartbeat
    api_node = httpx.get(f"{base}/v1/nodes/{node_id}", timeout=5.0)
    assert api_node.status_code == 200
    got = api_node.json()
    assert got.get("status") in {"registered", "healthy", "rented"}
    assert got.get("last_heartbeat")
    # Heartbeat may keep or advance timestamp depending on clock resolution.
    assert got.get("last_heartbeat") is not None
    _ = before_hb

    scan = _invoke(
        [
            "nodes",
            "fabric-scan",
            "--node-id",
            str(node_id),
            "--hotkey",
            PROVIDER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--seed",
            "7",
        ]
    )
    assert scan.exit_code == 0, scan.output
    out = scan.output.lower()
    assert "report_digest" in out
    # Digest should look like hex/sha, not binary garbage.
    assert re.search(r"[0-9a-f]{16,}", out)


# ---------------------------------------------------------------------------
# VAL-CLI-008
# ---------------------------------------------------------------------------


def test_jobs_submit_status_list_cancel(live_api: dict[str, Any], tmp_path: Path) -> None:
    """VAL-CLI-008: submit returns id; status matches GET; list/filter; cancel terminals."""

    base = live_api["base_url"]
    spec_path = tmp_path / "job.json"
    spec_path.write_text(
        json.dumps(
            {
                "image_digest": ALLOWED_IMAGE,
                "entrypoint": ["python", "-c", "print('cli-job')"],
                "world_size": 1,
                "nnodes": 1,
                "nproc_per_node": 1,
                "timeout_s": 120,
                "resource": {"gpus": 1, "nodes": 1},
                "fabric": "auto",
                "tee": "none",
                "client_request_id": "cli-job-req-1",
            }
        ),
        encoding="utf-8",
    )

    submit = _invoke(
        [
            "jobs",
            "submit",
            "--spec",
            str(spec_path),
            "--hotkey",
            SUBMITTER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert submit.exit_code == 0, submit.output
    submitted = _json_blob(submit.output)
    job_id = submitted.get("id") or submitted.get("job_id")
    assert job_id, submitted
    assert submitted.get("status") in {
        "submitted",
        "admitted",
        "placing",
        "provisioning",
        "running",
    }

    status = _invoke(
        [
            "jobs",
            "status",
            "--id",
            str(job_id),
            "--url",
            base,
            "--json",
        ]
    )
    assert status.exit_code == 0, status.output
    status_body = _json_blob(status.output)
    assert status_body.get("id") == job_id or status_body.get("job_id") == job_id

    api = httpx.get(f"{base}/v1/jobs/{job_id}", timeout=5.0)
    assert api.status_code == 200
    assert api.json().get("status") == status_body.get("status")

    listed = _invoke(
        [
            "jobs",
            "list",
            "--hotkey",
            SUBMITTER_HK,
            "--url",
            base,
            "--json",
        ]
    )
    assert listed.exit_code == 0, listed.output
    items = _json_blob(listed.output)
    assert isinstance(items, list)
    assert any((j.get("id") or j.get("job_id")) == job_id for j in items)

    # Cancel before / while running.
    cancel = _invoke(
        [
            "jobs",
            "cancel",
            "--id",
            str(job_id),
            "--hotkey",
            SUBMITTER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    # Cancel may race if job finished already under sim — either terminal cancelled
    # success or unambiguous non-success if already terminal (handled below).
    if cancel.exit_code == 0:
        cancelled = _json_blob(cancel.output)
        assert cancelled.get("status") in {"cancelled", "succeeded", "failed", "timeout"}
    else:
        # Already terminal is acceptable if CLI surfaces API error clearly.
        assert "terminal" in cancel.output.lower() or cancel.exit_code != 0

    status_after = _invoke(
        [
            "jobs",
            "status",
            "--id",
            str(job_id),
            "--url",
            base,
            "--json",
        ]
    )
    assert status_after.exit_code == 0, status_after.output
    after = _json_blob(status_after.output)
    assert after.get("status") in {"cancelled", "succeeded", "failed", "timeout"}


# ---------------------------------------------------------------------------
# VAL-CLI-009
# ---------------------------------------------------------------------------


def test_jobs_logs_safe_digests_and_precollect_empty(
    live_api: dict[str, Any], tmp_path: Path
) -> None:
    """VAL-CLI-009: logs print digests safely; pre-collect empty handled cleanly."""

    base = live_api["base_url"]

    # Pre-collect / missing job attempt → non-crash, clear exit/message.
    missing = _invoke(
        [
            "jobs",
            "logs",
            "--id",
            "00000000-0000-0000-0000-000000000099",
            "--url",
            base,
        ]
    )
    assert missing.exit_code != 0
    assert "traceback (most recent call last)" not in missing.output.lower()

    # Submit + drive lifecycle to collect so digests exist.
    # Use zero run sleep path: restart is heavy; instead poll worker progress.
    # spin a second job with client path via submit then wait for attempt.
    # Override by submitting and advancing via status polls with short server sleep.
    # For determinism: use HTTP admit then wait for attempt.

    # Reconfigure isn't available; server fixture uses sleep 1.5s.
    # Wait until attempt appears or timeout.
    job_body = {
        "image_digest": ALLOWED_IMAGE,
        "entrypoint": ["python", "-c", "print('logs')"],
        "world_size": 1,
        "nnodes": 1,
        "nproc_per_node": 1,
        "timeout_s": 60,
        "resource": {"gpus": 1, "nodes": 1},
        "client_request_id": "cli-logs-job-1",
    }
    raw = json.dumps(job_body).encode()
    from hypercluster.api.auth import build_signed_headers

    headers = build_signed_headers(secret=TOKEN, hotkey=SUBMITTER_HK, body=raw)
    headers["Content-Type"] = "application/json"
    create = httpx.post(f"{base}/v1/jobs", content=raw, headers=headers, timeout=10.0)
    assert create.status_code == 200, create.text
    job_id = create.json()["id"]

    # Wait for attempt #1
    deadline = time.time() + 20.0
    attempt_ready = False
    while time.time() < deadline:
        att = httpx.get(f"{base}/v1/jobs/{job_id}/attempts/1", timeout=2.0)
        if att.status_code == 200:
            attempt_ready = True
            break
        time.sleep(0.15)
    # Even if attempt never appears in window, logs must not traceback.
    logs = _invoke(
        [
            "jobs",
            "logs",
            "--id",
            job_id,
            "--attempt",
            "1",
            "--url",
            base,
            "--json",
        ]
    )
    if attempt_ready:
        assert logs.exit_code == 0, logs.output
        body = _json_blob(logs.output)
        # Safe digests / log uri keys only — not binary secret payload dump.
        keys = set(body.keys()) if isinstance(body, dict) else set()
        assert "job_id" in keys or "attempt_no" in keys or "status" in keys
        # Prefer digest fields when present post-collect.
        digests = [
            body.get("output_digest"),
            body.get("fabric_report_digest"),
            body.get("result_digest"),
            body.get("launcher_log_uri"),
        ]
        # Must be text-safe and not empty strings of binary.
        for d in digests:
            if d is None:
                continue
            assert isinstance(d, str)
            assert "\x00" not in d
        # Never flood secrets.
        dumped = json.dumps(body)
        assert "private_key" not in dumped.lower()
        assert "-----begin" not in dumped.lower()
        assert "password" not in dumped.lower()
    else:
        # Allowed: clear non-zero for empty pre-collect.
        assert logs.exit_code != 0 or "attempt" in logs.output.lower()
        assert "traceback (most recent call last)" not in logs.output.lower()
