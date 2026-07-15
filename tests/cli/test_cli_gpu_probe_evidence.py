"""CLI GPU probe + evidence show/list (VAL-GPU-040 / VAL-GPU-041).

FakeSsh only (no real SSH, no product Verda, no set_weights).
Exit codes: 0 pass, 2 failed checks (design §5).
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
OWNER_HK = "cli-gpu-probe-owner-hotkey-aaaaaaaaaaaaaaaaaaaaaa"
FOREIGN_HK = "cli-gpu-probe-foreign-hotkey-bbbbbbbbbbbbbbbbbbbb"

runner = CliRunner()


@pytest.fixture
def live_api(settings_factory, tmp_path: Path) -> Any:
    """Live uvicorn with FakeSsh allowed for CLI probe coverage."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'cli-gpu-probe.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        node_liveness_seconds=120,
        ssh_transport="fake",
        allow_fake_ssh=True,
        fake_ssh_fixture="pass_all",
        require_docker_runtime=True,
        max_gpu_count=14,
        require_live_evidence=False,
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
    text = output.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    matches = list(re.finditer(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL))
    assert matches, f"no JSON in output:\n{output}"
    return json.loads(matches[-1].group(1))


def _register_node(
    base: str,
    *,
    hotkey: str = OWNER_HK,
    gpu_model: str = "1V100.6V",
    gpus: int = 1,
    ssh: str = "10.9.9.50:22",
) -> str:
    result = _invoke(
        [
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
    )
    assert result.exit_code == 0, result.output
    body = _json_blob(result.output)
    node_id = body.get("id")
    assert node_id, body
    return str(node_id)


def _api_get(base: str, path: str) -> dict[str, Any]:
    response = httpx.get(f"{base}{path}", timeout=10.0)
    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data, dict)
    return data


# ---------------------------------------------------------------------------
# VAL-GPU-040: probe-gpu exit 0 on pass_all, exit 2 on fatal fail
# ---------------------------------------------------------------------------


def test_nodes_probe_gpu_pass_all_exits_0(live_api: dict[str, Any]) -> None:
    """VAL-GPU-040: FakeSsh pass-all → CLI exit 0 + status=passed JSON."""

    base = live_api["base_url"]
    node_id = _register_node(base)

    result = _invoke(
        [
            "nodes",
            "probe-gpu",
            node_id,
            "--mode",
            "full",
            "--fixture",
            "pass_all",
            "--hotkey",
            OWNER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert result.exit_code == 0, result.output
    body = _json_blob(result.output)
    assert body.get("status") == "passed"
    evidence_id = body.get("evidence_id") or body.get("id")
    assert evidence_id
    assert int(body.get("checks_failed") or 0) == 0
    uuids = body.get("gpu_uuids") or []
    assert uuids or int(body.get("measured_gpu_count") or 0) >= 1
    text = json.dumps(body)
    assert "BEGIN PRIVATE KEY" not in text
    assert "verda" not in text.lower()


def test_nodes_probe_gpu_fatal_fail_exits_2(live_api: dict[str, Any]) -> None:
    """VAL-GPU-040: FakeSsh no_gpu fatal fail → CLI exit 2."""

    base = live_api["base_url"]
    node_id = _register_node(base, ssh="10.9.9.51:22")

    result = _invoke(
        [
            "nodes",
            "probe-gpu",
            node_id,
            "--fixture",
            "no_gpu",
            "--hotkey",
            OWNER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert result.exit_code == 2, result.output
    body = _json_blob(result.output)
    assert body.get("status") in {"failed", "error"}
    evidence_id = body.get("evidence_id") or body.get("id")
    assert evidence_id
    # nvidia_smi_list should be among failed fatals when checks present.
    checks = body.get("checks") or []
    if checks:
        by_id = {c.get("id"): c for c in checks if isinstance(c, dict)}
        if "nvidia_smi_list" in by_id:
            assert by_id["nvidia_smi_list"].get("passed") is False


def test_nodes_probe_gpu_requires_auth(live_api: dict[str, Any]) -> None:
    """Signed mutate path fails closed without hotkey/token."""

    base = live_api["base_url"]
    node_id = _register_node(base, ssh="10.9.9.52:22")
    result = _invoke(
        [
            "nodes",
            "probe-gpu",
            node_id,
            "--url",
            base,
            "--json",
        ]
    )
    assert result.exit_code == 2, result.output


def test_nodes_probe_gpu_sim_pass_all_and_fail(live_api: dict[str, Any]) -> None:
    """probe-gpu-sim convenience: --pass-all exit 0; --fail maps to fixture fail."""

    base = live_api["base_url"]
    node_ok = _register_node(base, ssh="10.9.9.53:22")
    node_fail = _register_node(base, ssh="10.9.9.54:22")

    ok = _invoke(
        [
            "nodes",
            "probe-gpu-sim",
            node_ok,
            "--pass-all",
            "--hotkey",
            OWNER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert ok.exit_code == 0, ok.output
    assert _json_blob(ok.output).get("status") == "passed"

    # --fail nvidia_smi_list uses the no_gpu FakeSsh fixture (design helper).
    bad = _invoke(
        [
            "nodes",
            "probe-gpu-sim",
            node_fail,
            "--fail",
            "nvidia_smi_list",
            "--hotkey",
            OWNER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert bad.exit_code == 2, bad.output
    assert _json_blob(bad.output).get("status") in {"failed", "error"}


# ---------------------------------------------------------------------------
# VAL-GPU-041: evidence list / show / latest match API fields
# ---------------------------------------------------------------------------


def test_nodes_evidence_list_show_latest_match_api(live_api: dict[str, Any]) -> None:
    """VAL-GPU-041: CLI evidence show/list/latest field parity with GET API."""

    base = live_api["base_url"]
    node_id = _register_node(base, ssh="10.9.9.60:22")

    # Seed two probes so list has newest-first ordering.
    first = _invoke(
        [
            "nodes",
            "probe-gpu",
            node_id,
            "--mode",
            "full",
            "--fixture",
            "pass_all",
            "--hotkey",
            OWNER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert first.exit_code == 0, first.output
    first_body = _json_blob(first.output)
    first_id = first_body.get("evidence_id") or first_body.get("id")

    second = _invoke(
        [
            "nodes",
            "probe-gpu",
            node_id,
            "--mode",
            "quick",
            "--fixture",
            "pass_all",
            "--hotkey",
            OWNER_HK,
            "--token",
            TOKEN,
            "--url",
            base,
            "--json",
        ]
    )
    assert second.exit_code == 0, second.output
    second_body = _json_blob(second.output)
    second_id = second_body.get("evidence_id") or second_body.get("id")
    assert second_id and second_id != first_id

    # list
    listed = _invoke(
        [
            "nodes",
            "evidence",
            "list",
            node_id,
            "--url",
            base,
            "--json",
        ]
    )
    assert listed.exit_code == 0, listed.output
    list_payload = _json_blob(listed.output)
    items = list_payload.get("items") if isinstance(list_payload, dict) else list_payload
    assert isinstance(items, list)
    assert len(items) >= 2
    cli_ids = [it.get("evidence_id") or it.get("id") for it in items]
    assert cli_ids[0] == second_id

    api_list = _api_get(base, f"/v1/nodes/{node_id}/probes/gpu")
    api_items = api_list.get("items") or []
    api_ids = [it.get("evidence_id") or it.get("id") for it in api_items]
    assert cli_ids == api_ids
    for cli_it, api_it in zip(items, api_items, strict=False):
        for key in ("status", "gpu_uuids", "evidence_id", "id"):
            if key in api_it:
                assert cli_it.get(key) == api_it.get(key), (key, cli_it, api_it)
        cli_dig = cli_it.get("digests") or {}
        api_dig = api_it.get("digests") or {}
        for dkey in ("evidence_sha256", "inventory_sha256"):
            if api_dig.get(dkey):
                assert cli_dig.get(dkey) == api_dig.get(dkey)

    # latest
    latest = _invoke(
        [
            "nodes",
            "evidence",
            "latest",
            node_id,
            "--url",
            base,
            "--json",
        ]
    )
    assert latest.exit_code == 0, latest.output
    latest_body = _json_blob(latest.output)
    latest_id = latest_body.get("evidence_id") or latest_body.get("id")
    assert latest_id == second_id
    api_latest = _api_get(base, f"/v1/nodes/{node_id}/probes/gpu/latest")
    assert (api_latest.get("evidence_id") or api_latest.get("id")) == latest_id
    assert latest_body.get("status") == api_latest.get("status")
    assert list(latest_body.get("gpu_uuids") or []) == list(api_latest.get("gpu_uuids") or [])
    lat_dig = latest_body.get("digests") or {}
    api_lat_dig = api_latest.get("digests") or {}
    if api_lat_dig.get("evidence_sha256"):
        assert lat_dig.get("evidence_sha256") == api_lat_dig.get("evidence_sha256")

    # show by evidence id (global when node not required / optional --node-id)
    shown = _invoke(
        [
            "nodes",
            "evidence",
            "show",
            str(second_id),
            "--url",
            base,
            "--json",
        ]
    )
    assert shown.exit_code == 0, shown.output
    show_body = _json_blob(shown.output)
    assert (show_body.get("evidence_id") or show_body.get("id")) == second_id
    assert isinstance(show_body.get("checks"), list)
    assert len(show_body["checks"]) >= 1
    for c in show_body["checks"]:
        assert "id" in c
        assert "fatal" in c
        assert "passed" in c

    api_show = _api_get(base, f"/v1/evidence/gpu/{second_id}")
    assert (api_show.get("evidence_id") or api_show.get("id")) == second_id
    assert show_body.get("status") == api_show.get("status")
    show_dig = show_body.get("digests") or {}
    api_show_dig = api_show.get("digests") or {}
    if api_show_dig.get("evidence_sha256"):
        assert show_dig.get("evidence_sha256") == api_show_dig.get("evidence_sha256")
    show_uuid = show_body.get("gpu_uuids")
    api_uuid = api_show.get("gpu_uuids")
    if show_uuid is not None and api_uuid is not None:
        assert list(show_uuid) == list(api_uuid)

    text = json.dumps(show_body)
    assert "BEGIN PRIVATE KEY" not in text
    assert "BEGIN RSA PRIVATE KEY" not in text


def test_nodes_evidence_list_empty_for_unprobed(live_api: dict[str, Any]) -> None:
    """Empty list for never-probed node — not forged rows."""

    base = live_api["base_url"]
    node_id = _register_node(base, ssh="10.9.9.61:22")
    listed = _invoke(
        [
            "nodes",
            "evidence",
            "list",
            node_id,
            "--url",
            base,
            "--json",
        ]
    )
    assert listed.exit_code == 0, listed.output
    payload = _json_blob(listed.output)
    items = payload.get("items") if isinstance(payload, dict) else payload
    assert items == []


def test_cli_help_includes_probe_and_evidence() -> None:
    """Subcommands discoverable via --help (packaging sanity)."""

    nodes_help = _invoke(["nodes", "--help"])
    assert nodes_help.exit_code == 0, nodes_help.output
    assert "probe-gpu" in nodes_help.output
    assert "evidence" in nodes_help.output

    ev_help = _invoke(["nodes", "evidence", "--help"])
    assert ev_help.exit_code == 0, ev_help.output
    for cmd in ("list", "show", "latest"):
        assert cmd in ev_help.output
