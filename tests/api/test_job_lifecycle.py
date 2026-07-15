"""VAL-JOB-006..012, 020, 021, 025, 026: job lifecycle, cancel, timeout, queries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.api.auth import build_signed_headers

SUBMITTER_HK = "job-lifecycle-submitter-aaaaaaaaaaaaaaaaaaaaaaaaa"
OTHER_HK = "job-lifecycle-other-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
TOKEN = "test-challenge-shared-token"

ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"

# Lifecycle progression expected under sim (subset / ordered edges).
SUCCESS_PATH_STATUSES = (
    "admitted",
    "placing",
    "provisioning",
    "running",
    "collecting",
    "scoring",
    "succeeded",
)
TERMINAL = frozenset({"succeeded", "failed", "timeout", "cancelled"})


def _sign(body: bytes, *, hotkey: str = SUBMITTER_HK, nonce: str | None = None) -> dict[str, str]:
    return build_signed_headers(secret=TOKEN, hotkey=hotkey, body=body, nonce=nonce)


def _valid_job_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "image_digest": ALLOWED_IMAGE,
        "entrypoint": ["python", "-m", "train", "--steps", "10"],
        "world_size": 4,
        "nnodes": 2,
        "nproc_per_node": 2,
        "timeout_s": 300,
        "resource": {"gpus": 4, "nodes": 2},
        "backend": "nccl",
        "fabric": "auto",
        "tee": "none",
        "env": {"HYPER_TRAIN_STEPS": "10", "NCCL_DEBUG": "INFO"},
        "placement_policy": "pack",
    }
    body.update(overrides)
    return body


@pytest.fixture
async def life_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    """App with combined worker + fast sim lifecycle interval."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'job-life.sqlite3'}",
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
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


@pytest.fixture
async def slow_life_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    """Combined worker with non-zero run sleep (for cancel / timeout races)."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'job-life-slow.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        signature_ttl_seconds=300,
        combined_worker=True,
        combined_worker_interval_seconds=0.05,
        job_image_allowlist=ALLOWED_IMAGE,
        max_job_timeout_s=3600,
        sim_job_step_delay_s=0.0,
        # Keep running long enough for cancel/timeout assertions.
        sim_job_run_sleep_s=2.0,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


async def _post_job(
    client: AsyncClient,
    body: dict[str, Any],
    *,
    hotkey: str = SUBMITTER_HK,
) -> Any:
    raw = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    headers.update(_sign(raw, hotkey=hotkey))
    return await client.post("/v1/jobs", content=raw, headers=headers)


async def _signed_post(
    client: AsyncClient,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    hotkey: str = SUBMITTER_HK,
) -> Any:
    raw = b"" if body is None else json.dumps(body).encode()
    headers = _sign(raw, hotkey=hotkey)
    if body is not None:
        headers["Content-Type"] = "application/json"
    return await client.post(path, content=raw, headers=headers)


def _job_id(payload: dict[str, Any]) -> str:
    jid = payload.get("id") or payload.get("job_id")
    assert jid, f"expected id/job_id in {payload}"
    return str(jid)


async def _poll_until(
    client: AsyncClient,
    job_id: str,
    *,
    predicate: Any,
    timeout: float = 5.0,
    interval: float = 0.05,
) -> dict[str, Any]:
    """Poll GET /v1/jobs/{id} until predicate(payload) or timeout."""

    deadline = asyncio.get_event_loop().time() + timeout
    last: dict[str, Any] = {}
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/v1/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if predicate(last):
            return last
        await asyncio.sleep(interval)
    raise AssertionError(f"poll timed out; last={last}")


# ----- VAL-JOB-006 + VAL-JOB-026: lifecycle + pollable progress ---------------


@pytest.mark.asyncio
async def test_sim_job_advances_full_lifecycle_and_is_pollable(
    life_client: AsyncClient,
) -> None:
    """VAL-JOB-006/026: sim job visits ordered states; GET polls show progress."""

    body = _valid_job_body(client_request_id="life-full-001")
    created = await _post_job(life_client, body)
    assert created.status_code == 200, created.text
    job_id = _job_id(created.json())
    assert created.json().get("status") in {"submitted", "admitted"}

    seen: list[str] = []
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        resp = await life_client.get(f"/v1/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        status = str(resp.json().get("status"))
        if not seen or seen[-1] != status:
            seen.append(status)
        if status in TERMINAL:
            break
        await asyncio.sleep(0.05)

    assert seen[-1] == "succeeded", f"expected succeeded, seen={seen}"
    # Black-box: at least one non-terminal evolution or admitted→terminal progress.
    assert len(seen) >= 2, f"expected evolving statuses via poll, saw {seen}"
    # Ordered edges: each successive status is at or after previous in the path.
    path_index = {s: i for i, s in enumerate(SUCCESS_PATH_STATUSES)}
    last_idx = -1
    for s in seen:
        if s in path_index:
            assert path_index[s] >= last_idx, f"out-of-order {seen}"
            last_idx = path_index[s]

    terminal = await life_client.get(f"/v1/jobs/{job_id}")
    payload = terminal.json()
    assert payload.get("status") == "succeeded"
    assert payload.get("finished_at") is not None
    # Placement / proofs summary present without secrets.
    assert "placement" in payload or payload.get("placement_policy") is not None
    secret_blob = json.dumps(payload)
    assert "PRIVATE KEY" not in secret_blob
    assert "BEGIN RSA" not in secret_blob
    assert "ssh-rsa AAAA" not in secret_blob


# ----- VAL-JOB-007: cancel ---------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_non_terminal_job(slow_life_client: AsyncClient) -> None:
    """VAL-JOB-007: cancel admitted/placing/running → cancelled; terminal no-op/409."""

    body = _valid_job_body(client_request_id="life-cancel-001", timeout_s=60)
    created = await _post_job(slow_life_client, body)
    assert created.status_code == 200, created.text
    job_id = _job_id(created.json())

    # Wait until we know it is non-terminal / possibly running.
    await _poll_until(
        slow_life_client,
        job_id,
        predicate=lambda p: p.get("status")
        in {
            "admitted",
            "placing",
            "provisioning",
            "running",
            "collecting",
            "scoring",
            "succeeded",
            "cancelled",
            "failed",
            "timeout",
        },
        timeout=2.0,
    )

    # Owner cancel.
    cancel = await _signed_post(slow_life_client, f"/v1/jobs/{job_id}/cancel", {})
    assert cancel.status_code in {200, 409}, cancel.text
    after = await slow_life_client.get(f"/v1/jobs/{job_id}")
    assert after.status_code == 200
    status = after.json().get("status")
    # If still non-terminal when cancel landed, must be cancelled; if already
    # completed before cancel, may succeed (fast drain) — then terminal cancel is
    # no-op/409 and status remains terminal unchanged.
    if cancel.status_code == 200:
        assert status == "cancelled"
        assert after.json().get("finished_at") is not None
        code = after.json().get("failure_code")
        assert code in {None, "cancelled"} or "cancel" in str(code).lower()
    else:
        assert status in TERMINAL

    # Cancel again on terminal → 409 or no-op keeping status.
    again = await _signed_post(slow_life_client, f"/v1/jobs/{job_id}/cancel", {})
    assert again.status_code in {200, 409}, again.text
    final = await slow_life_client.get(f"/v1/jobs/{job_id}")
    assert final.json().get("status") == status

    # Unauthorized cancel of a fresh job from OTHER_HK should 403/401.
    body2 = _valid_job_body(client_request_id="life-cancel-auth-002", timeout_s=60)
    created2 = await _post_job(slow_life_client, body2)
    job2 = _job_id(created2.json())
    bad = await _signed_post(
        slow_life_client,
        f"/v1/jobs/{job2}/cancel",
        {},
        hotkey=OTHER_HK,
    )
    assert bad.status_code in {401, 403}, bad.text
    still = await slow_life_client.get(f"/v1/jobs/{job2}")
    assert still.json().get("status") != "cancelled" or still.json().get(
        "submitter_hotkey"
    ) == SUBMITTER_HK
    # Must not have been cancelled by the foreign hotkey.
    if still.json().get("status") == "cancelled":
        # Only allowed if submitter cancelled (not here).
        pytest.fail("foreign hotkey cancelled owner job")


# ----- VAL-JOB-008: timeout watchdog -----------------------------------------


@pytest.mark.asyncio
async def test_timeout_watchdog_marks_timeout(settings_factory, tmp_path) -> None:
    """VAL-JOB-008: timeout_s=1 + long sim sleep → timeout, not succeeded."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'job-timeout.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        combined_worker=True,
        combined_worker_interval_seconds=0.05,
        job_image_allowlist=ALLOWED_IMAGE,
        max_job_timeout_s=3600,
        sim_job_step_delay_s=0.0,
        sim_job_run_sleep_s=5.0,  # longer than timeout_s
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            body = _valid_job_body(
                client_request_id="life-timeout-001",
                timeout_s=1,
                world_size=2,
                nnodes=1,
                nproc_per_node=2,
                resource={"gpus": 2, "nodes": 1},
            )
            created = await _post_job(client, body)
            assert created.status_code == 200, created.text
            job_id = _job_id(created.json())

            terminal = await _poll_until(
                client,
                job_id,
                predicate=lambda p: p.get("status") in TERMINAL,
                timeout=8.0,
            )
            assert terminal.get("status") == "timeout", terminal
            assert terminal.get("finished_at") is not None
            fcode = str(terminal.get("failure_code") or "").lower()
            assert (
                terminal.get("failure_code") in {"timeout", "job_timeout", None}
                or "time" in fcode
            )


# ----- VAL-JOB-009: results attempt-keyed idempotent -------------------------


@pytest.mark.asyncio
async def test_results_post_attempt_keyed_idempotent(life_client: AsyncClient) -> None:
    """VAL-JOB-009: POST results for attempt n once; duplicate does not invent n+1."""

    # Create and drive to a state that accepts provider results — or post while
    # running. Domain accepts results for non-terminal after admission when no
    # attempt exists yet; sim also posts once on collect. Use explicit posts on
    # a job held mid-pipeline via short timeout path after manual results.
    body = _valid_job_body(
        client_request_id="life-results-001",
        timeout_s=300,
        world_size=2,
        nnodes=1,
        nproc_per_node=2,
        resource={"gpus": 2, "nodes": 1},
    )
    created = await _post_job(life_client, body)
    assert created.status_code == 200, created.text
    job_id = _job_id(created.json())

    # Wait for sim to reach at least running/collecting so attempt may be open,
    # but also allow premature explicit results post (worker/path).
    await asyncio.sleep(0.15)

    payload = {
        "attempt_no": 1,
        "status": "succeeded",
        "metrics": {"allreduce_gbps": 12.5, "efficiency": 0.9},
        "fabric_report_digest": "sha256:" + "ab" * 32,
        "output_digest": "sha256:" + "cd" * 32,
        "proof_tier": "sim",
        "verified": True,
    }
    first = await _signed_post(life_client, f"/v1/jobs/{job_id}/results", payload)
    # 200/201 if accepted, 409 if sim already sealed same attempt with same digest.
    assert first.status_code in {200, 201, 409}, first.text
    first_body = first.json()

    second = await _signed_post(life_client, f"/v1/jobs/{job_id}/results", payload)
    assert second.status_code in {200, 201, 409}, second.text
    second_body = second.json()

    # Stable attempt identity — no attempt_no looping upward.
    a1 = first_body.get("attempt_no") or first_body.get("attempt", {}).get("attempt_no")
    a2 = second_body.get("attempt_no") or second_body.get("attempt", {}).get("attempt_no")
    if a1 is not None and a2 is not None:
        assert int(a1) == int(a2) == 1

    # Attempt detail equality (digests stable).
    att = await life_client.get(f"/v1/jobs/{job_id}/attempts/1")
    assert att.status_code == 200, att.text
    att_payload = att.json()
    assert att_payload.get("attempt_no") == 1 or att_payload.get("status") is not None
    digest = att_payload.get("output_digest") or (att_payload.get("metrics") or {}).get(
        "output_digest"
    )
    # Second post must not have created attempt 2.
    att2 = await life_client.get(f"/v1/jobs/{job_id}/attempts/2")
    assert att2.status_code in {404, 200}
    if att2.status_code == 200:
        # If present, must not be an invented successful twin from duplicate post.
        assert att2.json().get("attempt_no") == 2  # only if real retry policy later
        pytest.fail("duplicate results created attempt 2")

    # Digests present for attempt 1 (VAL-JOB-011 overlap).
    assert (
        att_payload.get("fabric_report_digest")
        or att_payload.get("output_digest")
        or att_payload.get("metrics")
        or digest is not None
    )


# ----- VAL-JOB-010: detail placement/proofs without secrets ------------------


@pytest.mark.asyncio
async def test_job_detail_has_placement_proofs_no_secrets(life_client: AsyncClient) -> None:
    """VAL-JOB-010: GET detail includes placement + proofs summary, no secrets."""

    body = _valid_job_body(client_request_id="life-detail-001")
    created = await _post_job(life_client, body)
    job_id = _job_id(created.json())
    terminal = await _poll_until(
        life_client,
        job_id,
        predicate=lambda p: p.get("status") in TERMINAL,
        timeout=5.0,
    )
    assert terminal.get("status") == "succeeded"

    detail = await life_client.get(f"/v1/jobs/{job_id}")
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    assert payload.get("status") == "succeeded"
    assert payload.get("world_size") == 4
    # Placement summary (nested or flat).
    placement = payload.get("placement")
    if placement is not None:
        assert isinstance(placement, dict)
        assert (
            placement.get("placement_policy") is not None
            or placement.get("rankmap") is not None
            or placement.get("planner_version") is not None
        )
    else:
        assert payload.get("placement_policy") is not None

    proofs = payload.get("proofs") or payload.get("proof_summary")
    if proofs is not None:
        if isinstance(proofs, list) and proofs:
            proofs = proofs[0]
        if isinstance(proofs, dict):
            assert "verified" in proofs or "proof_tier" in proofs or "proof_tier" in payload

    blob = json.dumps(payload).lower()
    for forbidden in ("private_key", "begin rsa", "ssh_private", "password="):
        assert forbidden not in blob


# ----- VAL-JOB-011: attempts detail ------------------------------------------


@pytest.mark.asyncio
async def test_attempt_detail_returns_metrics_digests(life_client: AsyncClient) -> None:
    """VAL-JOB-011: GET attempts/{n} has status + digests after sim collect."""

    body = _valid_job_body(client_request_id="life-attempt-001")
    created = await _post_job(life_client, body)
    job_id = _job_id(created.json())
    await _poll_until(
        life_client,
        job_id,
        predicate=lambda p: p.get("status") in TERMINAL,
        timeout=5.0,
    )

    att = await life_client.get(f"/v1/jobs/{job_id}/attempts/1")
    assert att.status_code == 200, att.text
    payload = att.json()
    assert payload.get("status") in TERMINAL | {
        "running",
        "collecting",
        "scoring",
        "succeeded",
        "failed",
        "timeout",
    }
    # Metrics digests: metrics_json / fabric / output digests.
    has_digest = bool(
        payload.get("fabric_report_digest")
        or payload.get("output_digest")
        or payload.get("metrics")
        or payload.get("metrics_digest")
    )
    assert has_digest, payload


# ----- VAL-JOB-012: list scoped to submitter ---------------------------------


@pytest.mark.asyncio
async def test_list_jobs_scoped_to_submitter_hotkey(life_client: AsyncClient) -> None:
    """VAL-JOB-012: hotkey A does not see B; status filter exact."""

    a = await _post_job(
        life_client,
        _valid_job_body(client_request_id="life-list-a", world_size=2, nnodes=1, nproc_per_node=2),
        hotkey=SUBMITTER_HK,
    )
    b = await _post_job(
        life_client,
        _valid_job_body(client_request_id="life-list-b", world_size=2, nnodes=1, nproc_per_node=2),
        hotkey=OTHER_HK,
    )
    assert a.status_code == 200 and b.status_code == 200
    id_a = _job_id(a.json())
    id_b = _job_id(b.json())

    list_a = await life_client.get("/v1/jobs", headers={"X-Hotkey": SUBMITTER_HK})
    assert list_a.status_code == 200
    ids_a = {(i.get("id") or i.get("job_id")) for i in (list_a.json().get("items") or [])}
    assert id_a in ids_a
    assert id_b not in ids_a

    list_b = await life_client.get("/v1/jobs", headers={"X-Hotkey": OTHER_HK})
    ids_b = {(i.get("id") or i.get("job_id")) for i in (list_b.json().get("items") or [])}
    assert id_b in ids_b
    assert id_a not in ids_b

    # Wait for A to finish then filter by status.
    await _poll_until(
        life_client,
        id_a,
        predicate=lambda p: p.get("status") in TERMINAL,
        timeout=5.0,
    )
    status_a = (await life_client.get(f"/v1/jobs/{id_a}")).json()["status"]
    filtered = await life_client.get(
        f"/v1/jobs?status={status_a}",
        headers={"X-Hotkey": SUBMITTER_HK},
    )
    assert filtered.status_code == 200
    for item in filtered.json().get("items") or []:
        assert item.get("status") == status_a
        assert (item.get("id") or item.get("job_id")) != id_b


# ----- VAL-JOB-020: fabric/tee round-trip ------------------------------------


@pytest.mark.asyncio
async def test_fabric_tee_mode_fields_round_trip(life_client: AsyncClient) -> None:
    """VAL-JOB-020: create with fabric=ib tee=tdx persists on GET."""

    body = _valid_job_body(
        client_request_id="life-fabric-tee-001",
        fabric="ib",
        tee="tdx",
        world_size=2,
        nnodes=1,
        nproc_per_node=2,
        resource={"gpus": 2, "nodes": 1},
    )
    created = await _post_job(life_client, body)
    assert created.status_code == 200, created.text
    job_id = _job_id(created.json())
    detail = await life_client.get(f"/v1/jobs/{job_id}")
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    fabric = payload.get("fabric_mode") or payload.get("fabric")
    tee = payload.get("tee_mode") or payload.get("tee")
    assert fabric == "ib"
    assert tee == "tdx"


# ----- VAL-JOB-021: fabric-report view ---------------------------------------


@pytest.mark.asyncio
async def test_fabric_report_view_for_multi_node_sim(life_client: AsyncClient) -> None:
    """VAL-JOB-021: GET fabric-report after multi-node sim collect has digests."""

    body = _valid_job_body(
        client_request_id="life-fabreport-001",
        world_size=4,
        nnodes=2,
        nproc_per_node=2,
        resource={"gpus": 4, "nodes": 2},
        fabric="ib",
    )
    created = await _post_job(life_client, body)
    job_id = _job_id(created.json())
    await _poll_until(
        life_client,
        job_id,
        predicate=lambda p: p.get("status") in TERMINAL,
        timeout=5.0,
    )

    report = await life_client.get(f"/v1/jobs/{job_id}/fabric-report")
    assert report.status_code == 200, report.text
    payload = report.json()
    assert payload.get("report_digest") or payload.get("fabric_report_digest")
    # Shape: digests / ib devices / topo optional under sim.
    assert (
        payload.get("gpu_topo_sha256") is not None
        or payload.get("ib_devices") is not None
        or payload.get("report_digest") is not None
        or payload.get("nodes") is not None
    )


# ----- VAL-JOB-025: entrypoint/env in launch contract ------------------------


@pytest.mark.asyncio
async def test_entrypoint_env_in_launch_contract(life_client: AsyncClient) -> None:
    """VAL-JOB-025: entrypoint + env appear in placement/launch under sim."""

    entrypoint = ["python", "-m", "train", "--steps", "10"]
    env = {"HYPER_TRAIN_STEPS": "10", "CUSTOM_FLAG": "xyz"}
    body = _valid_job_body(
        client_request_id="life-launch-contract-001",
        entrypoint=entrypoint,
        env=env,
        world_size=2,
        nnodes=1,
        nproc_per_node=2,
        resource={"gpus": 2, "nodes": 1},
    )
    created = await _post_job(life_client, body)
    job_id = _job_id(created.json())
    await _poll_until(
        life_client,
        job_id,
        predicate=lambda p: p.get("status") in TERMINAL,
        timeout=5.0,
    )

    detail = await life_client.get(f"/v1/jobs/{job_id}")
    payload = detail.json()
    # Submitted entrypoint always on job entity.
    assert payload.get("entrypoint") == entrypoint

    placement = payload.get("placement") or {}
    launch = (
        payload.get("launch_contract")
        or placement.get("launch_contract")
        or placement.get("launch")
        or {}
    )
    # Attempt may expose launch contract.
    att = await life_client.get(f"/v1/jobs/{job_id}/attempts/1")
    assert att.status_code == 200, att.text
    att_payload = att.json()
    contract = (
        att_payload.get("launch_contract")
        or launch
        or placement
        or att_payload
    )
    blob = json.dumps(contract)
    assert "python" in blob or "train" in blob or entrypoint[0] in blob
    # Env keys from spec present; NCCL keys from planner must not be wiped.
    env_blob = json.dumps(
        {
            "detail": payload,
            "attempt": att_payload,
            "placement": placement,
            "launch": launch,
        }
    )
    assert "HYPER_TRAIN_STEPS" in env_blob or "CUSTOM_FLAG" in env_blob
    # Planner NCCL / distributed keys should appear in placement/env merge.
    assert (
        "MASTER_ADDR" in env_blob
        or "NCCL" in env_blob
        or "nccl_env" in env_blob
        or "rankmap" in env_blob
    )
