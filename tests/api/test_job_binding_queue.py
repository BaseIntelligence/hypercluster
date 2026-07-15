"""VAL-JOB-013..019, 022..024: capacity binding, CAS claim, queue scaling, durability."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from hypercluster.api.auth import build_signed_headers

SUBMITTER_HK = "job-bind-submitter-aaaaaaaaaaaaaaaaaaaaaaaaa"
OTHER_HK = "job-bind-other-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
PROVIDER_HK = "job-bind-provider-ccccccccccccccccccccccccccc"
TOKEN = "test-challenge-shared-token"

ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"
TERMINAL = frozenset({"succeeded", "failed", "timeout", "cancelled"})


def _sign(body: bytes, *, hotkey: str = SUBMITTER_HK, nonce: str | None = None) -> dict[str, str]:
    return build_signed_headers(secret=TOKEN, hotkey=hotkey, body=body, nonce=nonce)


def _valid_job_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "image_digest": ALLOWED_IMAGE,
        "entrypoint": ["python", "-m", "train", "--steps", "5"],
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
    body.update(overrides)
    return body


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
    timeout: float = 6.0,
    interval: float = 0.05,
) -> dict[str, Any]:
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


async def _register_capacity(
    client: AsyncClient,
    *,
    provider: str = PROVIDER_HK,
    renter: str = SUBMITTER_HK,
    gpu_count: int = 8,
) -> tuple[str, str]:
    """Register provider/node/offer and rent → (lease_id, pod_id)."""

    raw = json.dumps({"display_name": "bind-provider"}).encode()
    headers = {"Content-Type": "application/json", **_sign(raw, hotkey=provider)}
    reg = await client.post("/v1/providers/register", content=raw, headers=headers)
    assert reg.status_code == 200, reg.text

    nbody = {
        "gpu_model": "H100",
        "gpu_count": gpu_count,
        "ssh_endpoint": "10.0.0.9:22",
        "inventory": {"has_ib": True, "ib_rate_gbps": 200},
    }
    nraw = json.dumps(nbody).encode()
    nheaders = {"Content-Type": "application/json", **_sign(nraw, hotkey=provider)}
    node = await client.post("/v1/nodes", content=nraw, headers=nheaders)
    assert node.status_code == 200, node.text
    node_id = node.json()["id"]

    obody = {
        "node_ids": [node_id],
        "price_per_hour": 1.5,
        "max_lifetime_hours": 12.0,
        "mode": "single",
    }
    oraw = json.dumps(obody).encode()
    oheaders = {"Content-Type": "application/json", **_sign(oraw, hotkey=provider)}
    offer = await client.post("/v1/offers", content=oraw, headers=oheaders)
    assert offer.status_code == 200, offer.text

    rraw = json.dumps({"lifetime_hours": 2.0}).encode()
    rheaders = {"Content-Type": "application/json", **_sign(rraw, hotkey=renter)}
    rented = await client.post(
        f"/v1/offers/{offer.json()['id']}/rent",
        content=rraw,
        headers=rheaders,
    )
    assert rented.status_code == 200, rented.text
    payload = rented.json()
    lease = payload.get("lease") or payload
    pod = payload.get("pod") or {}
    lease_id = lease.get("id") or payload.get("lease_id")
    pod_id = pod.get("id") or payload.get("pod_id")
    assert lease_id and pod_id, payload
    return str(lease_id), str(pod_id)


@pytest.fixture
async def bind_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    """Combined worker + marketplace capacity required (no free auto-sim bind)."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'job-bind.sqlite3'}",
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
        max_job_gpu_budget=64,
        sim_job_step_delay_s=0.0,
        sim_job_run_sleep_s=0.0,
        sim_auto_capacity=False,
        max_concurrent_large_jobs=2,
        large_job_world_size_threshold=4,
        max_concurrent_world_size_budget=16,
        sim_launch_fail=False,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


@pytest.fixture
async def auto_client(settings_factory, tmp_path) -> AsyncIterator[AsyncClient]:
    """Combined worker + sim auto capacity (default lifecycle path)."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'job-auto.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        combined_worker=True,
        combined_worker_interval_seconds=0.05,
        job_image_allowlist=ALLOWED_IMAGE,
        max_job_timeout_s=3600,
        max_job_gpu_budget=64,
        sim_auto_capacity=True,
        max_concurrent_large_jobs=2,
        large_job_world_size_threshold=4,
        max_concurrent_world_size_budget=8,
        sim_launch_fail=False,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


# ----- VAL-JOB-013: no capacity → placing then fail/timeout ---------------------


@pytest.mark.asyncio
async def test_without_capacity_stays_placing_then_fails(
    bind_client: AsyncClient,
) -> None:
    """VAL-JOB-013: no healthy offers/nodes + no auto capacity → not succeeded."""

    body = _valid_job_body(
        client_request_id="bind-nocap-001",
        timeout_s=1,
        world_size=2,
        nnodes=1,
        nproc_per_node=2,
        resource={"gpus": 2, "nodes": 1},
    )
    created = await _post_job(bind_client, body)
    assert created.status_code == 200, created.text
    job_id = _job_id(created.json())

    # May remain placing for a window.
    mid = await bind_client.get(f"/v1/jobs/{job_id}")
    early = mid.json().get("status")
    assert early in {
        "admitted",
        "placing",
        "provisioning",
        "failed",
        "timeout",
    }, early
    assert early != "succeeded"

    terminal = await _poll_until(
        bind_client,
        job_id,
        predicate=lambda p: p.get("status") in TERMINAL,
        timeout=8.0,
    )
    assert terminal.get("status") in {"failed", "timeout"}, terminal
    assert terminal.get("status") != "succeeded"
    code = str(terminal.get("failure_code") or "").lower()
    assert (
        any(
            token in code
            for token in (
                "capacity",
                "no_capacity",
                "unavailable",
                "timeout",
            )
        )
        or terminal.get("status") == "timeout"
    ), terminal
    assert terminal.get("lease_id") in {None, ""} or terminal.get("pod_id") is None


# ----- VAL-JOB-014: bind active lease/pod --------------------------------------


@pytest.mark.asyncio
async def test_job_binds_active_lease_pod(bind_client: AsyncClient) -> None:
    """VAL-JOB-014: job with lease_id/pod_id from rent reaches running/succeeded."""

    lease_id, pod_id = await _register_capacity(bind_client)
    body = _valid_job_body(
        client_request_id="bind-lease-001",
        lease_id=lease_id,
        pod_id=pod_id,
        timeout_s=60,
    )
    created = await _post_job(bind_client, body)
    assert created.status_code == 200, created.text
    job_id = _job_id(created.json())

    terminal = await _poll_until(
        bind_client,
        job_id,
        predicate=lambda p: p.get("status") in TERMINAL,
        timeout=6.0,
    )
    assert terminal.get("status") == "succeeded", terminal
    assert terminal.get("lease_id") == lease_id
    assert terminal.get("pod_id") == pod_id

    # Foreign lease must not bind for another renter's job — submitter != renter fails.
    lease2, pod2 = await _register_capacity(
        bind_client,
        renter=OTHER_HK,
        provider=PROVIDER_HK + "x",
    )
    bad = _valid_job_body(
        client_request_id="bind-lease-foreign-002",
        lease_id=lease2,
        pod_id=pod2,
        timeout_s=2,
    )
    created2 = await _post_job(bind_client, bad, hotkey=SUBMITTER_HK)
    assert created2.status_code == 200, created2.text
    job2 = _job_id(created2.json())
    term2 = await _poll_until(
        bind_client,
        job2,
        predicate=lambda p: p.get("status") in TERMINAL,
        timeout=6.0,
    )
    assert term2.get("status") in {"failed", "timeout"}
    assert term2.get("status") != "succeeded"


# ----- VAL-JOB-015: concurrent large jobs respect world_size budget ------------


@pytest.mark.asyncio
async def test_concurrent_large_jobs_respect_budget(auto_client: AsyncClient) -> None:
    """VAL-JOB-015: flood large jobs; concurrent running ≤ cap; queue drains."""

    # Threshold=4, max concurrent large=2, world budget=8 → large job world_size=4.
    job_ids: list[str] = []
    for i in range(5):
        body = _valid_job_body(
            client_request_id=f"bind-cap-large-{i}",
            world_size=4,
            nnodes=2,
            nproc_per_node=2,
            resource={"gpus": 4, "nodes": 2},
            timeout_s=120,
        )
        resp = await _post_job(auto_client, body)
        assert resp.status_code == 200, resp.text
        job_ids.append(_job_id(resp.json()))

    # Observe concurrent running count never exceeds cap while advancing.
    max_running_large = 0
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        statuses: list[str] = []
        for jid in job_ids:
            r = await auto_client.get(f"/v1/jobs/{jid}")
            assert r.status_code == 200
            statuses.append(str(r.json().get("status")))
        running = sum(1 for s in statuses if s == "running")
        max_running_large = max(max_running_large, running)
        if all(s in TERMINAL for s in statuses):
            break
        await asyncio.sleep(0.05)

    assert max_running_large <= 2, f"running concurrent large jobs > cap: {max_running_large}"

    # Eventually all drain (no deadlock / corruption).
    for jid in job_ids:
        term = await _poll_until(
            auto_client,
            jid,
            predicate=lambda p: p.get("status") in TERMINAL,
            timeout=8.0,
        )
        assert term.get("status") in TERMINAL


# ----- VAL-JOB-016: CAS atomic claim -------------------------------------------


@pytest.mark.asyncio
async def test_cas_only_one_worker_claims_place_or_launch(tmp_path, settings_factory) -> None:
    """VAL-JOB-016: concurrent workers claim; single placement + one active attempt."""

    from hypercluster.db.database import Database
    from hypercluster.domain.job_lifecycle import (
        claim_and_advance_job,
        get_attempt,
        get_placement,
        list_attempts,
    )
    from hypercluster.domain.jobs import admit_job
    from hypercluster.settings import HyperSettings

    db_path = tmp_path / "job-cas.sqlite3"
    database = Database(f"sqlite+aiosqlite:///{db_path}")
    await database.init()
    hyper = HyperSettings(
        sim_auto_capacity=True,
        max_concurrent_large_jobs=8,
        max_concurrent_world_size_budget=64,
        large_job_world_size_threshold=4,
    )

    async with database.session() as session:
        job, _ = await admit_job(
            session,
            hotkey=SUBMITTER_HK,
            image_digest=ALLOWED_IMAGE,
            entrypoint=["python", "-m", "train"],
            world_size=2,
            nnodes=1,
            nproc_per_node=2,
            resource={"gpus": 2},
            timeout_s=60,
            client_request_id="cas-001",
            image_allowlist=frozenset({ALLOWED_IMAGE}),
        )
        job_id = job.id

    # Two concurrent claim/advance from admitted → placing/provisioning.
    async def worker(worker_id: str) -> str | None:
        async with database.session() as session:
            updated = await claim_and_advance_job(
                session,
                job_id,
                worker_id=worker_id,
                hyper=hyper,
                run_sleep_s=0.0,
            )
            return updated.status if updated is not None else None

    results = await asyncio.gather(worker("w1"), worker("w2"), worker("w3"))
    # At least one should claim; others may no-op if already advanced.
    asserted = [r for r in results if r is not None]
    assert asserted, "expected at least one successful CAS claim"
    # Drive fully to terminal with one series of advances.
    for _ in range(12):
        async with database.session() as session:
            await claim_and_advance_job(
                session,
                job_id,
                worker_id="drain",
                hyper=hyper,
                run_sleep_s=0.0,
            )

    async with database.session() as session:
        placement = await get_placement(session, job_id)
        attempts = await list_attempts(session, job_id)
        active = [a for a in attempts if a.status == "running"]
        assert placement is not None, "exactly one placement expected after place"
        # Unique placement per job (single row).
        assert placement.job_id == job_id
        # At most one active attempt (and not dual concurrent).
        assert len(active) <= 1
        att1 = await get_attempt(session, job_id, 1)
        assert att1 is not None
        # No duplicate inventing pair of concurrent running rows.
        runningish = [a for a in attempts if a.status in {"running", "collecting"}]
        assert len(runningish) <= 1 or all(
            a.attempt_no == 1 for a in attempts
        ), f"unexpected parallel attempts: {[(a.attempt_no, a.status) for a in attempts]}"
        assert len({a.attempt_no for a in attempts}) == len(attempts)

    await database.close()


# ----- VAL-JOB-017: combined worker drains place/launch/score ------------------


@pytest.mark.asyncio
async def test_combined_worker_drains_end_to_end(auto_client: AsyncClient) -> None:
    """VAL-JOB-017: single process combined worker reaches terminal without sidecar."""

    health = await auto_client.get("/health")
    assert health.status_code == 200
    assert health.json().get("ready") is True

    body = _valid_job_body(client_request_id="bind-combined-001")
    created = await _post_job(auto_client, body)
    assert created.status_code == 200, created.text
    job_id = _job_id(created.json())
    terminal = await _poll_until(
        auto_client,
        job_id,
        predicate=lambda p: p.get("status") in TERMINAL,
        timeout=6.0,
    )
    assert terminal.get("status") == "succeeded"
    assert terminal.get("finished_at") is not None


# ----- VAL-JOB-018: failed launch ----------------------------------------------


@pytest.mark.asyncio
async def test_failed_launch_sets_failed_code(settings_factory, tmp_path) -> None:
    """VAL-JOB-018: forced launch fail → failed + failure_code, not succeeded."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'job-launch-fail.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        combined_worker=True,
        combined_worker_interval_seconds=0.05,
        job_image_allowlist=ALLOWED_IMAGE,
        sim_auto_capacity=True,
        sim_launch_fail=True,
        max_job_timeout_s=3600,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            body = _valid_job_body(client_request_id="bind-launch-fail-001", timeout_s=60)
            created = await _post_job(client, body)
            assert created.status_code == 200, created.text
            job_id = _job_id(created.json())
            terminal = await _poll_until(
                client,
                job_id,
                predicate=lambda p: p.get("status") in TERMINAL,
                timeout=6.0,
            )
            assert terminal.get("status") == "failed", terminal
            code = str(terminal.get("failure_code") or "").lower()
            assert "launch" in code or code in {
                "launch_failed",
                "sim_launch_fail",
                "failed",
            }, terminal
            assert terminal.get("status") != "succeeded"


# ----- VAL-JOB-019: teardown frees capacity ------------------------------------


@pytest.mark.asyncio
async def test_teardown_after_terminal_frees_for_relist(
    bind_client: AsyncClient,
) -> None:
    """VAL-JOB-019: after job terminal, lease end frees nodes; no permanent deadlock."""

    lease_id, pod_id = await _register_capacity(bind_client, gpu_count=4)
    body = _valid_job_body(
        client_request_id="bind-teardown-001",
        lease_id=lease_id,
        pod_id=pod_id,
        timeout_s=60,
    )
    created = await _post_job(bind_client, body)
    job_id = _job_id(created.json())
    terminal = await _poll_until(
        bind_client,
        job_id,
        predicate=lambda p: p.get("status") in TERMINAL,
        timeout=6.0,
    )
    assert terminal.get("status") == "succeeded"

    # Terminate lease (end of rental) — capacity free for re-use / re-list.
    term = await _signed_post(bind_client, f"/v1/leases/{lease_id}/terminate", {})
    assert term.status_code == 200, term.text
    lease_body = term.json().get("lease") or term.json()
    assert lease_body.get("status") in {"terminated", "expired"}

    pod = await bind_client.get(f"/v1/pods/{pod_id}")
    assert pod.status_code == 200
    assert pod.json().get("status") in {"stopped", "stopping"}

    # Re-register capacity on free inventory: new node path or re-offer after free.
    lease2, pod2 = await _register_capacity(
        bind_client,
        provider=PROVIDER_HK + "-2",
        renter=SUBMITTER_HK,
        gpu_count=4,
    )
    body2 = _valid_job_body(
        client_request_id="bind-teardown-002",
        lease_id=lease2,
        pod_id=pod2,
    )
    created2 = await _post_job(bind_client, body2)
    assert created2.status_code == 200, created2.text
    job2 = _job_id(created2.json())
    term2 = await _poll_until(
        bind_client,
        job2,
        predicate=lambda p: p.get("status") in TERMINAL,
        timeout=6.0,
    )
    assert term2.get("status") == "succeeded"


# ----- VAL-JOB-022: fair queue — micro not starved under giants ----------------


@pytest.mark.asyncio
async def test_fair_queue_micro_jobs_complete_under_giants(
    auto_client: AsyncClient,
) -> None:
    """VAL-JOB-022: under mixed flood, micro jobs complete within bound."""

    giant_ids: list[str] = []
    micro_ids: list[str] = []
    for i in range(4):
        body = _valid_job_body(
            client_request_id=f"bind-giant-{i}",
            world_size=4,
            nnodes=2,
            nproc_per_node=2,
            resource={"gpus": 4, "nodes": 2},
            timeout_s=120,
        )
        r = await _post_job(auto_client, body)
        assert r.status_code == 200
        giant_ids.append(_job_id(r.json()))

    for i in range(4):
        body = _valid_job_body(
            client_request_id=f"bind-micro-{i}",
            world_size=1,
            nnodes=1,
            nproc_per_node=1,
            resource={"gpus": 1, "nodes": 1},
            timeout_s=120,
        )
        r = await _post_job(auto_client, body)
        assert r.status_code == 200
        micro_ids.append(_job_id(r.json()))

    # All micros complete in reasonable time (fairness bound).
    for jid in micro_ids:
        term = await _poll_until(
            auto_client,
            jid,
            predicate=lambda p: p.get("status") in TERMINAL,
            timeout=8.0,
        )
        assert term.get("status") in TERMINAL, f"micro {jid} not terminal: {term}"

    # Giants also eventually finish (system not deadlocked).
    for jid in giant_ids:
        term = await _poll_until(
            auto_client,
            jid,
            predicate=lambda p: p.get("status") in TERMINAL,
            timeout=10.0,
        )
        assert term.get("status") in TERMINAL


# ----- VAL-JOB-023: health remains 200 under modest storm ----------------------


@pytest.mark.asyncio
async def test_health_200_during_job_storm(auto_client: AsyncClient) -> None:
    """VAL-JOB-023: GET /health and /ready stay 200 during concurrent sim jobs."""

    job_ids: list[str] = []
    for i in range(6):
        body = _valid_job_body(
            client_request_id=f"bind-storm-{i}",
            world_size=2 if i % 2 == 0 else 1,
            nnodes=1,
            nproc_per_node=2 if i % 2 == 0 else 1,
            resource={"gpus": 2 if i % 2 == 0 else 1, "nodes": 1},
        )
        r = await _post_job(auto_client, body)
        assert r.status_code == 200
        job_ids.append(_job_id(r.json()))

    health_codes: list[int] = []
    ready_codes: list[int] = []
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        h = await auto_client.get("/health")
        r = await auto_client.get("/ready")
        health_codes.append(h.status_code)
        ready_codes.append(r.status_code)
        statuses = []
        for jid in job_ids:
            jr = await auto_client.get(f"/v1/jobs/{jid}")
            statuses.append(jr.json().get("status"))
        if all(s in TERMINAL for s in statuses):
            break
        await asyncio.sleep(0.05)

    assert health_codes, "expected health polls"
    assert all(c == 200 for c in health_codes), health_codes
    assert all(c == 200 for c in ready_codes), ready_codes


# ----- VAL-JOB-024: SQLite durability across restart ---------------------------


@pytest.mark.asyncio
async def test_sqlite_persistence_survives_restart(
    settings_factory,
    tmp_path,
) -> None:
    """VAL-JOB-024: create job, stop app, restart same DB → GET same id/status."""

    from hypercluster.app import create_app
    from hypercluster.settings import HyperSettings

    db_url = f"sqlite+aiosqlite:///{tmp_path / 'job-durable.sqlite3'}"
    settings = settings_factory(
        database_url=db_url,
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        combined_worker=True,
        combined_worker_interval_seconds=0.05,
        job_image_allowlist=ALLOWED_IMAGE,
        sim_auto_capacity=True,
        max_job_timeout_s=3600,
    )

    job_id: str
    status_before: str

    app1 = create_app(settings, hyper_settings=hyper)
    async with app1.router.lifespan_context(app1):
        transport = ASGITransport(app=app1)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            body = _valid_job_body(client_request_id="bind-durable-001")
            created = await _post_job(client, body)
            assert created.status_code == 200, created.text
            job_id = _job_id(created.json())
            # Wait until at least admitted (may advance further under combined worker).
            mid = await _poll_until(
                client,
                job_id,
                predicate=lambda p: p.get("status") is not None,
                timeout=2.0,
            )
            status_before = str(mid.get("status"))
            # Capture a terminal if quick; otherwise pre-terminal is fine.
            if status_before not in TERMINAL:
                try:
                    mid = await _poll_until(
                        client,
                        job_id,
                        predicate=lambda p: p.get("status") in TERMINAL,
                        timeout=4.0,
                    )
                    status_before = str(mid.get("status"))
                except AssertionError:
                    # Still non-terminal is ok for durability of the row itself.
                    mid = (await client.get(f"/v1/jobs/{job_id}")).json()
                    status_before = str(mid.get("status"))

    # Process "restart" — new app against same sqlite file.
    app2 = create_app(settings, hyper_settings=hyper)
    async with app2.router.lifespan_context(app2):
        transport = ASGITransport(app=app2)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            after = await client.get(f"/v1/jobs/{job_id}")
            assert after.status_code == 200, after.text
            payload = after.json()
            assert (payload.get("id") or payload.get("job_id")) == job_id
            # Row survived; status is either same or advanced (worker can continue).
            assert payload.get("status") is not None
            if status_before in TERMINAL:
                assert payload.get("status") == status_before
            assert payload.get("submitter_hotkey") == SUBMITTER_HK
            assert payload.get("client_request_id") == "bind-durable-001"
