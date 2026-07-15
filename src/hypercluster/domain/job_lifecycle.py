"""Job lifecycle state machine, cancel, timeout, results, and sim advancedrain.

Fulfills VAL-JOB-006..012, 020, 021, 025, 026 for M3 job lifecycle slice.

State machine (architecture §6.2)::

    admitted → placing → provisioning → running → collecting → scoring → succeeded
         ↘ cancelled (any non-terminal)  ↘ timeout  ↘ failed

Sim path under HYPER_COMBINED_WORKER advances one step per tick with optional
run sleep; timeout watches started_at + timeout_s.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import (
    Job,
    JobAttempt,
    JobFabricReport,
    JobPlacement,
    JobProof,
    isoformat_utc,
    utc_now,
)
from hypercluster.domain.jobs import JobError, get_job

logger = logging.getLogger(__name__)

# Status constants
JOB_STATUS_ADMITTED = "admitted"
JOB_STATUS_PLACING = "placing"
JOB_STATUS_PROVISIONING = "provisioning"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COLLECTING = "collecting"
JOB_STATUS_SCORING = "scoring"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"
JOB_STATUS_TIMEOUT = "timeout"

SUCCESS_PATH: tuple[str, ...] = (
    JOB_STATUS_ADMITTED,
    JOB_STATUS_PLACING,
    JOB_STATUS_PROVISIONING,
    JOB_STATUS_RUNNING,
    JOB_STATUS_COLLECTING,
    JOB_STATUS_SCORING,
    JOB_STATUS_SUCCEEDED,
)

TERMINAL_STATUSES = frozenset(
    {
        JOB_STATUS_SUCCEEDED,
        JOB_STATUS_FAILED,
        JOB_STATUS_CANCELLED,
        JOB_STATUS_TIMEOUT,
    }
)

NON_TERMINAL_STATUSES = frozenset(
    {
        "submitted",
        JOB_STATUS_ADMITTED,
        JOB_STATUS_PLACING,
        JOB_STATUS_PROVISIONING,
        JOB_STATUS_RUNNING,
        JOB_STATUS_COLLECTING,
        JOB_STATUS_SCORING,
    }
)

# Allowed transitions (forward only + cancel/timeout/fail from non-terminal).
_FORWARD: dict[str, str] = {
    "submitted": JOB_STATUS_ADMITTED,
    JOB_STATUS_ADMITTED: JOB_STATUS_PLACING,
    JOB_STATUS_PLACING: JOB_STATUS_PROVISIONING,
    JOB_STATUS_PROVISIONING: JOB_STATUS_RUNNING,
    JOB_STATUS_RUNNING: JOB_STATUS_COLLECTING,
    JOB_STATUS_COLLECTING: JOB_STATUS_SCORING,
    JOB_STATUS_SCORING: JOB_STATUS_SUCCEEDED,
}

PLANNER_VERSION = "fabric-planner.v1"
SIM_PROOF_TIER = "sim"


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def _sha256_hex(payload: bytes | str) -> str:
    data = payload if isinstance(payload, bytes) else payload.encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def build_sim_rankmap(
    *,
    job_id: str,
    nnodes: int,
    nproc_per_node: int,
    world_size: int,
) -> list[dict[str, Any]]:
    """Synthetic rankmap for local sim (node_id = sim-node-{i})."""

    rankmap: list[dict[str, Any]] = []
    rank = 0
    for node_idx in range(nnodes):
        node_id = f"sim-node-{node_idx}"
        for local_rank in range(nproc_per_node):
            if rank >= world_size:
                break
            rankmap.append(
                {
                    "rank": rank,
                    "node_id": node_id,
                    "local_rank": local_rank,
                    "gpu_index": local_rank,
                    "job_id": job_id,
                }
            )
            rank += 1
    return rankmap


def build_sim_nccl_env(
    *,
    fabric_mode: str,
    backend: str = "nccl",
) -> dict[str, str]:
    """NCCL env matrix stub (planner contract v1)."""

    env: dict[str, str] = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_SOCKET_IFNAME": "lo",
        "HYPER_BACKEND": backend,
        "HYPER_FABRIC_MODE": fabric_mode,
    }
    if fabric_mode == "ib":
        env["NCCL_NET"] = "IB"
        env["NCCL_IB_HCA"] = "mlx5_0"
        env["NCCL_IB_GID_INDEX"] = "3"
    elif fabric_mode == "eth":
        env["NCCL_NET"] = "Socket"
    return env


def build_launch_contract(
    *,
    job: Job,
    rankmap: list[dict[str, Any]],
    nccl_env: dict[str, str],
) -> dict[str, Any]:
    """Merge entrypoint + user env + planner NCCL into launch contract (VAL-JOB-025)."""

    user_env = job.env() or {}
    merged_env = {**nccl_env, **user_env}
    return {
        "image_digest": job.image_digest,
        "entrypoint": job.entrypoint(),
        "env": merged_env,
        "user_env": user_env,
        "nccl_env": nccl_env,
        "rankmap": rankmap,
        "world_size": int(job.world_size),
        "nnodes": int(job.nnodes),
        "nproc_per_node": int(job.nproc_per_node),
        "backend": job.backend,
        "fabric_mode": job.fabric_mode,
        "tee_mode": job.tee_mode,
        "timeout_s": int(job.timeout_s),
        "placement_policy": job.placement_policy,
        "planner_version": PLANNER_VERSION,
    }


def build_sim_fabric_report(
    *,
    job: Job,
    attempt_id: str | None,
) -> dict[str, Any]:
    """FabricReport-shaped payload for multi-node sim (VAL-JOB-021)."""

    ib_devices: list[dict[str, Any]] = []
    if job.fabric_mode in {"auto", "ib"}:
        for i in range(max(1, int(job.nnodes))):
            ib_devices.append(
                {
                    "name": f"mlx5_{i}",
                    "port": 1,
                    "rate_gbps": 200.0,
                    "state": "Active",
                    "node_id": f"sim-node-{i}",
                }
            )
    topo_text = f"sim-topo nnodes={job.nnodes} world={job.world_size} fabric={job.fabric_mode}"
    gpu_topo_sha256 = hashlib.sha256(topo_text.encode()).hexdigest()
    body = {
        "job_id": job.id,
        "attempt_id": attempt_id,
        "nnodes": int(job.nnodes),
        "world_size": int(job.world_size),
        "fabric_mode": job.fabric_mode,
        "ib_devices": ib_devices,
        "ib_rate_gbps": 200.0 if ib_devices else None,
        "gpu_topo_sha256": gpu_topo_sha256,
        "numa_map": {f"gpu{i}": i % 2 for i in range(int(job.world_size))},
        "nccl_version": "sim-2.21.5",
        "eth_ifaces": ["lo", "eth0"],
    }
    report_digest = _sha256_hex(_canonical_json(body))
    body["report_digest"] = report_digest
    return body


async def get_placement(session: AsyncSession, job_id: str) -> JobPlacement | None:
    result = await session.execute(
        select(JobPlacement)
        .where(JobPlacement.job_id == job_id)
        .order_by(JobPlacement.created_at.asc())
    )
    return result.scalars().first()


async def get_attempt(
    session: AsyncSession,
    job_id: str,
    attempt_no: int,
) -> JobAttempt | None:
    result = await session.execute(
        select(JobAttempt).where(
            JobAttempt.job_id == job_id,
            JobAttempt.attempt_no == attempt_no,
        )
    )
    return result.scalar_one_or_none()


async def list_attempts(session: AsyncSession, job_id: str) -> list[JobAttempt]:
    result = await session.execute(
        select(JobAttempt)
        .where(JobAttempt.job_id == job_id)
        .order_by(JobAttempt.attempt_no.asc())
    )
    return list(result.scalars().all())


async def get_latest_attempt(session: AsyncSession, job_id: str) -> JobAttempt | None:
    result = await session.execute(
        select(JobAttempt)
        .where(JobAttempt.job_id == job_id)
        .order_by(JobAttempt.attempt_no.desc())
    )
    return result.scalars().first()


async def get_proofs_for_attempt(session: AsyncSession, attempt_id: str) -> list[JobProof]:
    result = await session.execute(select(JobProof).where(JobProof.attempt_id == attempt_id))
    return list(result.scalars().all())


async def get_fabric_report(session: AsyncSession, job_id: str) -> JobFabricReport | None:
    result = await session.execute(
        select(JobFabricReport).where(JobFabricReport.job_id == job_id)
    )
    return result.scalar_one_or_none()


async def cancel_job(
    session: AsyncSession,
    *,
    job_id: str,
    hotkey: str,
) -> Job:
    """Owner-only cancel of non-terminal job (VAL-JOB-007)."""

    job = await get_job(session, job_id)
    if job is None:
        raise JobError("job_not_found", "job not found", status_code=404)
    if job.submitter_hotkey != hotkey:
        raise JobError(
            "forbidden",
            "only the submitter may cancel this job",
            status_code=403,
        )
    if is_terminal(job.status):
        raise JobError(
            "already_terminal",
            f"job is already terminal with status={job.status}",
            status_code=409,
        )

    now = utc_now()
    job.status = JOB_STATUS_CANCELLED
    job.failure_code = "cancelled"
    job.finished_at = now
    job.updated_at = now

    attempt = await get_latest_attempt(session, job_id)
    if attempt is not None and not is_terminal(attempt.status):
        attempt.status = JOB_STATUS_CANCELLED
        attempt.failure_code = "cancelled"
        attempt.finished_at = now

    await session.commit()
    await session.refresh(job)
    return job


async def post_job_results(
    session: AsyncSession,
    *,
    job_id: str,
    attempt_no: int = 1,
    status: str = "succeeded",
    metrics: dict[str, Any] | None = None,
    fabric_report_digest: str | None = None,
    output_digest: str | None = None,
    proof_tier: str = SIM_PROOF_TIER,
    verified: bool = True,
    verify_mode: str = "sim",
    failure_code: str | None = None,
    actor_hotkey: str | None = None,
) -> tuple[JobAttempt, bool]:
    """Provider/worker results post (VAL-JOB-009 attempt-keyed idempotent).

    Returns ``(attempt, created_or_updated)``. Duplicate posts with equivalent
    payload are idempotent; conflicting digests → 409.
    """

    if attempt_no < 1:
        raise JobError("invalid_attempt_no", "attempt_no must be >= 1", status_code=422)

    job = await get_job(session, job_id)
    if job is None:
        raise JobError("job_not_found", "job not found", status_code=404)

    # Owner or any authenticated worker can post in sim path (actor optional).
    _ = actor_hotkey

    envelope = {
        "attempt_no": attempt_no,
        "status": status,
        "metrics": metrics or {},
        "fabric_report_digest": fabric_report_digest,
        "output_digest": output_digest,
        "proof_tier": proof_tier,
        "verified": bool(verified),
        "failure_code": failure_code,
    }
    result_digest = _sha256_hex(_canonical_json(envelope))

    existing = await get_attempt(session, job_id, attempt_no)
    if existing is not None:
        # Idempotent: same digest → return existing.
        if existing.result_digest == result_digest or (
            existing.output_digest == output_digest
            and existing.fabric_report_digest == fabric_report_digest
            and existing.status == status
        ):
            return existing, False
        # Already sealed with different payload → conflict (no double metrics).
        if existing.result_digest is not None and existing.finished_at is not None:
            raise JobError(
                "result_conflict",
                "attempt already has results with a different digest",
                status_code=409,
            )
        # Unsealed attempt: apply update.
        attempt = existing
        created = False
    else:
        attempt = JobAttempt(
            id=str(uuid.uuid4()),
            job_id=job_id,
            attempt_no=attempt_no,
            status=status,
            started_at=job.started_at or utc_now(),
        )
        session.add(attempt)
        await session.flush()
        created = True

    now = utc_now()
    attempt.status = status
    attempt.metrics_json = json.dumps(metrics or {})
    attempt.fabric_report_digest = fabric_report_digest
    attempt.output_digest = output_digest
    attempt.result_digest = result_digest
    attempt.failure_code = failure_code
    attempt.finished_at = now
    if attempt.started_at is None:
        attempt.started_at = job.started_at or now

    # Attach proof summary (no secrets). Skip if already attached.
    proofs = [] if created else await get_proofs_for_attempt(session, attempt.id)
    if not proofs:
        session.add(
            JobProof(
                id=str(uuid.uuid4()),
                attempt_id=attempt.id,
                proof_tier=proof_tier,
                payload_json=json.dumps({"sim": True, "result_digest": result_digest}),
                verified=1 if verified else 0,
                verify_mode=verify_mode,
            )
        )

    # Persist fabric report view when digest provided.
    if fabric_report_digest:
        report = await get_fabric_report(session, job_id)
        if report is None:
            fab_body = build_sim_fabric_report(job=job, attempt_id=attempt.id)
            # Prefer caller digest when provided.
            fab_body["report_digest"] = fabric_report_digest
            session.add(
                JobFabricReport(
                    id=str(uuid.uuid4()),
                    job_id=job_id,
                    attempt_id=attempt.id,
                    collected_at=now,
                    ib_devices_json=json.dumps(fab_body.get("ib_devices") or []),
                    ib_rate_gbps=fab_body.get("ib_rate_gbps"),
                    gpu_topo_sha256=fab_body.get("gpu_topo_sha256"),
                    numa_map_json=json.dumps(fab_body.get("numa_map") or {}),
                    nccl_version=fab_body.get("nccl_version"),
                    report_digest=fabric_report_digest,
                    raw_json=json.dumps(fab_body),
                )
            )
        elif report.report_digest != fabric_report_digest and report.attempt_id != attempt.id:
            # Keep first report stable for attempt-keyed idempotency.
            pass

    terminal_status = status if status in TERMINAL_STATUSES else None
    if terminal_status and not is_terminal(job.status):
        if terminal_status == JOB_STATUS_SUCCEEDED:
            # Move through scoring if not yet; for explicit results, mark succeeded.
            job.status = JOB_STATUS_SUCCEEDED
            job.finished_at = now
            job.failure_code = None
        elif terminal_status in {
            JOB_STATUS_FAILED,
            JOB_STATUS_TIMEOUT,
            JOB_STATUS_CANCELLED,
        }:
            job.status = terminal_status
            job.finished_at = now
            job.failure_code = failure_code or terminal_status
        job.updated_at = now
    elif job.status in {
        JOB_STATUS_ADMITTED,
        JOB_STATUS_PLACING,
        JOB_STATUS_PROVISIONING,
        JOB_STATUS_RUNNING,
        JOB_STATUS_COLLECTING,
    } and status == JOB_STATUS_SUCCEEDED:
        # Do not skip ranking; leave worker loop to finish if in mid-pipeline
        # but seal metrics on the attempt.
        pass

    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        # Race: another worker created the attempt — re-read idempotently.
        existing2 = await get_attempt(session, job_id, attempt_no)
        if existing2 is not None:
            if existing2.result_digest in {None, result_digest} or (
                existing2.output_digest == output_digest
            ):
                return existing2, False
        raise JobError(
            "result_conflict",
            "concurrent results post conflict",
            status_code=409,
        ) from None

    await session.refresh(attempt)
    return attempt, created


def job_detail_public(
    job: Job,
    *,
    placement: JobPlacement | None = None,
    attempt: JobAttempt | None = None,
    proofs: list[JobProof] | None = None,
    fabric_report: JobFabricReport | None = None,
) -> dict[str, Any]:
    """GET job detail with placement/proofs summary and no secrets (VAL-JOB-010)."""

    body = job.to_dict()
    if placement is not None:
        place = placement.to_dict()
        body["placement"] = place
        launch = placement.launch_contract()
        if launch is not None:
            body["launch_contract"] = launch
        body["placement_policy"] = placement.placement_policy
    if attempt is not None:
        body["latest_attempt"] = {
            "attempt_no": attempt.attempt_no,
            "status": attempt.status,
            "fabric_report_digest": attempt.fabric_report_digest,
            "output_digest": attempt.output_digest,
            "finished_at": isoformat_utc(attempt.finished_at),
        }
        if attempt.launch_contract() is not None and "launch_contract" not in body:
            body["launch_contract"] = attempt.launch_contract()
    if proofs:
        body["proofs"] = [p.to_public_summary() for p in proofs]
        body["proof_summary"] = proofs[0].to_public_summary() if proofs else None
    elif attempt is not None:
        body["proofs"] = []
        body["proof_summary"] = {
            "proof_tier": SIM_PROOF_TIER,
            "verified": attempt.status == JOB_STATUS_SUCCEEDED,
            "verify_mode": "sim",
        }
    if fabric_report is not None:
        body["fabric_report_digest"] = fabric_report.report_digest
    return body


def attempt_to_public(attempt: JobAttempt) -> dict[str, Any]:
    return attempt.to_dict()


# ---------------------------------------------------------------------------
# Sim worker drain (combined worker items)
# ---------------------------------------------------------------------------


async def list_drainable_jobs(session: AsyncSession, *, limit: int = 32) -> list[Job]:
    """Jobs that still need lifecycle advancement (micro-first fair order)."""

    # Fair queue: prefer small world_size so micros are not starved (VAL-JOB-022).
    from hypercluster.domain.job_queue import list_drainable_jobs_fair

    return await list_drainable_jobs_fair(session, limit=limit)


async def _place_job(session: AsyncSession, job: Job) -> None:
    rankmap = build_sim_rankmap(
        job_id=job.id,
        nnodes=int(job.nnodes),
        nproc_per_node=int(job.nproc_per_node),
        world_size=int(job.world_size),
    )
    nccl_env = build_sim_nccl_env(fabric_mode=job.fabric_mode, backend=job.backend)
    launch = build_launch_contract(job=job, rankmap=rankmap, nccl_env=nccl_env)
    graph_digest = _sha256_hex(
        _canonical_json({"rankmap": rankmap, "policy": job.placement_policy})
    )

    existing = await get_placement(session, job.id)
    if existing is None:
        session.add(
            JobPlacement(
                id=str(uuid.uuid4()),
                job_id=job.id,
                rankmap_json=json.dumps(rankmap),
                placement_policy=job.placement_policy,
                nccl_env_json=json.dumps(nccl_env),
                planner_version=PLANNER_VERSION,
                launch_contract_json=json.dumps(launch),
                graph_digest=graph_digest,
            )
        )


async def _ensure_running_attempt(session: AsyncSession, job: Job) -> JobAttempt:
    """Return active attempt; reuse sealed attempt #1 rather than invent n+1.

    Result posts may seal attempt 1 early (provider path). Sim drain must not
    invent attempt 2 solely from that (VAL-JOB-009).
    """

    attempt = await get_latest_attempt(session, job.id)
    if attempt is not None and not is_terminal(attempt.status):
        return attempt
    if attempt is not None and is_terminal(attempt.status):
        # Keep the sealed attempt; try_reuse for collect step identities.
        return attempt
    placement = await get_placement(session, job.id)
    launch_json = placement.launch_contract_json if placement is not None else None
    attempt = JobAttempt(
        id=str(uuid.uuid4()),
        job_id=job.id,
        attempt_no=1,
        status=JOB_STATUS_RUNNING,
        launch_contract_json=launch_json,
        started_at=utc_now(),
    )
    session.add(attempt)
    await session.flush()
    return attempt


async def _collect_success(session: AsyncSession, job: Job, attempt: JobAttempt) -> None:
    # If provider already sealed results, keep digests (idempotent) and only
    # ensure fabric report / proof rows exist.
    if attempt.result_digest is not None and attempt.finished_at is not None:
        report = await get_fabric_report(session, job.id)
        if report is None and attempt.fabric_report_digest:
            fab = build_sim_fabric_report(job=job, attempt_id=attempt.id)
            fab["report_digest"] = attempt.fabric_report_digest
            session.add(
                JobFabricReport(
                    id=str(uuid.uuid4()),
                    job_id=job.id,
                    attempt_id=attempt.id,
                    collected_at=utc_now(),
                    ib_devices_json=json.dumps(fab.get("ib_devices") or []),
                    ib_rate_gbps=fab.get("ib_rate_gbps"),
                    gpu_topo_sha256=fab.get("gpu_topo_sha256"),
                    numa_map_json=json.dumps(fab.get("numa_map") or {}),
                    nccl_version=fab.get("nccl_version"),
                    report_digest=attempt.fabric_report_digest,
                    raw_json=json.dumps(fab),
                )
            )
        proofs = await get_proofs_for_attempt(session, attempt.id)
        if not proofs:
            session.add(
                JobProof(
                    id=str(uuid.uuid4()),
                    attempt_id=attempt.id,
                    proof_tier=SIM_PROOF_TIER,
                    payload_json=json.dumps({"sim": True, "tier": SIM_PROOF_TIER}),
                    verified=1,
                    verify_mode="sim",
                )
            )
        return

    fab = build_sim_fabric_report(job=job, attempt_id=attempt.id)
    metrics = {
        "allreduce_gbps": 18.0 * max(1, int(job.world_size)) / 4.0,
        "efficiency": 0.92,
        "wall_time_s": 0.05,
        "source": "sim_launcher",
    }
    output_digest = _sha256_hex(
        _canonical_json(
            {
                "job_id": job.id,
                "attempt_no": attempt.attempt_no,
                "entrypoint": job.entrypoint(),
                "ok": True,
            }
        )
    )
    result_digest = _sha256_hex(
        _canonical_json(
            {
                "attempt_no": attempt.attempt_no,
                "status": JOB_STATUS_SUCCEEDED,
                "metrics": metrics,
                "fabric_report_digest": fab["report_digest"],
                "output_digest": output_digest,
                "proof_tier": SIM_PROOF_TIER,
                "verified": True,
                "failure_code": None,
            }
        )
    )
    attempt.status = JOB_STATUS_SUCCEEDED
    attempt.metrics_json = json.dumps(metrics)
    attempt.fabric_report_digest = fab["report_digest"]
    attempt.output_digest = output_digest
    attempt.result_digest = result_digest
    attempt.finished_at = utc_now()

    proofs = await get_proofs_for_attempt(session, attempt.id)
    if not proofs:
        session.add(
            JobProof(
                id=str(uuid.uuid4()),
                attempt_id=attempt.id,
                proof_tier=SIM_PROOF_TIER,
                payload_json=json.dumps({"sim": True, "tier": SIM_PROOF_TIER}),
                verified=1,
                verify_mode="sim",
            )
        )

    report = await get_fabric_report(session, job.id)
    if report is None:
        session.add(
            JobFabricReport(
                id=str(uuid.uuid4()),
                job_id=job.id,
                attempt_id=attempt.id,
                collected_at=utc_now(),
                ib_devices_json=json.dumps(fab.get("ib_devices") or []),
                ib_rate_gbps=fab.get("ib_rate_gbps"),
                gpu_topo_sha256=fab.get("gpu_topo_sha256"),
                numa_map_json=json.dumps(fab.get("numa_map") or {}),
                nccl_version=fab.get("nccl_version"),
                report_digest=fab["report_digest"],
                raw_json=json.dumps(fab),
            )
        )


def _as_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


async def should_timeout(job: Job, *, now: datetime | None = None) -> bool:
    """True when started_at + timeout_s elapsed (VAL-JOB-008)."""

    if is_terminal(job.status):
        return False
    started = _as_aware(job.started_at)
    if started is None:
        # Use admitted_at as fall-back once past placing so tiny timeouts still fire.
        if job.status in {
            JOB_STATUS_RUNNING,
            JOB_STATUS_COLLECTING,
            JOB_STATUS_SCORING,
            JOB_STATUS_PROVISIONING,
        }:
            started = _as_aware(job.admitted_at) or _as_aware(job.created_at)
        else:
            return False
    if started is None:
        return False
    clock = _as_aware(now) or utc_now()
    deadline = started + timedelta(seconds=max(0, int(job.timeout_s)))
    return clock >= deadline


async def mark_timeout(session: AsyncSession, job: Job) -> Job:
    now = utc_now()
    job.status = JOB_STATUS_TIMEOUT
    job.failure_code = "timeout"
    job.finished_at = now
    job.updated_at = now
    attempt = await get_latest_attempt(session, job.id)
    if attempt is not None and not is_terminal(attempt.status):
        attempt.status = JOB_STATUS_TIMEOUT
        attempt.failure_code = "timeout"
        attempt.finished_at = now
    await session.commit()
    await session.refresh(job)
    return job


async def mark_failed(
    session: AsyncSession,
    job: Job,
    *,
    failure_code: str,
) -> Job:
    """Mark job (and open attempt) failed with a failure_code (VAL-JOB-018)."""

    now = utc_now()
    job.status = JOB_STATUS_FAILED
    job.failure_code = failure_code
    job.finished_at = now
    job.updated_at = now
    attempt = await get_latest_attempt(session, job.id)
    if attempt is not None and not is_terminal(attempt.status):
        attempt.status = JOB_STATUS_FAILED
        attempt.failure_code = failure_code
        attempt.finished_at = now
    await session.commit()
    await session.refresh(job)
    return job


async def advance_job_one_step(
    session: AsyncSession,
    job: Job,
    *,
    run_sleep_s: float = 0.0,
    hyper: Any | None = None,
    worker_id: str | None = None,
) -> Job:
    """Advance a single job one lifecycle step (or timeout/cancel/capacity).

    CAS status transitions guarantee only one worker places/launches a job
    at a time (VAL-JOB-016). Capacity bind + concurrency budget for
    VAL-JOB-013..015.
    """

    from hypercluster.domain.job_queue import (
        FAILURE_LAUNCH,
        apply_bind,
        can_admit_to_running,
        cas_status_transition,
        try_bind_capacity,
    )

    _ = worker_id  # reserved for future multi-worker claim telemetry

    await session.refresh(job)
    if is_terminal(job.status):
        return job

    if await should_timeout(job):
        return await mark_timeout(session, job)

    current = job.status
    if current == "submitted":
        won = await cas_status_transition(
            session,
            job_id=job.id,
            from_statuses={"submitted"},
            to_status=JOB_STATUS_ADMITTED,
        )
        if won:
            await session.commit()
        await session.refresh(job)

    # Capacity bind gate while admitted/placing (VAL-JOB-013/014).
    if job.status in {JOB_STATUS_ADMITTED, JOB_STATUS_PLACING}:
        bind = await try_bind_capacity(session, job, hyper=hyper)
        if bind.failure_code:
            return await mark_failed(session, job, failure_code=bind.failure_code)
        if bind.wait:
            # Stay in placing so health/poll observe non-succeeded wait.
            if job.status == JOB_STATUS_ADMITTED:
                won = await cas_status_transition(
                    session,
                    job_id=job.id,
                    from_statuses={JOB_STATUS_ADMITTED},
                    to_status=JOB_STATUS_PLACING,
                )
                if won:
                    await session.commit()
                    await session.refresh(job)
            return job
        if bind.bound:
            await apply_bind(session, job, bind)
            if job.status == JOB_STATUS_ADMITTED:
                won = await cas_status_transition(
                    session,
                    job_id=job.id,
                    from_statuses={JOB_STATUS_ADMITTED},
                    to_status=JOB_STATUS_PLACING,
                )
                if won:
                    await session.commit()
                await session.refresh(job)

    nxt = _FORWARD.get(job.status)
    if nxt is None:
        return job

    # Concurrency budget: hold in placing when large jobs exceed caps (VAL-JOB-015).
    if job.status == JOB_STATUS_PLACING and nxt == JOB_STATUS_PROVISIONING:
        if not await can_admit_to_running(session, job, hyper=hyper):
            return job

    # CAS claim of the next status before side effects (VAL-JOB-016).
    from_status = job.status
    claimed = await cas_status_transition(
        session,
        job_id=job.id,
        from_statuses={from_status},
        to_status=nxt,
        extra={"started_at": job.started_at or utc_now()}
        if nxt == JOB_STATUS_RUNNING and job.started_at is None
        else None,
    )
    if not claimed:
        await session.rollback()
        await session.refresh(job)
        return job

    # Reload after CAS so local object matches the claimed status.
    await session.refresh(job)

    # Side effects on the status we just claimed.
    if nxt == JOB_STATUS_PROVISIONING:
        await _place_job(session, job)
    if nxt == JOB_STATUS_RUNNING:
        job.started_at = job.started_at or utc_now()
        # Forced launch fail (VAL-JOB-018) before collect.
        launch_fail = bool(getattr(hyper, "sim_launch_fail", False)) if hyper else False
        if launch_fail:
            job.status = JOB_STATUS_FAILED
            job.failure_code = FAILURE_LAUNCH
            job.finished_at = utc_now()
            job.updated_at = utc_now()
            attempt = await get_latest_attempt(session, job.id)
            if attempt is None:
                attempt = await _ensure_running_attempt(session, job)
            attempt.status = JOB_STATUS_FAILED
            attempt.failure_code = FAILURE_LAUNCH
            attempt.finished_at = utc_now()
            await session.commit()
            await session.refresh(job)
            return job
        await _ensure_running_attempt(session, job)
    if from_status == JOB_STATUS_RUNNING and nxt == JOB_STATUS_COLLECTING:
        # Optional sim run sleep — re-check cancel/timeout after.
        if run_sleep_s and run_sleep_s > 0:
            import asyncio

            await asyncio.sleep(run_sleep_s)
            await session.refresh(job)
            if is_terminal(job.status):
                return job
            if await should_timeout(job):
                # Revert collecting claim is hard; mark timeout from any.
                return await mark_timeout(session, job)
        attempt = await get_latest_attempt(session, job.id)
        if attempt is None:
            attempt = await _ensure_running_attempt(session, job)
            await session.flush()
        await _collect_success(session, job, attempt)
    if nxt == JOB_STATUS_SUCCEEDED:
        job.finished_at = utc_now()
        job.failure_code = None
        attempt = await get_latest_attempt(session, job.id)
        if attempt is not None and attempt.status != JOB_STATUS_SUCCEEDED:
            attempt.status = JOB_STATUS_SUCCEEDED
            attempt.finished_at = attempt.finished_at or utc_now()

    job.updated_at = utc_now()
    await session.commit()
    await session.refresh(job)
    return job


async def claim_and_advance_job(
    session: AsyncSession,
    job_id: str,
    *,
    worker_id: str = "worker",
    hyper: Any | None = None,
    run_sleep_s: float = 0.0,
) -> Job | None:
    """CAS-style claim a single job by id and advance one step (VAL-JOB-016)."""

    job = await get_job(session, job_id)
    if job is None or is_terminal(job.status):
        return job
    return await advance_job_one_step(
        session,
        job,
        run_sleep_s=run_sleep_s,
        hyper=hyper,
        worker_id=worker_id,
    )


async def drain_jobs_once(
    session: AsyncSession,
    *,
    run_sleep_s: float = 0.0,
    limit: int = 16,
    hyper: Any | None = None,
) -> int:
    """Advance each drainable job by one step. Returns jobs touched."""

    jobs = await list_drainable_jobs(session, limit=limit)
    advanced = 0
    for job in jobs:
        before = job.status
        try:
            updated = await advance_job_one_step(
                session,
                job,
                run_sleep_s=run_sleep_s if job.status == JOB_STATUS_RUNNING else 0.0,
                hyper=hyper,
            )
            if updated.status != before:
                advanced += 1
        except Exception:  # noqa: BLE001 — worker must not die on one bad job
            logger.exception("job lifecycle advance failed for %s", job.id)
            try:
                await session.rollback()
            except Exception:  # noqa: BLE001
                pass
    return advanced


async def run_job_to_terminal(
    session: AsyncSession,
    job_id: str,
    *,
    run_sleep_s: float = 0.0,
    max_steps: int = 20,
    hyper: Any | None = None,
) -> Job:
    """Utility for tests: advance until terminal or max_steps."""

    job = await get_job(session, job_id)
    if job is None:
        raise JobError("job_not_found", "job not found", status_code=404)
    for _ in range(max_steps):
        if is_terminal(job.status):
            return job
        job = await advance_job_one_step(
            session,
            job,
            run_sleep_s=run_sleep_s,
            hyper=hyper,
        )
    return job


__all__ = [
    "JOB_STATUS_ADMITTED",
    "JOB_STATUS_CANCELLED",
    "JOB_STATUS_COLLECTING",
    "JOB_STATUS_FAILED",
    "JOB_STATUS_PLACING",
    "JOB_STATUS_PROVISIONING",
    "JOB_STATUS_RUNNING",
    "JOB_STATUS_SCORING",
    "JOB_STATUS_SUCCEEDED",
    "JOB_STATUS_TIMEOUT",
    "NON_TERMINAL_STATUSES",
    "PLANNER_VERSION",
    "SUCCESS_PATH",
    "TERMINAL_STATUSES",
    "advance_job_one_step",
    "attempt_to_public",
    "build_launch_contract",
    "build_sim_fabric_report",
    "build_sim_nccl_env",
    "build_sim_rankmap",
    "cancel_job",
    "claim_and_advance_job",
    "drain_jobs_once",
    "get_attempt",
    "get_fabric_report",
    "get_latest_attempt",
    "get_placement",
    "get_proofs_for_attempt",
    "is_terminal",
    "job_detail_public",
    "list_attempts",
    "list_drainable_jobs",
    "mark_failed",
    "mark_timeout",
    "post_job_results",
    "run_job_to_terminal",
    "should_timeout",
]
