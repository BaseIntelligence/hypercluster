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
from hypercluster.fabric.planner import PLANNER_VERSION  # fabric-planner.v1

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
    policy: str = "pack",
    fabric: str = "auto",
    reports: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Rankmap via topology planner when reports exist; else sequential sim stubs.

    When ``reports`` are provided, uses fabric planner pack/spread (VAL-FAB-004+).
    Without reports, falls back to ``sim-node-{i}`` sequential layout for
    existing job lifecycle tests that do not inject FabricReports.
    """

    if reports:
        from hypercluster.fabric.planner import PlacementRequest, place_ranks, rankmap_as_dicts

        result = place_ranks(
            PlacementRequest(
                job_id=job_id,
                world_size=world_size,
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                policy="spread" if policy == "spread" else "pack",  # type: ignore[arg-type]
                fabric=fabric if fabric in {"auto", "ib", "eth", "nvlink_only"} else "auto",  # type: ignore[arg-type]
                node_reports=list(reports),
            )
        )
        if result.ok:
            return rankmap_as_dicts(result)

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
    reports: list[Any] | None = None,
) -> dict[str, str]:
    """NCCL env matrix via fabric gates (VAL-FAB-021 / planner contract v1).

    eth / auto-on-eth never force ``NCCL_NET=IB``. ``fabric=ib`` only sets IB
    transport when mode evaluation sees active devices (or sim optimistic
    empty-report path still requests IB for pure job stubs without inventory).
    """

    from hypercluster.fabric.gates import build_nccl_env_for_mode, evaluate_fabric_mode

    report_list = list(reports or [])
    mode = (fabric_mode or "auto").strip().lower()
    # Lifecycle sim without bound node reports: preserve prior stub behavior
    # for ib (claim IB keys) so existing job tests keep NCCL_NET=IB; eth never
    # sets IB (VAL-FAB-021).
    if not report_list:
        env: dict[str, str] = {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_SOCKET_IFNAME": "lo",
            "HYPER_BACKEND": backend,
            "HYPER_FABRIC_MODE": mode,
        }
        if mode == "ib":
            env["NCCL_NET"] = "IB"
            env["NCCL_IB_HCA"] = "mlx5_0"
            env["NCCL_IB_GID_INDEX"] = "3"
        elif mode == "eth":
            env["NCCL_NET"] = "Socket"
        elif mode == "auto":
            # auto without inventory stays sockets (VAL-FAB-003 path).
            env["NCCL_NET"] = "Socket"
        else:
            env["NCCL_NET"] = "Socket"
        return env

    mode_eval = evaluate_fabric_mode(fabric_mode=mode, reports=report_list)
    return build_nccl_env_for_mode(
        fabric_mode=mode,
        reports=report_list if mode_eval.ok or mode != "ib" else report_list,
        backend=backend,
    )


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
    rankmap: list[dict[str, Any]] | None = None,
    reports: list[Any] | None = None,
) -> dict[str, Any]:
    """FabricReport-shaped multi-node bundle for sim jobs (VAL-JOB-021 / VAL-FAB-024).

    When ``rankmap`` is provided, bundles digests for each participating node so
    ``|nodes|`` matches ``|unique node_ids in rankmap|``.
    """

    from hypercluster.fabric.report import bundle_job_fabric_report

    effective_rankmap: list[dict[str, Any]]
    if rankmap:
        effective_rankmap = list(rankmap)
    else:
        # Synthetic pack-style stubs when no placement yet.
        effective_rankmap = []
        for i in range(max(1, int(job.nnodes))):
            for local in range(max(1, int(job.nproc_per_node))):
                rank = len(effective_rankmap)
                if rank >= int(job.world_size):
                    break
                effective_rankmap.append(
                    {
                        "rank": rank,
                        "node_id": f"sim-node-{i}",
                        "local_rank": local,
                        "gpu_index": local,
                    }
                )
            if len(effective_rankmap) >= int(job.world_size):
                break

    return bundle_job_fabric_report(
        job_id=job.id,
        attempt_id=attempt_id,
        rankmap=effective_rankmap,
        fabric_mode=job.fabric_mode,
        world_size=int(job.world_size),
        nnodes=int(job.nnodes),
        reports=list(reports) if reports else None,
        nccl_version="sim-2.21.5",
    )


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
    # Persist verify_mode + dstack_verdict_json (VAL-TEE-009).
    proofs = [] if created else await get_proofs_for_attempt(session, attempt.id)
    if not proofs:
        from hypercluster.domain.tee_proofs import (
            ORDINARY_PROOF_TIER,
            build_sim_proof,
        )
        from hypercluster.domain.tee_proofs import (
            SIM_PROOF_TIER as TEE_SIM,
        )

        tier = (proof_tier or TEE_SIM).strip().lower()
        mode = (verify_mode or "sim").strip().lower()
        # For pure sim / ordinary posts keep no-live-bonus invariant.
        if tier in {TEE_SIM, ORDINARY_PROOF_TIER, "none"} or mode == "sim":
            proof_row = build_sim_proof(
                attempt_id=attempt.id,
                job=job,
                integrity_fail=not verified,
            )
            # Honour caller verified flag if they sealed a sim proof themselves.
            proof_row.verified = 1 if verified else 0
            proof_row.proof_tier = tier if tier else proof_row.proof_tier
            proof_row.verify_mode = mode
            session.add(proof_row)
        else:
            # Non-sim claim without offline verify → unverified (VAL-TEE-008).
            verdict = {
                "is_valid": False,
                "quote_verified": False,
                "verify_mode": mode,
                "reason_codes": ["unverified_claim_no_offline_verify"],
            }
            session.add(
                JobProof(
                    id=str(uuid.uuid4()),
                    attempt_id=attempt.id,
                    proof_tier=tier,
                    payload_json=json.dumps(
                        {"result_digest": result_digest, "claimed_tier": tier}
                    ),
                    verified=0,
                    verify_mode=mode if mode in {"offline_fixture", "live", "sim"} else "sim",
                    dstack_verdict_json=json.dumps(verdict),
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
    """Plan rankmap + NCCL env; fail closed for fabric=ib with zero devices.

    VAL-FAB-002/011/021: when bound to a pod, load member FabricReports, require
    full cluster reports for multi-node, and refuse IB mode without devices.
    """

    from hypercluster.domain.fabric_reports import load_latest_reports_for_nodes
    from hypercluster.domain.leases import get_pod
    from hypercluster.fabric.gates import (
        evaluate_cluster_member_reports,
        evaluate_fabric_mode,
    )

    reports: list[Any] = []
    member_ids: list[str] = []
    pod_mode = "single"
    if job.pod_id:
        pod = await get_pod(session, job.pod_id)
        if pod is not None:
            member_ids = list(pod.node_ids())
            pod_mode = pod.mode or ("cluster" if len(member_ids) > 1 else "single")
            if member_ids:
                reports = await load_latest_reports_for_nodes(session, member_ids)
                need_cluster = pod_mode == "cluster" or int(job.nnodes) > 1
                if need_cluster:
                    cluster_eval = evaluate_cluster_member_reports(
                        mode="cluster",
                        member_node_ids=member_ids,
                        reports=reports,
                    )
                    if not cluster_eval.may_launch:
                        raise JobError(
                            cluster_eval.failure_code
                            or "cluster_fabric_reports_incomplete",
                            cluster_eval.reason
                            or "cluster requires FabricReports for all members",
                            status_code=409,
                        )

    mode = (job.fabric_mode or "auto").strip().lower()
    # VAL-FAB-002: with bound reports or empty for ib-required, fail closed.
    if mode == "ib" and (reports or member_ids):
        mode_eval = evaluate_fabric_mode(fabric_mode=mode, reports=reports)
        if not mode_eval.may_succeed:
            raise JobError(
                mode_eval.failure_code or "missing_ib",
                mode_eval.reason or "fabric=ib fails closed without IB devices",
                status_code=409,
            )

    policy = (job.placement_policy or "pack").strip().lower()
    if policy not in {"pack", "spread"}:
        policy = "pack"

    # Prefer topology-aware planner when FabricReports are available.
    planner_graph_digest: str | None = None
    rankmap: list[dict[str, Any]]
    nccl_env: dict[str, str]
    if reports:
        from hypercluster.fabric.planner import PlacementRequest, place_ranks, rankmap_as_dicts

        # Restrict planning to bound member nodes when present.
        plan_reports = reports
        if member_ids:
            allowed = set(member_ids)
            plan_reports = [r for r in reports if r.node_id in allowed] or reports

        fabric_mode = mode if mode in {"auto", "ib", "eth", "nvlink_only"} else "auto"
        plan = place_ranks(
            PlacementRequest(
                job_id=job.id,
                world_size=int(job.world_size),
                nnodes=int(job.nnodes),
                nproc_per_node=int(job.nproc_per_node),
                policy=policy,  # type: ignore[arg-type]
                fabric=fabric_mode,  # type: ignore[arg-type]
                node_reports=list(plan_reports),
            )
        )
        if not plan.ok:
            raise JobError(
                plan.failure_code or "placement_failed",
                plan.reason or "topology planner could not place ranks",
                status_code=409,
            )
        rankmap = rankmap_as_dicts(plan)
        nccl_env = dict(plan.nccl_env)
        nccl_env.setdefault("HYPER_BACKEND", job.backend)
        planner_graph_digest = plan.graph_digest
    else:
        rankmap = build_sim_rankmap(
            job_id=job.id,
            nnodes=int(job.nnodes),
            nproc_per_node=int(job.nproc_per_node),
            world_size=int(job.world_size),
            policy=policy,
            fabric=mode,
            reports=None,
        )
        if member_ids and rankmap:
            for binding in rankmap:
                idx = 0
                try:
                    raw = str(binding.get("node_id") or "")
                    if raw.startswith("sim-node-"):
                        idx = int(raw.split("-")[-1])
                except (TypeError, ValueError):
                    idx = 0
                if 0 <= idx < len(member_ids):
                    binding["node_id"] = member_ids[idx]
        nccl_env = build_sim_nccl_env(
            fabric_mode=job.fabric_mode,
            backend=job.backend,
            reports=None,
        )

    launch = build_launch_contract(job=job, rankmap=rankmap, nccl_env=nccl_env)
    graph_digest = planner_graph_digest or _sha256_hex(
        _canonical_json(
            {
                "rankmap": rankmap,
                "policy": policy,
                "member_ids": member_ids,
                "planner_version": PLANNER_VERSION,
            }
        )
    )

    existing = await get_placement(session, job.id)
    if existing is None:
        session.add(
            JobPlacement(
                id=str(uuid.uuid4()),
                job_id=job.id,
                rankmap_json=json.dumps(rankmap),
                placement_policy=policy,
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


async def _collect_success(
    session: AsyncSession,
    job: Job,
    attempt: JobAttempt,
    *,
    hyper: Any | None = None,
) -> None:
    """Collect launch metrics + multi-node fabric bundle (VAL-FAB-013/014/015/024)."""

    from hypercluster.domain.fabric_reports import load_latest_reports_for_nodes
    from hypercluster.domain.leases import get_pod
    from hypercluster.fabric.launcher import (
        LaunchRequest,
        placement_result_from_dicts,
        sim_launch,
    )

    placement = await get_placement(session, job.id)
    rankmap: list[dict[str, Any]] = list(placement.rankmap()) if placement is not None else []
    nccl_env: dict[str, str] = dict(placement.nccl_env()) if placement is not None else {}
    graph_digest = (placement.graph_digest if placement is not None else None) or ""
    planner_version = (
        placement.planner_version if placement is not None else PLANNER_VERSION
    )

    reports: list[Any] = []
    if job.pod_id:
        pod = await get_pod(session, job.pod_id)
        if pod is not None:
            member_ids = list(pod.node_ids())
            if member_ids:
                reports = await load_latest_reports_for_nodes(session, member_ids)

    # Honesty injects from HyperSettings (sim knobs).
    honesty_level = "l1"
    inventory_spoof = False
    eth_fallback_injected = False
    inject_status: str | None = None
    inject_sleep_s = 0.0
    if hyper is not None:
        honesty_level = str(getattr(hyper, "sim_honesty_level", "l1") or "l1")
        inventory_spoof = bool(getattr(hyper, "sim_inventory_spoof", False))
        # VAL-FAB-012 black-box: HYPER_SIM_ETH_FALLBACK → eth_fallback_injected
        eth_fallback_injected = bool(getattr(hyper, "sim_eth_fallback", False))
        if bool(getattr(hyper, "sim_launch_fail", False)):
            inject_status = "failed"
        if bool(getattr(hyper, "sim_launch_timeout", False)):
            inject_status = "timeout"
        inject_sleep_s = float(getattr(hyper, "sim_launch_inject_sleep_s", 0.0) or 0.0)

    place_result = placement_result_from_dicts(
        rankmap=rankmap
        or [
            {
                "rank": r,
                "node_id": f"sim-node-{r // max(1, int(job.nproc_per_node))}",
                "local_rank": r % max(1, int(job.nproc_per_node)),
                "gpu_index": r % max(1, int(job.nproc_per_node)),
            }
            for r in range(int(job.world_size))
        ],
        nccl_env=nccl_env,
        planner_version=planner_version,
        graph_digest=graph_digest or "sha256:" + ("0" * 64),
        job_id=job.id,
    )
    if not rankmap:
        rankmap = [b.to_public() for b in place_result.rankmap]

    launch_result = sim_launch(
        LaunchRequest(
            placement=place_result,
            image_digest=job.image_digest,
            entrypoint=list(job.entrypoint()),
            env=dict(job.env() or {}),
            timeout_s=int(job.timeout_s),
            fabric_mode=job.fabric_mode or "auto",
            honesty_level=honesty_level if honesty_level in {"l0", "l1", "l2"} else "l1",  # type: ignore[arg-type]
            inject_status=inject_status,  # type: ignore[arg-type]
            inject_sleep_s=inject_sleep_s,
            eth_fallback_injected=eth_fallback_injected,
            inventory_spoof=inventory_spoof,
            node_reports=list(reports),
            seed=0,
        )
    )

    fab = build_sim_fabric_report(
        job=job,
        attempt_id=attempt.id,
        rankmap=rankmap,
        reports=reports or None,
    )

    # If provider already sealed results, keep digests (idempotent) and only
    # ensure fabric report / proof rows exist.
    if attempt.result_digest is not None and attempt.finished_at is not None:
        report = await get_fabric_report(session, job.id)
        if report is None and attempt.fabric_report_digest:
            if "report_digest" not in fab or attempt.fabric_report_digest:
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
        from hypercluster.domain.tee_proofs import ensure_attempt_proof, score_attempt_with_tee

        proofs = await get_proofs_for_attempt(session, attempt.id)
        if not proofs:
            await ensure_attempt_proof(session, job=job, attempt=attempt)
        await score_attempt_with_tee(
            session,
            job=job,
            attempt=attempt,
            correctness=1.0,
            efficiency=1.0,
            fabric_gate=1.0,
            hyper=hyper,
        )
        return

    # Map LaunchResult status → attempt/job (VAL-FAB-014).
    if launch_result.status == "failed":
        attempt.status = JOB_STATUS_FAILED
        attempt.failure_code = launch_result.failure_code or "sim_launch_fail"
        attempt.metrics_json = json.dumps(launch_result.metrics_json())
        attempt.fabric_report_digest = fab["report_digest"]
        attempt.finished_at = utc_now()
        job.status = JOB_STATUS_FAILED
        job.failure_code = attempt.failure_code
        job.finished_at = utc_now()
        return

    if launch_result.status == "timeout":
        attempt.status = JOB_STATUS_TIMEOUT
        attempt.failure_code = "timeout"
        attempt.metrics_json = json.dumps(launch_result.metrics_json())
        attempt.fabric_report_digest = fab["report_digest"]
        attempt.finished_at = utc_now()
        job.status = JOB_STATUS_TIMEOUT
        job.failure_code = "timeout"
        job.finished_at = utc_now()
        return

    metrics = launch_result.metrics_json()
    output_digest = _sha256_hex(
        _canonical_json(
            {
                "job_id": job.id,
                "attempt_no": attempt.attempt_no,
                "entrypoint": job.entrypoint(),
                "ok": True,
                "fabric_artifact_digest": launch_result.fabric_artifact_digest,
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
                "failure_code": launch_result.failure_code,
                "fabric_gate": launch_result.fabric_gate,
                "composite": launch_result.composite,
            }
        )
    )
    attempt.status = JOB_STATUS_SUCCEEDED
    attempt.metrics_json = json.dumps(metrics)
    attempt.fabric_report_digest = fab["report_digest"]
    attempt.output_digest = output_digest
    attempt.result_digest = result_digest
    attempt.finished_at = utc_now()
    if launch_result.failure_code:
        attempt.failure_code = launch_result.failure_code

    from hypercluster.domain.tee_proofs import ensure_attempt_proof, score_attempt_with_tee

    proofs = await get_proofs_for_attempt(session, attempt.id)
    if not proofs:
        # VAL-TEE-005/009/015: sim/ordinary proof with verdict JSON; no live bonus.
        await ensure_attempt_proof(
            session,
            job=job,
            attempt=attempt,
            integrity_fail=bool(launch_result.integrity_fail),
            fabric_gate=float(launch_result.fabric_gate),
        )

    # Persist four-factor score; tee_bonus locked to 1.0 for sim/ordinary.
    efficiency = 1.0
    if launch_result.metrics is not None:
        efficiency = float(getattr(launch_result.metrics, "efficiency", 1.0) or 1.0)
    await score_attempt_with_tee(
        session,
        job=job,
        attempt=attempt,
        correctness=0.0 if launch_result.integrity_fail else 1.0,
        efficiency=efficiency,
        fabric_gate=float(launch_result.fabric_gate),
        hyper=hyper,
        integrity_fail=bool(launch_result.integrity_fail),
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
        try:
            await _place_job(session, job)
        except JobError as exc:
            # Fabric admission failures (VAL-FAB-002/011) → terminal failed.
            job.status = JOB_STATUS_FAILED
            job.failure_code = exc.code
            job.finished_at = utc_now()
            job.updated_at = utc_now()
            await session.commit()
            await session.refresh(job)
            return job
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
        await _collect_success(session, job, attempt, hyper=hyper)
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
