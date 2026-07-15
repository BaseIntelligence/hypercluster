"""Capacity binding, CAS claim, concurrency budget, fair queue (M3 scaling).

Fulfills VAL-JOB-013..019, 022..024 helpers used by the job lifecycle drain.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import Job, JobAttempt, JobPlacement, Pod, utc_now
from hypercluster.domain.leases import (
    ACTIVE_LEASE_STATUSES,
    get_lease,
    get_pod,
)

logger = logging.getLogger(__name__)

# Keep in sync with job_lifecycle.NON_TERMINAL_STATUSES (avoid circular import).
_NON_TERMINAL_STATUSES = frozenset(
    {
        "submitted",
        "admitted",
        "placing",
        "provisioning",
        "running",
        "collecting",
        "scoring",
    }
)

# Jobs that consume concurrency / world_size budget while mid-flight.
BUDGET_HOLDING_STATUSES = frozenset(
    {
        "provisioning",
        "running",
        "collecting",
        "scoring",
    }
)

FAILURE_NO_CAPACITY = "no_capacity"
FAILURE_FOREIGN_LEASE = "foreign_lease"
FAILURE_LEASE_INACTIVE = "lease_inactive"
FAILURE_LAUNCH = "launch_failed"


class CapacityBindResult:
    """Outcome of a capacity bind attempt."""

    __slots__ = ("bound", "wait", "failure_code", "lease_id", "pod_id")

    def __init__(
        self,
        *,
        bound: bool = False,
        wait: bool = False,
        failure_code: str | None = None,
        lease_id: str | None = None,
        pod_id: str | None = None,
    ) -> None:
        self.bound = bound
        self.wait = wait
        self.failure_code = failure_code
        self.lease_id = lease_id
        self.pod_id = pod_id


def _as_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _hyper_flag(hyper: Any, name: str, default: Any) -> Any:
    if hyper is None:
        return default
    return getattr(hyper, name, default)


async def cas_status_transition(
    session: AsyncSession,
    *,
    job_id: str,
    from_statuses: frozenset[str] | set[str] | tuple[str, ...],
    to_status: str,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Atomic status CAS: only one worker wins the transition (VAL-JOB-016).

    Returns True when this worker applied the update (rowcount == 1).
    """

    values: dict[str, Any] = {
        "status": to_status,
        "updated_at": utc_now(),
    }
    if extra:
        values.update(extra)
    result = await session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status.in_(tuple(from_statuses)))
        .values(**values)
    )
    return bool(getattr(result, "rowcount", 0) == 1)


async def can_admit_to_running(
    session: AsyncSession,
    job: Job,
    *,
    hyper: Any | None,
) -> bool:
    """Whether advancing this job into provisioned/running stays under caps."""

    threshold = int(_hyper_flag(hyper, "large_job_world_size_threshold", 4) or 4)
    max_large = int(_hyper_flag(hyper, "max_concurrent_large_jobs", 4) or 4)
    max_world = int(_hyper_flag(hyper, "max_concurrent_world_size_budget", 64) or 64)

    result = await session.execute(
        select(Job).where(Job.status.in_(tuple(BUDGET_HOLDING_STATUSES)))
    )
    holders = [j for j in result.scalars().all() if j.id != job.id]
    large_count = sum(1 for j in holders if int(j.world_size) >= threshold)
    world_sum = sum(int(j.world_size) for j in holders)

    if int(job.world_size) >= threshold and large_count >= max_large:
        return False
    if world_sum + int(job.world_size) > max_world:
        return False
    return True


def fair_queue_stmt(*, limit: int = 32) -> Select[tuple[Job]]:
    """Order micro jobs ahead of giants for fairness (VAL-JOB-022)."""

    return (
        select(Job)
        .where(Job.status.in_(tuple(_NON_TERMINAL_STATUSES)))
        .order_by(Job.world_size.asc(), Job.created_at.asc())
        .limit(limit)
    )


async def list_drainable_jobs_fair(session: AsyncSession, *, limit: int = 32) -> list[Job]:
    result = await session.execute(fair_queue_stmt(limit=limit))
    return list(result.scalars().all())


async def try_bind_capacity(
    session: AsyncSession,
    job: Job,
    *,
    hyper: Any | None = None,
) -> CapacityBindResult:
    """Bind lease/pod capacity for a job (VAL-JOB-013/014).

    Policy:
    - Explicit lease_id/pod_id → validate renter + active, then bind.
    - sim_auto_capacity=True → treat capacity as available (no FK bind required).
    - Otherwise wait until capacity_wait_timeout_s then fail with no_capacity.
    """

    sim_auto = bool(_hyper_flag(hyper, "sim_auto_capacity", True))
    wait_s = float(_hyper_flag(hyper, "capacity_wait_timeout_s", 2.0) or 2.0)

    # Already bound on the job row.
    if job.lease_id and job.pod_id:
        lease = await get_lease(session, job.lease_id)
        pod = await get_pod(session, job.pod_id)
        if lease is None or pod is None:
            return CapacityBindResult(failure_code=FAILURE_LEASE_INACTIVE)
        if lease.renter_hotkey != job.submitter_hotkey:
            return CapacityBindResult(failure_code=FAILURE_FOREIGN_LEASE)
        if lease.status not in ACTIVE_LEASE_STATUSES:
            return CapacityBindResult(failure_code=FAILURE_LEASE_INACTIVE)
        if pod.lease_id != lease.id:
            return CapacityBindResult(failure_code=FAILURE_LEASE_INACTIVE)
        return CapacityBindResult(
            bound=True,
            lease_id=lease.id,
            pod_id=pod.id,
        )

    # Explicit lease only (pod inferred).
    if job.lease_id and not job.pod_id:
        lease = await get_lease(session, job.lease_id)
        if lease is None:
            return CapacityBindResult(failure_code=FAILURE_LEASE_INACTIVE)
        if lease.renter_hotkey != job.submitter_hotkey:
            return CapacityBindResult(failure_code=FAILURE_FOREIGN_LEASE)
        if lease.status not in ACTIVE_LEASE_STATUSES:
            return CapacityBindResult(failure_code=FAILURE_LEASE_INACTIVE)
        pod_result = await session.execute(select(Pod).where(Pod.lease_id == lease.id))
        pod = pod_result.scalar_one_or_none()
        if pod is None:
            return CapacityBindResult(failure_code=FAILURE_LEASE_INACTIVE)
        return CapacityBindResult(bound=True, lease_id=lease.id, pod_id=pod.id)

    if sim_auto:
        # Synthetic capacity path — no marketplace FK write.
        return CapacityBindResult(bound=True)

    # No capacity configured: wait then fail.
    base = _as_aware(job.admitted_at) or _as_aware(job.created_at) or utc_now()
    elapsed = (utc_now() - base).total_seconds()
    if elapsed >= wait_s:
        return CapacityBindResult(failure_code=FAILURE_NO_CAPACITY)
    return CapacityBindResult(wait=True)


async def apply_bind(
    session: AsyncSession,
    job: Job,
    bind: CapacityBindResult,
) -> None:
    """Persist lease/pod ids on the job when bind yields real marketplace ids."""

    _ = session
    if bind.lease_id and bind.pod_id:
        if job.lease_id != bind.lease_id or job.pod_id != bind.pod_id:
            job.lease_id = bind.lease_id
            job.pod_id = bind.pod_id
            job.updated_at = utc_now()


async def placement_count(session: AsyncSession, job_id: str) -> int:
    result = await session.execute(
        select(JobPlacement.id).where(JobPlacement.job_id == job_id)
    )
    return len(list(result.scalars().all()))


async def active_attempt_count(session: AsyncSession, job_id: str) -> int:
    result = await session.execute(
        select(JobAttempt).where(
            JobAttempt.job_id == job_id,
            JobAttempt.status.in_(("running", "collecting", "scoring")),
        )
    )
    return len(list(result.scalars().all()))


__all__ = [
    "BUDGET_HOLDING_STATUSES",
    "CapacityBindResult",
    "FAILURE_FOREIGN_LEASE",
    "FAILURE_LAUNCH",
    "FAILURE_LEASE_INACTIVE",
    "FAILURE_NO_CAPACITY",
    "active_attempt_count",
    "apply_bind",
    "can_admit_to_running",
    "cas_status_transition",
    "fair_queue_stmt",
    "list_drainable_jobs_fair",
    "placement_count",
    "try_bind_capacity",
]
