"""Job admit domain service (static gates, client_request_id idempotency).

Fulfills VAL-JOB-001..005 for M3 job admit slice. Lifecycle advancement beyond
``admitted`` is owned by later M3 features.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import Job, utc_now

JOB_STATUS_ADMITTED = "admitted"
JOB_STATUS_SUBMITTED = "submitted"

# Hang at admit for this slice; later features own transitions into placement.
ADMIT_STATUSES = frozenset({JOB_STATUS_SUBMITTED, JOB_STATUS_ADMITTED})

DEFAULT_IMAGE_ALLOWLIST = (
    "sha256:sim000000000000000000000000000000000000000000000000000000000001,"
    "sha256:cafebabecafebabecafebabecafebabecafebabecafebabecafebabecafebabe,"
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

DEFAULT_MAX_JOB_WORLD_SIZE = 64
DEFAULT_MAX_JOB_NNODES = 16
DEFAULT_MAX_JOB_NPROC_PER_NODE = 8
DEFAULT_MAX_JOB_TIMEOUT_S = 86_400
DEFAULT_MAX_JOB_GPU_BUDGET = 128

_UNSAFE_ENTRYPOINT_PATTERN = re.compile(
    r"(?:^|[\s;/])"
    r"(?:(?:\.{2})|/etc/|/proc/|/sys/|/root/|/bin/|/usr/bin/|/sbin/)"
    r"|[;&|`$]|[\r\n]|\x00",
)
_SHELL_META = re.compile(r"[;&|`$<>\\]|[\r\n\x00]")
# Absolute path fragments and path traversal in any entrypoint token.
_PATH_UNSAFE_TOKEN = re.compile(
    r"(?:\.\./|\.\.\\|"
    r"^/(?:etc|proc|sys|root|bin|sbin|usr|var|home|tmp)/|"
    r"(?:^|[\s=])/etc/|"
    r"(?:^|[\s=])/bin/|"
    r"(?:^|[\s=])/usr/|"
    r"(?:^|[\s=])/sbin/|"
    r"(?:^|[\s=])/proc/|"
    r"(?:^|[\s=])/sys/)",
    re.IGNORECASE,
)

_VALID_BACKENDS = frozenset({"nccl", "gloo"})
_VALID_FABRIC = frozenset({"auto", "ib", "eth", "nvlink_only"})
_VALID_TEE = frozenset({"none", "tdx", "tdx+gpu_cc"})
_VALID_PLACEMENT = frozenset({"pack", "spread"})


class JobError(Exception):
    """Domain error for job admit / read operations."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def parse_image_allowlist(raw: str | list[str] | None) -> frozenset[str]:
    """Parse comma-separated or list image digest allowlist."""

    if raw is None:
        items = DEFAULT_IMAGE_ALLOWLIST.split(",")
    elif isinstance(raw, list):
        items = [str(x) for x in raw]
    else:
        items = str(raw).split(",")
    cleaned = {x.strip() for x in items if x and str(x).strip()}
    if not cleaned:
        cleaned = {x.strip() for x in DEFAULT_IMAGE_ALLOWLIST.split(",") if x.strip()}
    return frozenset(cleaned)


def _positive_int(value: Any, *, field: str) -> int:
    if value is None:
        raise JobError(
            f"missing_{field}",
            f"{field} is required",
            status_code=422,
        )
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise JobError(
            f"invalid_{field}",
            f"{field} must be a positive integer",
            status_code=422,
        ) from exc
    if number < 1:
        raise JobError(
            f"invalid_{field}",
            f"{field} must be a positive integer",
            status_code=422,
        )
    return number


def validate_world_size_dims(
    *,
    world_size: Any,
    nnodes: Any,
    nproc_per_node: Any,
) -> tuple[int, int, int]:
    """Ensure world_size == nnodes * nproc_per_node (VAL-JOB-003)."""

    ws = _positive_int(world_size, field="world_size")
    nn = _positive_int(nnodes, field="nnodes")
    npp = _positive_int(nproc_per_node, field="nproc_per_node")
    if ws != nn * npp:
        raise JobError(
            "world_size_mismatch",
            f"world_size ({ws}) must equal nnodes * nproc_per_node ({nn} * {npp} = {nn * npp})",
            status_code=422,
        )
    return ws, nn, npp


def validate_image_digest(image_digest: Any, *, allowlist: frozenset[str]) -> str:
    """Fail closed if image is missing or not on allowlist (VAL-JOB-002)."""

    if image_digest is None or not str(image_digest).strip():
        raise JobError(
            "missing_image_digest",
            "image_digest is required",
            status_code=422,
        )
    digest = str(image_digest).strip()
    if digest not in allowlist:
        raise JobError(
            "image_not_allowed",
            "image_digest is not on the challenge allowlist",
            status_code=422,
        )
    return digest


def validate_entrypoint(entrypoint: Any) -> list[str]:
    """Reject path-unsafe / shell-meta entrypoint tokens (VAL-JOB-002)."""

    if entrypoint is None:
        raise JobError(
            "missing_entrypoint",
            "entrypoint is required",
            status_code=422,
        )
    if not isinstance(entrypoint, list) or not entrypoint:
        raise JobError(
            "invalid_entrypoint",
            "entrypoint must be a non-empty list of strings",
            status_code=422,
        )
    tokens: list[str] = []
    for item in entrypoint:
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            raise JobError(
                "invalid_entrypoint",
                "entrypoint tokens must be strings",
                status_code=422,
            )
        token = str(item)
        if token == "" or token.strip() == "":
            raise JobError(
                "path_unsafe",
                "entrypoint contains empty tokens",
                status_code=422,
            )
        if _SHELL_META.search(token) or _PATH_UNSAFE_TOKEN.search(token):
            raise JobError(
                "path_unsafe",
                "entrypoint contains path-unsafe or shell-meta content",
                status_code=422,
            )
        # Absolute binary first tokens like /bin/sh, /usr/bin/...
        if token.startswith("/") and (
            token.startswith(("/bin/", "/usr/", "/sbin/", "/etc/", "/proc/", "/sys/", "/root/"))
            or token in {"/bin/sh", "/bin/bash", "/usr/bin/env"}
        ):
            raise JobError(
                "path_unsafe",
                "entrypoint uses disallowed absolute system path",
                status_code=422,
            )
        if ".." in token.split("/"):
            raise JobError(
                "path_unsafe",
                "entrypoint contains path traversal",
                status_code=422,
            )
        tokens.append(token)
    # Entire joined string scan for smuggling.
    joined = " ".join(tokens)
    if _SHELL_META.search(joined) or ".." in joined:
        raise JobError(
            "path_unsafe",
            "entrypoint contains path-unsafe content",
            status_code=422,
        )
    return tokens


def validate_resource_budget(
    resource: Any,
    *,
    world_size: int,
    nnodes: int,
    timeout_s: int,
    max_world_size: int,
    max_nnodes: int,
    max_nproc_per_node: int,
    nproc_per_node: int,
    max_timeout_s: int,
    max_gpu_budget: int,
) -> dict[str, Any]:
    """Static resource/timeout/dimension caps (VAL-JOB-002 budget_exceeded)."""

    if resource is None:
        raise JobError(
            "missing_resource",
            "resource is required",
            status_code=422,
        )
    if not isinstance(resource, dict):
        raise JobError(
            "invalid_resource",
            "resource must be an object",
            status_code=422,
        )

    if world_size > max_world_size:
        raise JobError(
            "budget_exceeded",
            f"world_size {world_size} exceeds max {max_world_size}",
            status_code=422,
        )
    if nnodes > max_nnodes:
        raise JobError(
            "budget_exceeded",
            f"nnodes {nnodes} exceeds max {max_nnodes}",
            status_code=422,
        )
    if nproc_per_node > max_nproc_per_node:
        raise JobError(
            "budget_exceeded",
            f"nproc_per_node {nproc_per_node} exceeds max {max_nproc_per_node}",
            status_code=422,
        )
    if timeout_s > max_timeout_s:
        raise JobError(
            "budget_exceeded",
            f"timeout_s {timeout_s} exceeds max {max_timeout_s}",
            status_code=422,
        )
    if timeout_s < 1:
        raise JobError(
            "invalid_timeout_s",
            "timeout_s must be a positive integer",
            status_code=422,
        )

    gpus_raw = resource.get("gpus")
    if gpus_raw is not None:
        try:
            gpus = int(gpus_raw)
        except (TypeError, ValueError) as exc:
            raise JobError(
                "invalid_resource",
                "resource.gpus must be an integer",
                status_code=422,
            ) from exc
        if gpus < 0 or not math.isfinite(float(gpus)):
            raise JobError(
                "invalid_resource",
                "resource.gpus must be non-negative",
                status_code=422,
            )
        if gpus > max_gpu_budget:
            raise JobError(
                "budget_exceeded",
                f"resource.gpus {gpus} exceeds max {max_gpu_budget}",
                status_code=422,
            )

    # Multiplying ranks also cannot exceed GPU budget (default rank packs 1 GPU).
    if world_size > max_gpu_budget:
        raise JobError(
            "budget_exceeded",
            f"world_size {world_size} exceeds gpu budget {max_gpu_budget}",
            status_code=422,
        )

    # Normalize a shallow copy for persistence (JSON-safe scalars only).
    out: dict[str, Any] = {}
    for key, value in resource.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, float) and not math.isfinite(value):
                raise JobError(
                    "invalid_resource",
                    f"resource.{key} must be finite",
                    status_code=422,
                )
            out[str(key)] = value
        elif isinstance(value, list):
            out[str(key)] = [
                x for x in value if isinstance(x, (str, int, float, bool)) or x is None
            ]
        elif isinstance(value, dict):
            out[str(key)] = {
                str(k): v
                for k, v in value.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }
    return out


def _normalize_enum(value: Any, *, allowed: frozenset[str], field: str, default: str) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    raw = str(value).strip().lower()
    if raw not in allowed:
        raise JobError(
            f"invalid_{field}",
            f"{field} must be one of {sorted(allowed)}",
            status_code=422,
        )
    return raw


async def get_job(session: AsyncSession, job_id: str) -> Job | None:
    result = await session.execute(select(Job).where(Job.id == job_id))
    return result.scalar_one_or_none()


async def get_job_by_client_request(
    session: AsyncSession,
    *,
    hotkey: str,
    client_request_id: str,
) -> Job | None:
    if not client_request_id:
        return None
    result = await session.execute(
        select(Job).where(
            Job.submitter_hotkey == hotkey,
            Job.client_request_id == client_request_id,
        )
    )
    return result.scalar_one_or_none()


async def list_jobs(
    session: AsyncSession,
    *,
    hotkey: str | None = None,
    status: str | None = None,
) -> list[Job]:
    """List jobs scoped to submitter hotkey (fail-closed when hotkey missing)."""

    if not hotkey:
        return []
    stmt = select(Job).where(Job.submitter_hotkey == hotkey).order_by(Job.created_at.asc())
    if status is not None and status.strip():
        stmt = stmt.where(Job.status == status.strip().lower())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def admit_job(
    session: AsyncSession,
    *,
    hotkey: str,
    image_digest: Any,
    entrypoint: Any,
    world_size: Any,
    nnodes: Any,
    nproc_per_node: Any,
    resource: Any,
    timeout_s: Any,
    client_request_id: str | None = None,
    backend: str | None = "nccl",
    fabric: str | None = "auto",
    tee: str | None = "none",
    env: dict[str, str] | None = None,
    placement_policy: str | None = "pack",
    lease_id: str | None = None,
    pod_id: str | None = None,
    image_allowlist: frozenset[str] | None = None,
    max_world_size: int = DEFAULT_MAX_JOB_WORLD_SIZE,
    max_nnodes: int = DEFAULT_MAX_JOB_NNODES,
    max_nproc_per_node: int = DEFAULT_MAX_JOB_NPROC_PER_NODE,
    max_timeout_s: int = DEFAULT_MAX_JOB_TIMEOUT_S,
    max_gpu_budget: int = DEFAULT_MAX_JOB_GPU_BUDGET,
) -> tuple[Job, bool]:
    """Static-admit a HyperJob.

    Returns ``(job, created)`` where ``created`` is False when an existing row
    was returned for the same ``(hotkey, client_request_id)`` (VAL-JOB-005).
    """

    if not hotkey or not str(hotkey).strip():
        raise JobError("missing_hotkey", "submitter hotkey is required", status_code=401)

    request_key = (client_request_id or "").strip() or None
    if request_key:
        existing = await get_job_by_client_request(
            session,
            hotkey=hotkey,
            client_request_id=request_key,
        )
        if existing is not None:
            return existing, False

    allowlist = image_allowlist if image_allowlist is not None else parse_image_allowlist(None)
    digest = validate_image_digest(image_digest, allowlist=allowlist)
    tokens = validate_entrypoint(entrypoint)
    ws, nn, npp = validate_world_size_dims(
        world_size=world_size,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
    )
    try:
        timeout = int(timeout_s)
    except (TypeError, ValueError) as exc:
        raise JobError(
            "invalid_timeout_s",
            "timeout_s must be a positive integer",
            status_code=422,
        ) from exc
    resource_out = validate_resource_budget(
        resource,
        world_size=ws,
        nnodes=nn,
        nproc_per_node=npp,
        timeout_s=timeout,
        max_world_size=max_world_size,
        max_nnodes=max_nnodes,
        max_nproc_per_node=max_nproc_per_node,
        max_timeout_s=max_timeout_s,
        max_gpu_budget=max_gpu_budget,
    )
    backend_out = _normalize_enum(backend, allowed=_VALID_BACKENDS, field="backend", default="nccl")
    fabric_out = _normalize_enum(fabric, allowed=_VALID_FABRIC, field="fabric", default="auto")
    tee_out = _normalize_enum(tee, allowed=_VALID_TEE, field="tee", default="none")
    placement_out = _normalize_enum(
        placement_policy,
        allowed=_VALID_PLACEMENT,
        field="placement_policy",
        default="pack",
    )

    env_json: str | None = None
    if env is not None:
        if not isinstance(env, dict):
            raise JobError("invalid_env", "env must be an object", status_code=422)
        cleaned = {str(k): str(v) for k, v in env.items()}
        env_json = json.dumps(cleaned)

    now = utc_now()
    job = Job(
        id=str(uuid.uuid4()),
        submitter_hotkey=hotkey,
        client_request_id=request_key,  # None when not idempotent
        status=JOB_STATUS_ADMITTED,
        image_digest=digest,
        entrypoint_json=json.dumps(tokens),
        world_size=ws,
        nnodes=nn,
        nproc_per_node=npp,
        backend=backend_out,
        fabric_mode=fabric_out,
        tee_mode=tee_out,
        env_json=env_json,
        resource_json=json.dumps(resource_out),
        timeout_s=timeout,
        placement_policy=placement_out,
        lease_id=lease_id,
        pod_id=pod_id,
        admitted_at=now,
        created_at=now,
        updated_at=now,
    )
    session.add(job)
    try:
        await session.commit()
    except IntegrityError:
        # Concurrent duplicate client_request_id for same hotkey.
        await session.rollback()
        if request_key:
            existing = await get_job_by_client_request(
                session,
                hotkey=hotkey,
                client_request_id=request_key,
            )
            if existing is not None:
                return existing, False
        raise JobError(
            "job_create_conflict",
            "could not create job due to concurrent conflict",
            status_code=409,
        ) from None
    await session.refresh(job)
    return job, True


def job_to_public(job: Job) -> dict[str, Any]:
    """Serialize job for API responses (no secrets)."""

    return job.to_dict()


__all__ = [
    "ADMIT_STATUSES",
    "DEFAULT_IMAGE_ALLOWLIST",
    "DEFAULT_MAX_JOB_GPU_BUDGET",
    "DEFAULT_MAX_JOB_NNODES",
    "DEFAULT_MAX_JOB_NPROC_PER_NODE",
    "DEFAULT_MAX_JOB_TIMEOUT_S",
    "DEFAULT_MAX_JOB_WORLD_SIZE",
    "JOB_STATUS_ADMITTED",
    "JOB_STATUS_SUBMITTED",
    "JobError",
    "admit_job",
    "get_job",
    "get_job_by_client_request",
    "job_to_public",
    "list_jobs",
    "parse_image_allowlist",
    "validate_entrypoint",
    "validate_image_digest",
    "validate_resource_budget",
    "validate_world_size_dims",
]
