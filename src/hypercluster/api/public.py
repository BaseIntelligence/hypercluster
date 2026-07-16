"""Public challenge routes (proxied by Base master when registered).

Identity routes (`/health`, `/ready`, `/version`) are installed by
`create_challenge_app` and are not registered here. Internal routes under
`/internal/*` are owned by the SDK factory and must never carry `@public_route`.

Marketplace providers/nodes/offers/leases/pods (VAL-MKT-001..021, 025..029, 031).
"""

from __future__ import annotations

from typing import Any

from base.challenge_sdk import public_route
from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from hypercluster.api.auth import DbSession, RequireMiner
from hypercluster.domain.fabric_reports import (
    FabricReportError,
    fabric_scan_node,
    get_latest_fabric_report,
    report_to_public,
)
from hypercluster.domain.gpu_probes import (
    GpuProbeError,
    attach_external_evidence,
    evidence_to_public,
    evidence_to_summary,
    get_evidence_global,
    get_latest_node_evidence,
    get_node_evidence,
    list_node_evidence,
    run_node_gpu_probe,
    soft_heartbeat_probe_meta,
)
from hypercluster.domain.job_lifecycle import (
    attempt_to_public,
    cancel_job,
    get_attempt,
    get_fabric_report,
    get_latest_attempt,
    get_placement,
    get_proofs_for_attempt,
    job_detail_public,
    post_job_results,
)
from hypercluster.domain.jobs import (
    JobError,
    admit_job,
    get_job,
    job_to_public,
    parse_image_allowlist,
)
from hypercluster.domain.jobs import (
    list_jobs as list_jobs_for_hotkey,
)
from hypercluster.domain.leases import (
    LeaseError,
    get_lease,
    get_pod,
    get_pod_by_lease,
    lease_to_public,
    list_leases,
    pod_to_public,
    rent_offer,
    terminate_lease,
)
from hypercluster.domain.nodes import (
    NodeError,
    get_node,
    list_nodes,
    node_heartbeat,
    node_to_public,
    register_node,
)
from hypercluster.domain.offers import (
    DEFAULT_MAX_OFFER_LIFETIME_HOURS,
    DEFAULT_MAX_OFFER_PRICE_PER_HOUR,
    OFFER_STATUS_LISTED,
    OfferError,
    create_offer,
    get_offer,
    list_offers,
    offer_to_public,
    parse_require_ib_query,
    withdraw_offer,
)
from hypercluster.domain.providers import (
    ProviderError,
    get_provider_by_hotkey,
    list_providers,
    provider_heartbeat,
    provider_to_public,
    register_provider,
)

router = APIRouter()


class ProviderRegisterRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=256)


class NodeRegisterRequest(BaseModel):
    gpu_model: str = Field(..., min_length=1, max_length=128)
    gpu_count: int = Field(..., ge=1)
    hostname: str | None = Field(default=None, max_length=256)
    ssh_endpoint: str | None = Field(default=None, max_length=512)
    cpu_cores: int | None = Field(default=None, ge=1)
    mem_gb: float | None = Field(default=None, ge=0)
    location_hint: str | None = Field(default=None, max_length=128)
    tee_capability: str = Field(default="none", max_length=32)
    inventory: dict[str, Any] | None = None
    node_id: str | None = Field(default=None, max_length=36)


class NodeHeartbeatRequest(BaseModel):
    node_id: str | None = Field(default=None, max_length=36)


class GpuProbeRequest(BaseModel):
    """Start GPU probe on a owned node (VAL-GPU-001).

    Never accept raw PEM: only optional key_ref {kind,name} for env/file.
    """

    mode: str = Field(default="full", max_length=16)
    timeout_s: int | None = Field(default=None, ge=1, le=3600)
    # key_ref only (kind=env|file, name). private keys in body are rejected.
    key_ref: dict[str, str] | None = None
    # CI/FakeSsh: optional fixture name override (ignored when real transport).
    fixture: str | None = Field(default=None, max_length=64)

    model_config = {"extra": "forbid"}


class ExternalGpuEvidenceRequest(BaseModel):
    """Ops attach of externally gathered evidence (VAL-GPU-006)."""

    evidence: dict[str, Any]
    claimed_digest: str | None = Field(default=None, max_length=128)

    model_config = {"extra": "forbid"}


class FabricScanRequest(BaseModel):
    """Fabric-scan body (VAL-FAB-018). Sim source is the CI default."""

    source: str = Field(default="sim", max_length=32)
    seed: int = Field(default=0, ge=0)
    topo_variant: str = Field(default="pack", max_length=32)


class OfferCreateRequest(BaseModel):
    """Offer publish body; price/lifetime hard guards also enforced in domain."""

    node_ids: list[str] = Field(..., min_length=1)
    # Optional on the wire so missing keys surface as domain 422 codes (not body-schema),
    # matching VAL-MKT-009 matrix of "missing price/lifetime".
    price_per_hour: float | None = Field(default=None)
    max_lifetime_hours: float | None = Field(default=None)
    mode: str = Field(default="single", max_length=16)
    require_ib: bool = False
    tee: str | None = Field(default=None, max_length=32)
    gpu_model: str | None = Field(default=None, max_length=128)
    gpu_count: int | None = Field(default=None, ge=1)
    location_hint: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None


class RentRequest(BaseModel):
    """Renter rent body; lifetime ≤ offer max; optional max_price renter bound."""

    lifetime_hours: float | None = Field(default=None, gt=0)
    max_price: float | None = Field(default=None, gt=0)


class TerminateLeaseRequest(BaseModel):
    reason: str | None = Field(default="renter_cancel", max_length=256)


class JobAdmitRequest(BaseModel):
    """HyperJob admit body (architecture §6.1); static gates in domain."""

    image_digest: str = Field(..., min_length=1, max_length=256)
    entrypoint: list[str] = Field(..., min_length=1)
    world_size: int = Field(..., ge=1)
    nnodes: int = Field(..., ge=1)
    nproc_per_node: int = Field(..., ge=1)
    resource: dict[str, Any]
    timeout_s: int = Field(..., ge=1)
    client_request_id: str | None = Field(default=None, max_length=128)
    backend: str = Field(default="nccl", max_length=32)
    fabric: str = Field(default="auto", max_length=32)
    tee: str = Field(default="none", max_length=32)
    env: dict[str, str] | None = None
    placement_policy: str = Field(default="pack", max_length=16)
    lease_id: str | None = Field(default=None, max_length=36)
    pod_id: str | None = Field(default=None, max_length=36)


class JobCancelRequest(BaseModel):
    reason: str | None = Field(default="user_cancel", max_length=256)


class IdleReclaimRequest(BaseModel):
    """Sim-only idle reclaim tick (VAL-CROSS-011 / VAL-MKT-020).

    Ages node heartbeats optionally, then runs the tenant short-circuit
    reclaim path so free stale inventory can go offline without killing
    active rentals.
    """

    liveness_seconds: int = Field(default=30, ge=1, le=86_400)
    age_heartbeats_seconds: int | None = Field(default=None, ge=0, le=86_400_000)


class DrainModeRequest(BaseModel):
    """Sim/ops drain toggle (VAL-CROSS-026).

    When ``draining=true``, the required ``not_draining`` readiness probe fails
    so ``GET /ready`` returns 503 and the SDK mutation middleware rejects new
    writes with ``runtime_not_ready``. Combined worker continues finishing
    in-flight jobs from SQLite.
    """

    draining: bool = True


class JobResultsRequest(BaseModel):
    """Provider/worker result envelope (VAL-JOB-009 attempt-keyed).

    Optional TEE fields (VAL-TEE-008/009/012): when quote_b64 is provided with
    verify_mode=offline_fixture the challenge verifies offline and persists
    dstack_verdict_json + verified flag. Garbage quotes stay unverified and get
    no tee_bonus.
    """

    attempt_no: int = Field(default=1, ge=1)
    status: str = Field(default="succeeded", max_length=32)
    metrics: dict[str, Any] | None = None
    fabric_report_digest: str | None = Field(default=None, max_length=128)
    output_digest: str | None = Field(default=None, max_length=128)
    proof_tier: str = Field(default="sim", max_length=32)
    verified: bool = True
    verify_mode: str = Field(default="sim", max_length=32)
    failure_code: str | None = Field(default=None, max_length=64)
    # Optional integrity inject codes (VAL-CROSS-025) — seals composite 0.
    integrity_codes: list[str] | None = None
    # Optional offline TEE material.
    quote_b64: str | None = None
    gpu_evidence: dict[str, Any] | None = None
    report_data_hex: str | None = None
    tee_nonce: str | None = None


def _header_hotkey(request: Request) -> str | None:
    return request.headers.get("x-hotkey") or request.headers.get("X-Hotkey")


def _offer_caps(request: Request) -> tuple[float, float]:
    """Read system offer caps from HyperSettings (env-tunable)."""

    hyper = getattr(request.app.state, "hyper_settings", None)
    price_cap = DEFAULT_MAX_OFFER_PRICE_PER_HOUR
    lifetime_cap = DEFAULT_MAX_OFFER_LIFETIME_HOURS
    if hyper is not None:
        price_cap = float(getattr(hyper, "max_offer_price_per_hour", price_cap) or price_cap)
        lifetime_cap = float(
            getattr(hyper, "max_offer_lifetime_hours", lifetime_cap) or lifetime_cap
        )
    return price_cap, lifetime_cap


def _job_admit_kwargs(request: Request) -> dict[str, Any]:
    """Resolve job admit caps/allowlist from HyperSettings."""

    hyper = getattr(request.app.state, "hyper_settings", None)
    allowlist_raw = None
    max_world = 64
    max_nnodes = 16
    max_nproc = 8
    max_timeout = 86_400
    max_gpus = 128
    if hyper is not None:
        allowlist_raw = getattr(hyper, "job_image_allowlist", None)
        max_world = int(getattr(hyper, "max_job_world_size", max_world) or max_world)
        max_nnodes = int(getattr(hyper, "max_job_nnodes", max_nnodes) or max_nnodes)
        max_nproc = int(getattr(hyper, "max_job_nproc_per_node", max_nproc) or max_nproc)
        max_timeout = int(getattr(hyper, "max_job_timeout_s", max_timeout) or max_timeout)
        max_gpus = int(getattr(hyper, "max_job_gpu_budget", max_gpus) or max_gpus)
    return {
        "image_allowlist": parse_image_allowlist(allowlist_raw),
        "max_world_size": max_world,
        "max_nnodes": max_nnodes,
        "max_nproc_per_node": max_nproc,
        "max_timeout_s": max_timeout,
        "max_gpu_budget": max_gpus,
    }


@public_route(tags=["marketplace"])
@router.post("/v1/providers/register", status_code=status.HTTP_200_OK)
async def providers_register(
    body: ProviderRegisterRequest,
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Provider hotkey onboarding (VAL-MKT-001). Idempotent per hotkey."""

    try:
        provider, created = await register_provider(
            session,
            hotkey=identity.hotkey,
            display_name=body.display_name,
        )
    except ProviderError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    payload = provider_to_public(provider)
    payload["created"] = created
    return payload


@public_route(tags=["marketplace"])
@router.get("/v1/providers")
async def providers_list(
    session: DbSession,
    request: Request,
) -> dict[str, Any]:
    """List providers. Optional X-Hotkey scopes to owner (VAL-MKT-002)."""

    providers = await list_providers(session, hotkey=_header_hotkey(request))
    return {"items": [provider_to_public(p) for p in providers]}


@public_route(tags=["marketplace"])
@router.get("/v1/providers/me")
async def providers_me(
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Return the caller's own provider (requires signature)."""

    provider = await get_provider_by_hotkey(session, identity.hotkey)
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "provider_not_found", "message": "not registered"},
        )
    return provider_to_public(provider)


@public_route(tags=["marketplace"])
@router.post("/v1/providers/heartbeat")
async def providers_heartbeat(
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Advance provider liveness without mutating identity (VAL-MKT-003)."""

    try:
        provider = await provider_heartbeat(session, hotkey=identity.hotkey)
    except ProviderError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return provider_to_public(provider)


@public_route(tags=["marketplace"])
@router.post("/v1/nodes", status_code=status.HTTP_200_OK)
async def nodes_register(
    body: NodeRegisterRequest,
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Register / update node inventory with GPU + IB fields (VAL-MKT-004/005)."""

    try:
        node, created = await register_node(
            session,
            hotkey=identity.hotkey,
            gpu_model=body.gpu_model,
            gpu_count=body.gpu_count,
            hostname=body.hostname,
            ssh_endpoint=body.ssh_endpoint,
            cpu_cores=body.cpu_cores,
            mem_gb=body.mem_gb,
            location_hint=body.location_hint,
            tee_capability=body.tee_capability,
            inventory=body.inventory,
            node_id=body.node_id,
        )
    except NodeError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    payload = node_to_public(node)
    payload["created"] = created
    return payload


@public_route(tags=["marketplace"])
@router.get("/v1/nodes")
async def nodes_list(
    session: DbSession,
    request: Request,
) -> dict[str, Any]:
    """List nodes with capability fields (VAL-MKT-007). X-Hotkey scopes to owner."""

    nodes = await list_nodes(session, hotkey=_header_hotkey(request))
    return {"items": [node_to_public(n) for n in nodes]}


@public_route(tags=["marketplace"])
@router.post("/v1/nodes/heartbeat")
async def nodes_heartbeat(
    identity: RequireMiner,
    session: DbSession,
    request: Request,
    body: NodeHeartbeatRequest | None = None,
) -> dict[str, Any]:
    """Refresh last_heartbeat for owned node(s) (VAL-MKT-006 / VAL-GPU-011).

    Under ``HYPER_REQUIRE_LIVE_EVIDENCE``, unverified nodes get soft-warning
    metadata (or soft 409 when mode is fail_closed). Never 5xx solely due to
    missing probe evidence defaults.
    """

    req = body if body is not None else NodeHeartbeatRequest()
    hyper = getattr(request.app.state, "hyper_settings", None)
    liveness = 120
    require_ev = False
    require_mode = "soft"
    if hyper is not None:
        liveness = int(getattr(hyper, "node_liveness_seconds", 120))
        require_ev = bool(getattr(hyper, "require_live_evidence", False))
        require_mode = str(getattr(hyper, "require_live_evidence_mode", "soft") or "soft")
    try:
        nodes = await node_heartbeat(
            session,
            hotkey=identity.hotkey,
            node_id=req.node_id,
            liveness_seconds=liveness,
        )
    except NodeError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    meta = soft_heartbeat_probe_meta(
        nodes,
        require_live_evidence=require_ev,
        mode=require_mode,
    )
    if meta.get("block_status") == 409:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": meta.get("code") or "gpu_probe_unverified",
                "message": meta.get("message") or "live evidence required for heartbeat",
                "unverified_node_ids": meta.get("unverified_node_ids") or [],
            },
        )
    payload: dict[str, Any] = {"items": [node_to_public(n) for n in nodes]}
    if require_ev:
        payload["require_live_evidence"] = True
        payload["gpu_probe_warning"] = bool(meta.get("gpu_probe_warning"))
        payload["unverified_node_ids"] = list(meta.get("unverified_node_ids") or [])
        payload["require_live_evidence_mode"] = require_mode
    return payload


@public_route(tags=["marketplace"])
@router.get("/v1/nodes/{node_id}")
async def nodes_get(
    node_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Get a single node with capability fields (VAL-MKT-007)."""

    node = await get_node(session, node_id)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "node_not_found", "message": "node not found"},
        )
    return node_to_public(node)


@public_route(tags=["gpu-probe"])
@router.post("/v1/nodes/{node_id}/probes/gpu", status_code=status.HTTP_200_OK)
async def nodes_probe_gpu(
    node_id: str,
    body: GpuProbeRequest,
    identity: RequireMiner,
    session: DbSession,
    request: Request,
) -> dict[str, Any]:
    """Start SSH GPU probe (owner-signed; FakeSsh pass-all → status=passed).

    VAL-GPU-001 / never accept raw PEM / never set_weights.
    """

    hyper = getattr(request.app.state, "hyper_settings", None)
    raw_body = body.model_dump(mode="json")
    try:
        evidence, _row = await run_node_gpu_probe(
            session,
            node_id=node_id,
            hotkey=identity.hotkey,
            mode="quick" if str(body.mode).lower() == "quick" else "full",
            timeout_s=body.timeout_s,
            key_ref=body.key_ref,
            fixture_name=body.fixture,
            body=raw_body,
            settings=hyper,
        )
    except GpuProbeError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return evidence_to_public(evidence)


@public_route(tags=["gpu-probe"])
@router.get("/v1/nodes/{node_id}/probes/gpu/latest")
async def nodes_probe_gpu_latest(
    node_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Latest GPU evidence for a node (black-box GET; VAL-GPU-002)."""

    row = await get_latest_node_evidence(session, node_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "gpu_evidence_not_found",
                "message": "no GPU evidence for node",
            },
        )
    return evidence_to_public(row)


@public_route(tags=["gpu-probe"])
@router.get("/v1/nodes/{node_id}/probes/gpu/{evidence_id}")
async def nodes_probe_gpu_by_id(
    node_id: str,
    evidence_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Evidence by id with checks + redacted raw (VAL-GPU-003)."""

    row = await get_node_evidence(session, node_id, evidence_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "gpu_evidence_not_found",
                "message": "evidence not found for node",
            },
        )
    return evidence_to_public(row)


@public_route(tags=["gpu-probe"])
@router.get("/v1/nodes/{node_id}/probes/gpu")
async def nodes_probe_gpu_list(
    node_id: str,
    session: DbSession,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """List evidence newest-first; empty for unknown/unprobed node (VAL-GPU-004)."""

    rows = await list_node_evidence(session, node_id, limit=limit)
    return {"items": [evidence_to_summary(r) for r in rows]}


@public_route(tags=["gpu-probe"])
@router.get("/v1/evidence/gpu/{evidence_id}")
async def evidence_gpu_global(
    evidence_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Global evidence lookup by id (VAL-GPU-005)."""

    row = await get_evidence_global(session, evidence_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "gpu_evidence_not_found",
                "message": "evidence not found",
            },
        )
    return evidence_to_public(row)


@public_route(tags=["gpu-probe"])
@router.post("/v1/nodes/{node_id}/evidence/gpu", status_code=status.HTTP_200_OK)
async def nodes_evidence_gpu_attach(
    node_id: str,
    body: ExternalGpuEvidenceRequest,
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Owner-signed external evidence attach; reject bad digests (VAL-GPU-006)."""

    try:
        evidence, _row = await attach_external_evidence(
            session,
            node_id=node_id,
            hotkey=identity.hotkey,
            evidence_payload=body.evidence,
            claimed_digest=body.claimed_digest,
            body=body.model_dump(mode="json"),
        )
    except GpuProbeError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    payload = evidence_to_public(evidence)
    payload["attached"] = True
    return payload


@public_route(tags=["fabric"])
@router.post("/v1/nodes/{node_id}/fabric-scan", status_code=status.HTTP_200_OK)
async def nodes_fabric_scan(
    node_id: str,
    body: FabricScanRequest,
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Produce and accept a FabricReport for a registered node (VAL-FAB-018).

    Signing identity must own the node (or node must be public/sim in ownership); ownership
    is enforced against marketplace provider linkage.
    """

    node = await get_node(session, node_id)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "node_not_found", "message": "node not found"},
        )
    provider = await get_provider_by_hotkey(session, identity.hotkey)
    if provider is None or provider.id != node.provider_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "node_not_owned", "message": "node belongs to another provider"},
        )

    source = (body.source or "sim").strip().lower()
    if source not in {"sim", "scan", "inject", "manual"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_source", "message": "source must be sim|scan|inject|manual"},
        )
    try:
        report = await fabric_scan_node(
            session,
            node_id=node_id,
            source=source,  # type: ignore[arg-type]
            seed=int(body.seed),
            topo_variant=body.topo_variant or "pack",
        )
    except FabricReportError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return report_to_public(report)


@public_route(tags=["fabric"])
@router.get("/v1/nodes/{node_id}/fabric-report")
async def nodes_fabric_report(
    node_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Return latest accepted fabric report for a node (VAL-FAB-001/018)."""

    node = await get_node(session, node_id)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "node_not_found", "message": "node not found"},
        )
    row = await get_latest_fabric_report(session, node_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "fabric_report_not_found",
                "message": "no fabric report for node; run fabric-scan first",
            },
        )
    return report_to_public(row)


@public_route(tags=["marketplace"])
@router.post("/v1/offers", status_code=status.HTTP_200_OK)
async def offers_create(
    body: OfferCreateRequest,
    identity: RequireMiner,
    session: DbSession,
    request: Request,
) -> dict[str, Any]:
    """Publish capacity offer with hard price/lifetime guards (VAL-MKT-008..011)."""

    price_cap, lifetime_cap = _offer_caps(request)
    try:
        offer = await create_offer(
            session,
            hotkey=identity.hotkey,
            node_ids=body.node_ids,
            price_per_hour=body.price_per_hour,
            max_lifetime_hours=body.max_lifetime_hours,
            mode=body.mode,
            require_ib=body.require_ib,
            tee=body.tee,
            gpu_model=body.gpu_model,
            gpu_count=body.gpu_count,
            location_hint=body.location_hint,
            metadata=body.metadata,
            max_price_cap=price_cap,
            max_lifetime_cap=lifetime_cap,
        )
    except OfferError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return offer_to_public(offer)


@public_route(tags=["marketplace"])
@router.get("/v1/offers")
async def offers_list(
    session: DbSession,
    gpu_model: str | None = Query(default=None),
    require_ib: str | None = Query(default=None),
    tee: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    mode: str | None = Query(default=None),
) -> dict[str, Any]:
    """Browse marketplace offers with composable filters (VAL-MKT-025..029).

    Default status is ``listed`` (rentable catalog). Pass ``status`` to override
    (e.g. ``withdrawn``). Capability filters compose AND with status.
    """

    # Default browse: listed only so withdrawn/leased never reappear as rentable.
    status_value: str | None
    if status_filter is None:
        status_value = OFFER_STATUS_LISTED
    elif status_filter.strip().lower() in {"", "all", "*"}:
        status_value = None
    else:
        status_value = status_filter.strip().lower()

    try:
        require_ib_flag = parse_require_ib_query(require_ib)
        items = await list_offers(
            session,
            status=status_value,
            gpu_model=gpu_model,
            require_ib=require_ib_flag,
            tee=tee,
            mode=mode,
        )
    except OfferError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return {"items": [offer_to_public(o) for o in items]}


@public_route(tags=["marketplace"])
@router.get("/v1/offers/{offer_id}")
async def offers_get(
    offer_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Get a single offer by id (any status)."""

    offer = await get_offer(session, offer_id)
    if offer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "offer_not_found", "message": "offer not found"},
        )
    return offer_to_public(offer)


@public_route(tags=["marketplace"])
@router.delete("/v1/offers/{offer_id}")
async def offers_withdraw(
    offer_id: str,
    identity: RequireMiner,
    session: DbSession,
) -> dict[str, Any]:
    """Withdraw listing (VAL-MKT-012); owner-only; fail-closed under active lease."""

    try:
        offer = await withdraw_offer(
            session,
            hotkey=identity.hotkey,
            offer_id=offer_id,
        )
    except OfferError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return offer_to_public(offer)


@public_route(tags=["marketplace"])
@router.post("/v1/offers/{offer_id}/rent", status_code=status.HTTP_200_OK)
async def offers_rent(
    offer_id: str,
    identity: RequireMiner,
    session: DbSession,
    body: RentRequest | None = None,
) -> dict[str, Any]:
    """Rent listed offer → exclusive lease + pod (VAL-MKT-013/014/017/019)."""

    req = body if body is not None else RentRequest()
    try:
        lease, pod = await rent_offer(
            session,
            renter_hotkey=identity.hotkey,
            offer_id=offer_id,
            lifetime_hours=req.lifetime_hours,
            max_price=req.max_price,
            sim_ready=True,
        )
    except LeaseError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return {
        "lease": lease_to_public(lease),
        "pod": pod_to_public(pod),
    }


@public_route(tags=["marketplace"])
@router.get("/v1/leases")
async def leases_list(
    session: DbSession,
    request: Request,
    offer_id: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
) -> dict[str, Any]:
    """List leases for renter and/or provider hotkey (VAL-MKT-016).

    Scoped by optional X-Hotkey (no signature required for list view policy).
    Without X-Hotkey returns empty items (fail-closed identity scope).
    """

    hotkey = _header_hotkey(request)
    items = await list_leases(
        session,
        hotkey=hotkey,
        offer_id=offer_id,
        status=status_filter,
    )
    return {"items": [lease_to_public(x) for x in items]}


@public_route(tags=["marketplace"])
@router.get("/v1/leases/{lease_id}")
async def leases_get(
    lease_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Lease detail (status, offer_id, price, times) — VAL-MKT-016."""

    lease = await get_lease(session, lease_id)
    if lease is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "lease_not_found", "message": "lease not found"},
        )
    return lease_to_public(lease)


@public_route(tags=["marketplace"])
@router.post("/v1/leases/{lease_id}/terminate", status_code=status.HTTP_200_OK)
async def leases_terminate(
    lease_id: str,
    identity: RequireMiner,
    session: DbSession,
    body: TerminateLeaseRequest | None = None,
) -> dict[str, Any]:
    """Renter/provider terminate lease; pod stops; capacity free (VAL-MKT-015/021)."""

    req = body if body is not None else TerminateLeaseRequest()
    try:
        lease = await terminate_lease(
            session,
            hotkey=identity.hotkey,
            lease_id=lease_id,
            reason=req.reason,
            allow_provider=True,
        )
        pod = await get_pod_by_lease(session, lease.id)
    except LeaseError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    payload: dict[str, Any] = {"lease": lease_to_public(lease)}
    if pod is not None:
        payload["pod"] = pod_to_public(pod)
    return payload


@public_route(tags=["marketplace"])
@router.get("/v1/pods/{pod_id}")
async def pods_get(
    pod_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Pod detail with node binding and endpoints (VAL-MKT-017/019)."""

    pod = await get_pod(session, pod_id)
    if pod is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "pod_not_found", "message": "pod not found"},
        )
    return pod_to_public(pod)


@public_route(tags=["jobs"])
@router.post("/v1/jobs", status_code=status.HTTP_200_OK)
async def jobs_create(
    body: JobAdmitRequest,
    identity: RequireMiner,
    session: DbSession,
    request: Request,
) -> dict[str, Any]:
    """Admit HyperJob with static gates + idempotency (VAL-JOB-001..005)."""

    caps = _job_admit_kwargs(request)
    try:
        job, created = await admit_job(
            session,
            hotkey=identity.hotkey,
            image_digest=body.image_digest,
            entrypoint=body.entrypoint,
            world_size=body.world_size,
            nnodes=body.nnodes,
            nproc_per_node=body.nproc_per_node,
            resource=body.resource,
            timeout_s=body.timeout_s,
            client_request_id=body.client_request_id,
            backend=body.backend,
            fabric=body.fabric,
            tee=body.tee,
            env=body.env,
            placement_policy=body.placement_policy,
            lease_id=body.lease_id,
            pod_id=body.pod_id,
            **caps,
        )
    except JobError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    payload = job_to_public(job)
    payload["created"] = created
    return payload


@public_route(tags=["jobs"])
@router.get("/v1/jobs")
async def list_jobs(
    session: DbSession,
    request: Request,
    status_filter: str | None = Query(default=None, alias="status"),
) -> dict[str, Any]:
    """List jobs scoped to submitter hotkey (VAL-JOB-001/012 list surface).

    Without X-Hotkey returns empty items (fail-closed identity scope).
    """

    hotkey = _header_hotkey(request)
    items = await list_jobs_for_hotkey(
        session,
        hotkey=hotkey,
        status=status_filter,
    )
    return {"items": [job_to_public(j) for j in items]}


@public_route(tags=["jobs"])
@router.get("/v1/jobs/{job_id}")
async def jobs_get(
    job_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """Job detail: status, placement, proofs summary, no secrets (VAL-JOB-010/026)."""

    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "job_not_found", "message": "job not found"},
        )
    placement = await get_placement(session, job_id)
    attempt = await get_latest_attempt(session, job_id)
    proofs = await get_proofs_for_attempt(session, attempt.id) if attempt is not None else []
    fabric = await get_fabric_report(session, job_id)
    return job_detail_public(
        job,
        placement=placement,
        attempt=attempt,
        proofs=proofs,
        fabric_report=fabric,
    )


@public_route(tags=["jobs"])
@router.post("/v1/jobs/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def jobs_cancel(
    job_id: str,
    identity: RequireMiner,
    session: DbSession,
    body: JobCancelRequest | None = None,
) -> dict[str, Any]:
    """Cancel non-terminal job as submitter (VAL-JOB-007)."""

    _ = body  # reason reserved for audit trail later
    try:
        job = await cancel_job(session, job_id=job_id, hotkey=identity.hotkey)
    except JobError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return job_to_public(job)


@public_route(tags=["jobs"])
@router.get("/v1/jobs/{job_id}/attempts/{attempt_no}")
async def jobs_attempt_get(
    job_id: str,
    attempt_no: int,
    session: DbSession,
) -> dict[str, Any]:
    """Attempt detail + metrics digests (VAL-JOB-011)."""

    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "job_not_found", "message": "job not found"},
        )
    attempt = await get_attempt(session, job_id, attempt_no)
    if attempt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "attempt_not_found", "message": "attempt not found"},
        )
    return attempt_to_public(attempt)


@public_route(tags=["jobs"])
@router.get("/v1/jobs/{job_id}/fabric-report")
async def jobs_fabric_report(
    job_id: str,
    session: DbSession,
) -> dict[str, Any]:
    """FabricReport view for multi-node sim jobs (VAL-JOB-021)."""

    job = await get_job(session, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "job_not_found", "message": "job not found"},
        )
    report = await get_fabric_report(session, job_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "fabric_report_not_ready",
                "message": "fabric report not available yet (collect not finished)",
            },
        )
    return report.to_dict()


@public_route(tags=["jobs"])
@router.post("/v1/jobs/{job_id}/results", status_code=status.HTTP_200_OK)
async def jobs_post_results(
    job_id: str,
    identity: RequireMiner,
    session: DbSession,
    body: JobResultsRequest,
    request: Request,
) -> dict[str, Any]:
    """Provider/worker result envelope; attempt-keyed idempotent (VAL-JOB-009)."""

    # Merge integrity_codes into metrics so lifecycle + scoring see cheat inject.
    metrics = dict(body.metrics or {})
    if body.integrity_codes:
        existing_codes = metrics.get("integrity_codes")
        merged: list[str] = []
        if isinstance(existing_codes, list):
            merged.extend(str(c) for c in existing_codes)
        for code in body.integrity_codes:
            if str(code) not in merged:
                merged.append(str(code))
        metrics["integrity_codes"] = merged
        metrics["reason_codes"] = list(
            dict.fromkeys(
                [
                    *(
                        str(c)
                        for c in (metrics.get("reason_codes") or [])
                        if isinstance(metrics.get("reason_codes"), list)
                    ),
                    *merged,
                ]
            )
        )
        if any(
            c
            in {
                "rank_desync",
                "image_mutation",
                "image_compose_mutation",
                "inventory_spoof",
                "integrity_fail",
                "attestation_fail",
            }
            for c in merged
        ):
            metrics["integrity_fail"] = True
            if "rank_desync" in merged:
                metrics["rank_desync"] = True
    # Prefer explicit failure_code; else first integrity code when present.
    failure_code = body.failure_code
    if not failure_code and body.integrity_codes:
        failure_code = str(body.integrity_codes[0])

    try:
        attempt, created = await post_job_results(
            session,
            job_id=job_id,
            attempt_no=body.attempt_no,
            status=body.status,
            metrics=metrics,
            fabric_report_digest=body.fabric_report_digest,
            output_digest=body.output_digest,
            proof_tier=body.proof_tier,
            verified=body.verified and not bool(metrics.get("integrity_fail")),
            verify_mode=body.verify_mode,
            failure_code=failure_code,
            actor_hotkey=identity.hotkey,
        )
    except JobError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    # Optional offline TEE upgrade path on results post (VAL-TEE-006..009/012).
    if body.quote_b64:
        from hypercluster.attest.report_data import build_report_data
        from hypercluster.domain.job_lifecycle import get_proofs_for_attempt
        from hypercluster.domain.jobs import get_job as _get_job
        from hypercluster.domain.scoring_tee import persist_score_for_attempt
        from hypercluster.domain.tee_proofs import verify_and_build_proof

        job = await _get_job(session, job_id)
        if job is not None:
            report_data: bytes
            if body.report_data_hex:
                report_data = bytes.fromhex(body.report_data_hex)
            else:
                nonce = body.tee_nonce or "results-default-nonce"
                report_data = build_report_data(
                    job_id=job.id,
                    image_digest=job.image_digest,
                    nonce=nonce,
                )
            # Drop any sim placeholder before attaching offline verified proof.
            existing = await get_proofs_for_attempt(session, attempt.id)
            for row in existing:
                await session.delete(row)
            await session.flush()
            mode = (
                body.verify_mode
                if body.verify_mode in {"offline_fixture", "live", "sim"}
                else "offline_fixture"
            )
            proof, _result = verify_and_build_proof(
                attempt_id=attempt.id,
                job=job,
                quote_b64=body.quote_b64,
                report_data_expected=report_data,
                gpu_evidence=body.gpu_evidence,
                mode=mode,
                expected_gpu_nonce=body.tee_nonce,
            )
            session.add(proof)
            metrics = body.metrics or {}
            efficiency = float(metrics.get("efficiency", 1.0) or 1.0)
            fabric_gate = float(metrics.get("fabric_gate", 1.0) or 1.0)
            hyper = getattr(request.app.state, "hyper_settings", None)
            await persist_score_for_attempt(
                session,
                attempt_id=attempt.id,
                hotkey=job.submitter_hotkey,
                correctness=1.0 if proof.verified else 0.0,
                efficiency=efficiency,
                fabric_gate=fabric_gate,
                proof=proof,
                tee_mode=job.tee_mode or "none",
                hyper=hyper,
            )
            await session.commit()

    payload = attempt_to_public(attempt)
    payload["created"] = created
    payload["attempt"] = attempt_to_public(attempt)
    return payload


@public_route(tags=["scoring"])
@router.get("/v1/leaderboard")
async def leaderboard(
    session: DbSession,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    """Aggregated composite scores ordered by mass desc (VAL-SCORE-018/029).

    First-visit vacant DB returns HTTP 200 with empty ``items`` — never 5xx,
    never NaN ranks, never a fabricated registry-only leaderboard.
    """

    from hypercluster.domain.aggregation import build_leaderboard
    from hypercluster.settings import get_hyper_settings

    items = await build_leaderboard(
        session,
        hyper=get_hyper_settings(),
        limit=limit,
    )
    return {
        "items": items,
        "count": len(items),
        "empty": len(items) == 0,
    }


@public_route(tags=["scoring"])
@router.get("/v1/scores/{hotkey}")
async def scores_for_hotkey(
    hotkey: str,
    session: DbSession,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    """Per-hotkey score history with four factors + role (VAL-SCORE-001/008/026).

    Even when composite is 0, each row still exposes correctness, efficiency,
    fabric_gate, and tee_bonus for forensic debugging. Rows bind hotkey+role
    (demand|supply|joint). Absent hotkey → empty items (vacant-safe).
    """

    from hypercluster.domain.scoring import list_scores_for_hotkey, score_row_to_public

    rows = await list_scores_for_hotkey(session, hotkey, limit=limit)
    return {
        "hotkey": hotkey,
        "items": [score_row_to_public(row) for row in rows],
        "count": len(rows),
    }


@public_route(tags=["scoring"])
@router.get("/v1/weight-preview")
async def weight_preview(
    request: Request,
) -> dict[str, Any]:
    """Pending/latest raw weight map (VAL-SCORE-009/010/028; architecture §4.3).

    Returns the monochronic snapshot map when one exists (pending or acked),
    else the live aggregation window. Vacant → ``weights: {}`` burn-safe.
    Shape aligns with get_weights / push payload weights (VAL-SCORE-016).
    """

    from hypercluster.weights import weight_preview_payload

    database = getattr(request.app.state, "database", None)
    hyper = getattr(request.app.state, "hyper_settings", None)
    return await weight_preview_payload(database=database, hyper=hyper)


@public_route(tags=["sim"])
@router.post("/v1/sim/idle-reclaim", status_code=status.HTTP_200_OK)
async def sim_idle_reclaim(
    session: DbSession,
    body: IdleReclaimRequest | None = None,
) -> dict[str, Any]:
    """Run an idle-only reclaim tick under local sim (VAL-CROSS-011).

    Optionally ages every node heartbeat into the past so the sweep becomes
    deterministic without wall-clock waits. Active leases remain protected:
    ``run_idle_reclaim_sweep`` never offline-kills rented tenant capacity.
    """

    from datetime import timedelta

    from sqlalchemy import select

    from hypercluster.db.models import Lease, Node, utc_now
    from hypercluster.domain.leases import (
        ACTIVE_LEASE_STATUSES,
        get_pod_by_lease,
        run_idle_reclaim_sweep,
    )

    req = body if body is not None else IdleReclaimRequest()
    now = utc_now()
    aged = 0
    if req.age_heartbeats_seconds is not None and req.age_heartbeats_seconds > 0:
        delta = timedelta(seconds=int(req.age_heartbeats_seconds))
        result = await session.execute(select(Node))
        for node in result.scalars().all():
            node.last_heartbeat = now - delta
            node.updated_at = now
            aged += 1
        if aged:
            await session.commit()

    # Snapshot protected nodes (active leases) before sweep for evidence.
    lease_rows = await session.execute(
        select(Lease).where(Lease.status.in_(tuple(ACTIVE_LEASE_STATUSES)))
    )
    protected: list[str] = []
    for lease in lease_rows.scalars().all():
        pod = await get_pod_by_lease(session, lease.id)
        if pod is not None:
            protected.extend(pod.node_ids())

    offline_marked = await run_idle_reclaim_sweep(
        session,
        liveness_seconds=int(req.liveness_seconds),
    )
    return {
        "ok": True,
        "age_heartbeats_seconds": req.age_heartbeats_seconds,
        "liveness_seconds": req.liveness_seconds,
        "nodes_aged": aged,
        "offline_marked": offline_marked,
        "protected_node_ids": sorted(set(protected)),
    }


@public_route(tags=["sim"])
@router.post("/v1/sim/drain", status_code=status.HTTP_200_OK)
async def sim_set_drain(
    request: Request,
    body: DrainModeRequest | None = None,
) -> dict[str, Any]:
    """Toggle drain mode for READY 503 semantics (VAL-CROSS-026).

    In-flight jobs keep advancing via the combined worker; new POSTs are
    refused by the Base SDK `runtime_not_ready` middleware while ready=false.
    """

    from hypercluster.app import is_draining, set_draining

    req = body if body is not None else DrainModeRequest(draining=True)
    draining = set_draining(request.app, bool(req.draining))
    return {
        "ok": True,
        "draining": draining,
        "was_draining": is_draining(request.app) if draining else False,
    }


@public_route(tags=["sim"])
@router.get("/v1/sim/drain")
async def sim_get_drain(request: Request) -> dict[str, Any]:
    """Report drain flag (diagnostic surface for cross scenarios)."""

    from hypercluster.app import is_draining

    return {"ok": True, "draining": is_draining(request.app)}


__all__ = [
    "jobs_attempt_get",
    "jobs_cancel",
    "jobs_create",
    "jobs_fabric_report",
    "jobs_get",
    "jobs_post_results",
    "leaderboard",
    "scores_for_hotkey",
    "weight_preview",
    "leases_get",
    "leases_list",
    "leases_terminate",
    "list_jobs",
    "nodes_fabric_report",
    "nodes_fabric_scan",
    "nodes_get",
    "nodes_heartbeat",
    "nodes_list",
    "nodes_probe_gpu",
    "nodes_probe_gpu_by_id",
    "nodes_probe_gpu_latest",
    "nodes_probe_gpu_list",
    "nodes_evidence_gpu_attach",
    "evidence_gpu_global",
    "nodes_register",
    "offers_create",
    "offers_get",
    "offers_list",
    "offers_rent",
    "offers_withdraw",
    "pods_get",
    "providers_heartbeat",
    "providers_list",
    "providers_me",
    "providers_register",
    "router",
    "sim_get_drain",
    "sim_idle_reclaim",
    "sim_set_drain",
]
