"""GPU probe orchestration, evidence persistence, and public DTOs (M9 API).

Runs :func:`run_gpu_probe` against resolved SSH transport (FakeSsh in CI),
persists :class:`GpuHostEvidence` rows, merges inventory, never stores PEM,
never ``set_weights``. Formula unchanged.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import GpuHostEvidenceRow, Node, Provider, isoformat_utc, utc_now
from hypercluster.domain.nodes import get_node
from hypercluster.domain.providers import get_provider_by_hotkey
from hypercluster.probe.inventory_merge import (
    GPU_PROBE_STATUS_NONE,
    apply_probe_to_node_fields,
    merge_probe_into_inventory,
    probe_status_label,
)
from hypercluster.probe.keys import (
    KeyMaterialError,
    KeyRef,
    reject_body_private_key_fields,
)
from hypercluster.probe.pipeline import (
    GpuProbeConfig,
    GpuProbeContext,
    occupied_uuid_index,
    run_gpu_probe,
)
from hypercluster.probe.resolve import (
    TransportConfigError,
    resolve_fake_fixture,
    resolve_ssh_transport,
)
from hypercluster.probe.types import (
    CheckResult,
    ClaimedInventory,
    GpuHostEvidence,
    MeasuredGpu,
    MeasuredInventory,
    ProbeDigests,
    ProbeMode,
    canonical_json,
)
from hypercluster.settings import HyperSettings, get_hyper_settings

ProbeModeName = Literal["full", "quick"]


class GpuProbeError(Exception):
    """Domain error for GPU probe / evidence operations."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _parse_started(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def evidence_to_summary(row: GpuHostEvidenceRow | GpuHostEvidence) -> dict[str, Any]:
    """Compact public DTO for list/latest (digests + uuid list)."""

    if isinstance(row, GpuHostEvidence):
        return {
            "evidence_id": row.id,
            "id": row.id,
            "node_id": row.node_id,
            "status": row.status,
            "mode": row.mode,
            "transport": row.transport,
            "started_at": row.started_at,
            "finished_at": row.finished_at,
            "failure_code": row.failure_code,
            "checks_passed": sum(1 for c in row.checks if c.passed),
            "checks_failed": sum(1 for c in row.checks if c.fatal and not c.passed),
            "measured_gpu_count": row.measured.gpu_count,
            "gpu_uuids": row.measured.uuid_set(),
            "digests": row.digests.model_dump(mode="json"),
            "key_fingerprint": row.key_fingerprint,
        }

    try:
        digests = json.loads(row.digests_json or "{}")
    except (TypeError, ValueError):
        digests = {}
    try:
        uuids = json.loads(row.gpu_uuids_json or "[]")
    except (TypeError, ValueError):
        uuids = []
    return {
        "evidence_id": row.id,
        "id": row.id,
        "node_id": row.node_id,
        "status": row.status,
        "mode": row.mode,
        "transport": row.transport,
        "started_at": isoformat_utc(row.started_at),
        "finished_at": isoformat_utc(row.finished_at),
        "failure_code": row.failure_code,
        "checks_passed": int(row.checks_passed or 0),
        "checks_failed": int(row.checks_failed or 0),
        "measured_gpu_count": int(row.measured_gpu_count or 0),
        "gpu_uuids": uuids if isinstance(uuids, list) else [],
        "digests": digests if isinstance(digests, dict) else {},
        "key_fingerprint": row.key_fingerprint,
        "provider_hotkey": row.provider_hotkey,
    }


def evidence_to_public(row: GpuHostEvidenceRow | GpuHostEvidence) -> dict[str, Any]:
    """Full public evidence document (checks + redacted raw)."""

    if isinstance(row, GpuHostEvidence):
        pub = row.to_public()
        pub["evidence_id"] = row.id
        pub["checks_passed"] = sum(1 for c in row.checks if c.passed)
        pub["checks_failed"] = sum(1 for c in row.checks if c.fatal and not c.passed)
        pub["measured_gpu_count"] = row.measured.gpu_count
        pub["gpu_uuids"] = row.measured.uuid_set()
        return pub

    try:
        payload = json.loads(row.evidence_json or "{}")
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    # Prefer stored sealed document; re-hydrate summary fields.
    summary = evidence_to_summary(row)
    body = {**payload, **{k: v for k, v in summary.items() if v is not None}}
    body["evidence_id"] = row.id
    body["id"] = row.id
    # Strip any accidental PEM if closed-loop attach stored bad material.
    from hypercluster.probe.redact import redact_mapping

    return redact_mapping(body)


def row_from_evidence(
    evidence: GpuHostEvidence,
    *,
    provider_hotkey: str | None,
    source: str = "probe",
) -> GpuHostEvidenceRow:
    public = evidence.to_public()
    started = _parse_started(evidence.started_at) or utc_now()
    finished = _parse_started(evidence.finished_at)
    return GpuHostEvidenceRow(
        id=evidence.id,
        node_id=evidence.node_id,
        provider_hotkey=provider_hotkey or evidence.provider_hotkey,
        status=evidence.status,
        mode=evidence.mode,
        transport=evidence.transport,
        failure_code=evidence.failure_code,
        key_fingerprint=evidence.key_fingerprint,
        key_ref_json=None,
        started_at=started,
        finished_at=finished,
        checks_passed=sum(1 for c in evidence.checks if c.passed),
        checks_failed=sum(1 for c in evidence.checks if c.fatal and not c.passed),
        measured_gpu_count=int(evidence.measured.gpu_count or 0),
        gpu_uuids_json=json.dumps(evidence.measured.uuid_set()),
        digests_json=json.dumps(evidence.digests.model_dump(mode="json")),
        evidence_json=json.dumps(public),
        source=source,
        created_at=utc_now(),
    )


async def _assert_node_owner(
    session: AsyncSession,
    *,
    node_id: str,
    hotkey: str,
) -> tuple[Node, Provider]:
    node = await get_node(session, node_id)
    if node is None:
        raise GpuProbeError("node_not_found", "node not found", status_code=404)
    provider = await get_provider_by_hotkey(session, hotkey)
    if provider is None or provider.id != node.provider_id:
        raise GpuProbeError(
            "node_not_owned",
            "node belongs to another provider",
            status_code=403,
        )
    return node, provider


async def _occupied_uuids(
    session: AsyncSession,
    *,
    exclude_node_id: str | None,
) -> set[str]:
    result = await session.execute(select(Node))
    pairs: list[tuple[str, list[str]]] = []
    for node in result.scalars().all():
        uuids: list[str] = []
        if node.inventory_json:
            try:
                inv = json.loads(node.inventory_json)
            except (TypeError, ValueError):
                inv = {}
            if isinstance(inv, dict):
                raw = inv.get("gpu_uuids") or []
                if isinstance(raw, list):
                    uuids = [str(u) for u in raw if u]
        pairs.append((node.id, uuids))
    return occupied_uuid_index(pairs, exclude_node_id=exclude_node_id)


async def _prior_verified_uuids(session: AsyncSession, node_id: str) -> set[str] | None:
    result = await session.execute(
        select(GpuHostEvidenceRow)
        .where(
            GpuHostEvidenceRow.node_id == node_id,
            GpuHostEvidenceRow.status == "passed",
        )
        .order_by(GpuHostEvidenceRow.finished_at.desc())
    )
    row = result.scalars().first()
    if row is None:
        return None
    try:
        uuids = json.loads(row.gpu_uuids_json or "[]")
    except (TypeError, ValueError):
        return None
    if not isinstance(uuids, list) or not uuids:
        return None
    return {str(u) for u in uuids if u}


async def persist_evidence(
    session: AsyncSession,
    evidence: GpuHostEvidence,
    *,
    provider_hotkey: str | None,
    source: str = "probe",
    merge_inventory: bool = True,
) -> GpuHostEvidenceRow:
    """Insert evidence row and optionally merge into node inventory."""

    row = row_from_evidence(evidence, provider_hotkey=provider_hotkey, source=source)
    session.add(row)

    if merge_inventory and evidence.node_id:
        node = await get_node(session, evidence.node_id)
        if node is not None:
            patch = apply_probe_to_node_fields(
                inventory_json=node.inventory_json,
                evidence=evidence,
            )
            node.inventory_json = patch["inventory_json"]
            node.updated_at = utc_now()
            # Do not overwrite claim columns unless pass (ops retain measure).
            if evidence.status == "passed" and evidence.measured.gpu_count > 0:
                # Keep claim fields as authority for marketplace; store measured only in inventory.
                pass

    await session.commit()
    await session.refresh(row)
    return row


async def run_node_gpu_probe(
    session: AsyncSession,
    *,
    node_id: str,
    hotkey: str,
    mode: ProbeModeName = "full",
    timeout_s: int | None = None,
    key_ref: dict[str, str] | None = None,
    fixture_name: str | None = None,
    body: dict[str, Any] | None = None,
    settings: HyperSettings | None = None,
) -> tuple[GpuHostEvidence, GpuHostEvidenceRow]:
    """Owner-signed probe start path (VAL-GPU-001/002)."""

    try:
        reject_body_private_key_fields(body)
    except KeyMaterialError as exc:
        raise GpuProbeError(exc.code, exc.message, status_code=422) from exc

    node, provider = await _assert_node_owner(session, node_id=node_id, hotkey=hotkey)
    cfg = settings if settings is not None else get_hyper_settings()

    mode_norm: ProbeMode = "quick" if str(mode).lower() == "quick" else "full"
    claimed = ClaimedInventory(
        gpu_model=node.gpu_model or "unknown",
        gpu_count=int(node.gpu_count or 0),
    )

    key_ref_obj: KeyRef | None = None
    if key_ref:
        kind = str(key_ref.get("kind") or "file")
        name = str(key_ref.get("name") or "")
        if not name:
            raise GpuProbeError("invalid_key_ref", "key_ref.name is required", status_code=422)
        if kind not in {"env", "file", "agent"}:
            raise GpuProbeError(
                "invalid_key_ref",
                "key_ref.kind must be env|file|agent",
                status_code=422,
            )
        key_ref_obj = KeyRef(kind=kind, name=name)  # type: ignore[arg-type]

    host: str | None = None
    port = 22
    if node.ssh_endpoint:
        from hypercluster.probe.ssh_exec import parse_ssh_endpoint

        try:
            host, port = parse_ssh_endpoint(node.ssh_endpoint)
        except Exception:  # noqa: BLE001
            host = node.ssh_endpoint.split(":")[0] or None

    # For FakeSsh, claim/override may come from fixture bank so pass_all matches model.
    fixture_claimed = claimed
    prior = await _prior_verified_uuids(session, node_id)
    occupied = await _occupied_uuids(session, exclude_node_id=node_id)
    fx_name = fixture_name
    if (cfg.ssh_transport or "").lower() in {"fake", "fakessh", "sim"}:
        try:
            fx = resolve_fake_fixture(cfg, fixture_name=fx_name)
            fixture_claimed = fx.claimed
            # Respect fixture prior/occupied when present (fingerprint / uniqueness cases).
            if fx.prior_verified_uuids is not None and prior is None:
                prior = set(fx.prior_verified_uuids)
            if fx.occupied_uuids:
                occupied |= set(fx.occupied_uuids)
            if fx_name is None:
                fx_name = fx.name
        except TransportConfigError as exc:
            raise GpuProbeError(exc.code, exc.message, status_code=exc.status_code) from exc
    else:
        # Real path: keep node claim (registry source of truth).
        fixture_claimed = claimed

    try:
        transport = resolve_ssh_transport(
            cfg,
            fixture_name=fx_name,
            host=host,
            port=port,
            username=cfg.ssh_username or "root",
            key_ref=key_ref_obj,
        )
    except TransportConfigError as exc:
        raise GpuProbeError(exc.code, exc.message, status_code=exc.status_code) from exc

    probe_cfg = GpuProbeConfig(
        mode=mode_norm,
        max_gpu_count=int(cfg.max_gpu_count or 14),
        require_docker_runtime=bool(cfg.require_docker_runtime),
        skip_microbench=mode_norm == "quick",
    )
    # timeout_s reserved for real RealSsh wall budget; FakeSsh uses staged outcomes.
    _ = timeout_s

    key_fp: str | None = None
    if key_ref_obj is not None:
        try:
            from hypercluster.probe.keys import resolve_key_ref

            resolved = resolve_key_ref(key_ref_obj)
            key_fp = resolved.fingerprint
        except KeyMaterialError:
            key_fp = None

    ctx = GpuProbeContext(
        node_id=node.id,
        provider_hotkey=provider.hotkey,
        ssh_endpoint=node.ssh_endpoint,
        claimed=fixture_claimed if (cfg.ssh_transport or "").lower() in {
            "fake", "fakessh", "sim"
        } else ClaimedInventory(
            gpu_model=node.gpu_model or fixture_claimed.gpu_model,
            gpu_count=int(node.gpu_count or fixture_claimed.gpu_count),
        ),
        key_fingerprint=key_fp,
        occupied_uuids=occupied,
        prior_verified_uuids=prior,
    )

    # For FakeSsh + node claim mismatch (e.g. node says H100, fixture V100): use fixture claim
    # only when node model is the pass_all default family OR force claim from fixture in fake mode.
    if (cfg.ssh_transport or "").lower() in {"fake", "fakessh", "sim"}:
        # Align claim so CAT pass_all with owner-specified model that matches family aliases works.
        # When operator registered 1V100.6V for pass_all, we're good. If they used a different
        # model, still run; pipeline may fail model_match (intended).
        ctx.claimed = ClaimedInventory(
            gpu_model=node.gpu_model or fixture_claimed.gpu_model,
            gpu_count=int(node.gpu_count or fixture_claimed.gpu_count),
        )
        # Prefer fixture claim when it can normalize to same family as node, else keep node.
        # For tests we register 1V100.6V matching pass_all.

    evidence = run_gpu_probe(transport, ctx, config=probe_cfg)
    row = await persist_evidence(
        session,
        evidence,
        provider_hotkey=provider.hotkey,
        source="probe",
        merge_inventory=True,
    )
    return evidence, row


async def list_node_evidence(
    session: AsyncSession,
    node_id: str,
    *,
    limit: int = 50,
) -> list[GpuHostEvidenceRow]:
    """Newest-first evidence list; empty for unknown nodes (VAL-GPU-004)."""

    result = await session.execute(
        select(GpuHostEvidenceRow)
        .where(GpuHostEvidenceRow.node_id == node_id)
        .order_by(
            GpuHostEvidenceRow.created_at.desc(),
            GpuHostEvidenceRow.finished_at.desc(),
        )
        .limit(max(1, min(int(limit), 200)))
    )
    return list(result.scalars().all())


async def get_latest_node_evidence(
    session: AsyncSession,
    node_id: str,
) -> GpuHostEvidenceRow | None:
    rows = await list_node_evidence(session, node_id, limit=1)
    return rows[0] if rows else None


async def get_node_evidence(
    session: AsyncSession,
    node_id: str,
    evidence_id: str,
) -> GpuHostEvidenceRow | None:
    result = await session.execute(
        select(GpuHostEvidenceRow).where(
            GpuHostEvidenceRow.id == evidence_id,
            GpuHostEvidenceRow.node_id == node_id,
        )
    )
    return result.scalar_one_or_none()


async def get_evidence_global(
    session: AsyncSession,
    evidence_id: str,
) -> GpuHostEvidenceRow | None:
    result = await session.execute(
        select(GpuHostEvidenceRow).where(GpuHostEvidenceRow.id == evidence_id)
    )
    return result.scalar_one_or_none()


def compute_attach_digest(evidence_payload: dict[str, Any]) -> str:
    """Canonical digest covering signed external evidence payload."""

    payload = {
        "status": evidence_payload.get("status"),
        "claimed": evidence_payload.get("claimed"),
        "measured": evidence_payload.get("measured"),
        "checks": evidence_payload.get("checks"),
        "mode": evidence_payload.get("mode"),
        "transport": evidence_payload.get("transport"),
    }
    digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return f"sha256:{digest}"


async def attach_external_evidence(
    session: AsyncSession,
    *,
    node_id: str,
    hotkey: str,
    evidence_payload: dict[str, Any],
    claimed_digest: str | None,
    body: dict[str, Any] | None = None,
) -> tuple[GpuHostEvidence, GpuHostEvidenceRow]:
    """Ops attach of externally gathered evidence (VAL-GPU-006).

    Rejects private-key smuggling and digest mismatch.
    """

    try:
        reject_body_private_key_fields(body or evidence_payload)
    except KeyMaterialError as exc:
        raise GpuProbeError(exc.code, exc.message, status_code=422) from exc

    node, provider = await _assert_node_owner(session, node_id=node_id, hotkey=hotkey)

    if not isinstance(evidence_payload, dict):
        raise GpuProbeError(
            "invalid_evidence",
            "evidence body must be an object",
            status_code=422,
        )

    expected = compute_attach_digest(evidence_payload)
    claimed = (claimed_digest or "").strip()
    raw_digests = evidence_payload.get("digests")
    digests_in = raw_digests if isinstance(raw_digests, dict) else {}
    # Prefer explicit claimed_digest; fall back to evidence.digests.evidence_sha256.
    if not claimed:
        claimed = str((digests_in or {}).get("evidence_sha256") or "").strip()
    if not claimed:
        raise GpuProbeError(
            "digest_required",
            "claimed_digest (or digests.evidence_sha256) is required",
            status_code=422,
        )
    if claimed != expected:
        raise GpuProbeError(
            "digest_mismatch",
            "claimed digest does not match evidence payload",
            status_code=422,
        )

    status_raw = str(evidence_payload.get("status") or "failed")
    if status_raw not in {"passed", "failed", "error"}:
        status_raw = "failed"

    claimed_inv_raw = evidence_payload.get("claimed") or {}
    if not isinstance(claimed_inv_raw, dict):
        claimed_inv_raw = {}
    claimed_inv = ClaimedInventory(
        gpu_model=str(claimed_inv_raw.get("gpu_model") or node.gpu_model or "unknown"),
        gpu_count=int(claimed_inv_raw.get("gpu_count") or node.gpu_count or 0),
    )
    measured_raw = evidence_payload.get("measured") or {}
    if not isinstance(measured_raw, dict):
        measured_raw = {}
    gpus: list[MeasuredGpu] = []
    for g in measured_raw.get("gpus") or []:
        if not isinstance(g, dict):
            continue
        gpus.append(
            MeasuredGpu(
                name=str(g.get("name") or "unknown"),
                uuid=str(g.get("uuid") or ""),
                memory_total_mb=g.get("memory_total_mb"),
                driver_version=g.get("driver_version"),
                power_limit_w=g.get("power_limit_w"),
                power_default_w=g.get("power_default_w"),
                util_gpu=g.get("util_gpu"),
                util_mem=g.get("util_mem"),
                clocks_sm_mhz=g.get("clocks_sm_mhz"),
            )
        )
    docker_meta: dict[str, Any] = {}
    raw_docker = measured_raw.get("docker")
    if isinstance(raw_docker, dict):
        docker_meta = dict(raw_docker)
    measured = MeasuredInventory(
        gpu_count=int(measured_raw.get("gpu_count") or len(gpus)),
        gpus=gpus,
        cuda_runtime_hint=measured_raw.get("cuda_runtime_hint"),
        docker=docker_meta,
    )
    checks: list[CheckResult] = []
    for c in evidence_payload.get("checks") or []:
        if not isinstance(c, dict) or not c.get("id"):
            continue
        details: dict[str, Any] = {}
        raw_details = c.get("details")
        if isinstance(raw_details, dict):
            details = dict(raw_details)
        checks.append(
            CheckResult(
                id=str(c["id"]),
                fatal=bool(c.get("fatal", True)),
                passed=bool(c.get("passed", False)),
                halt=bool(c.get("halt", False)),
                message=str(c.get("message") or "ok"),
                duration_ms=int(c.get("duration_ms") or 0),
                details=details,
            )
        )

    mode_val = str(evidence_payload.get("mode") or "full")
    if mode_val not in {"full", "quick"}:
        mode_val = "full"
    transport_val = str(evidence_payload.get("transport") or "fake")
    if transport_val not in {"real", "fake"}:
        transport_val = "fake"
    raw_redacted: dict[str, Any] = {}
    raw_rr = evidence_payload.get("raw_redacted")
    if isinstance(raw_rr, dict):
        raw_redacted = dict(raw_rr)
    evidence = GpuHostEvidence(
        id=str(evidence_payload.get("id") or uuid.uuid4()),
        node_id=node.id,
        provider_hotkey=provider.hotkey,
        ssh_endpoint=node.ssh_endpoint,
        status=status_raw,  # type: ignore[arg-type]
        mode=mode_val,  # type: ignore[arg-type]
        transport=transport_val,  # type: ignore[arg-type]
        claimed=claimed_inv,
        measured=measured,
        checks=checks,
        digests=ProbeDigests(
            inventory_sha256=str((digests_in or {}).get("inventory_sha256") or expected),
            microbench_digest=(digests_in or {}).get("microbench_digest"),
            evidence_sha256=expected,
        ),
        failure_code=evidence_payload.get("failure_code"),
        key_fingerprint=evidence_payload.get("key_fingerprint"),
        raw_redacted=raw_redacted,
    )
    if evidence.finished_at is None:
        evidence.finished_at = evidence.started_at
    # Seal recomputes digests; external attach preserves claimed digest.
    evidence.seal()
    evidence.digests.evidence_sha256 = expected

    row = await persist_evidence(
        session,
        evidence,
        provider_hotkey=provider.hotkey,
        source="external",
        merge_inventory=True,
    )
    return evidence, row


def node_gpu_probe_status(node: Node) -> str:
    """Extract probe status from node inventory without inventing verified."""

    if not node.inventory_json:
        return GPU_PROBE_STATUS_NONE
    try:
        inv = json.loads(node.inventory_json)
    except (TypeError, ValueError):
        return GPU_PROBE_STATUS_NONE
    if not isinstance(inv, dict):
        return GPU_PROBE_STATUS_NONE
    status = inv.get("gpu_probe_status")
    if not status:
        return GPU_PROBE_STATUS_NONE
    return str(status)


def soft_heartbeat_probe_meta(
    nodes: list[Node],
    *,
    require_live_evidence: bool,
    mode: str = "soft",
) -> dict[str, Any]:
    """Advisory metadata for heartbeat under require-evidence flags (VAL-GPU-011).

    Soft mode never raises: returns warning flags for unverified nodes.
    Fail-closed mode may set ``block_status=409`` when all targeted nodes are
    unverified (caller decides whether to 409).
    """

    if not require_live_evidence:
        return {"require_live_evidence": False}

    unverified: list[str] = []
    for node in nodes:
        status = node_gpu_probe_status(node)
        if status != "verified":
            unverified.append(node.id)

    meta: dict[str, Any] = {
        "require_live_evidence": True,
        "require_live_evidence_mode": mode,
        "gpu_probe_warning": bool(unverified),
        "unverified_node_ids": unverified,
    }
    if mode in {"fail_closed", "hard", "strict"} and unverified:
        meta["block_status"] = 409
        meta["code"] = "gpu_probe_unverified"
        meta["message"] = "live evidence required; node not gpu_probe_status=verified"
    return meta


__all__ = [
    "GpuProbeError",
    "attach_external_evidence",
    "compute_attach_digest",
    "evidence_to_public",
    "evidence_to_summary",
    "get_evidence_global",
    "get_latest_node_evidence",
    "get_node_evidence",
    "list_node_evidence",
    "node_gpu_probe_status",
    "persist_evidence",
    "row_from_evidence",
    "run_node_gpu_probe",
    "soft_heartbeat_probe_meta",
    "merge_probe_into_inventory",
    "probe_status_label",
]
