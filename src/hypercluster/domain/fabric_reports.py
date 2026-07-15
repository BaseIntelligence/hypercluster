"""Persist node-level FabricReports and fabric-scan operations (VAL-FAB-001/018)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hypercluster.db.models import FabricReportRow, Node, utc_now
from hypercluster.domain.nodes import _extract_ib_flags, get_node
from hypercluster.fabric.discovery import (
    FabricReport,
    build_fabric_report,
    synthetic_ib_devices,
    synthetic_nvlink_topo_matrix,
    validate_accepted_report,
)
from hypercluster.sim.inventory import seed_sim_inventory


class FabricReportError(Exception):
    """Domain error for fabric report / fabric-scan operations."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _row_to_report(row: FabricReportRow) -> FabricReport:
    try:
        raw = json.loads(row.raw_json) if row.raw_json else {}
    except (TypeError, ValueError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}

    ib_devices = raw.get("ib_devices")
    if not isinstance(ib_devices, list):
        try:
            ib_devices = json.loads(row.ib_devices_json or "[]")
        except (TypeError, ValueError):
            ib_devices = []

    numa = raw.get("numa_map")
    if not isinstance(numa, dict):
        try:
            numa = json.loads(row.numa_map_json or "{}")
        except (TypeError, ValueError):
            numa = {}

    return FabricReport(
        node_id=row.node_id,
        collected_at=row.collected_at,
        ib_devices=ib_devices or [],
        ib_rate_gbps=row.ib_rate_gbps,
        gpu_gpu_topo_matrix=str(raw.get("gpu_gpu_topo_matrix") or ""),
        gpu_topo_sha256=row.gpu_topo_sha256 or "",
        numa_map={str(k): int(v) for k, v in (numa or {}).items()},
        nccl_version=row.nccl_version,
        eth_ifaces=list(raw.get("eth_ifaces") or []),
        report_digest=row.report_digest,
        gpu_count=int(raw.get("gpu_count") or 0),
        source=raw.get("source") or "scan",  # type: ignore[arg-type]
    )


def report_to_public(report: FabricReport | FabricReportRow) -> dict[str, Any]:
    if isinstance(report, FabricReport):
        return report.to_public()
    return _row_to_report(report).to_public()


async def get_latest_fabric_report(
    session: AsyncSession, node_id: str
) -> FabricReportRow | None:
    result = await session.execute(
        select(FabricReportRow)
        .where(FabricReportRow.node_id == node_id)
        .order_by(FabricReportRow.collected_at.desc())
    )
    return result.scalars().first()


async def list_fabric_reports(
    session: AsyncSession, node_id: str
) -> list[FabricReportRow]:
    result = await session.execute(
        select(FabricReportRow)
        .where(FabricReportRow.node_id == node_id)
        .order_by(FabricReportRow.collected_at.desc())
    )
    return list(result.scalars().all())


async def persist_fabric_report(
    session: AsyncSession,
    report: FabricReport,
    *,
    node: Node | None = None,
    update_node_inventory: bool = True,
) -> FabricReportRow:
    """Store an accepted FabricReport and denormalize node IB flags."""

    accepted = validate_accepted_report(report, gpu_count=report.gpu_count)
    public = accepted.to_public()
    row = FabricReportRow(
        id=str(uuid.uuid4()),
        node_id=accepted.node_id,
        collected_at=accepted.collected_at,
        ib_devices_json=json.dumps(public["ib_devices"]),
        ib_rate_gbps=accepted.ib_rate_gbps,
        gpu_topo_sha256=accepted.gpu_topo_sha256,
        numa_map_json=json.dumps(accepted.numa_map),
        nccl_version=accepted.nccl_version,
        report_digest=accepted.report_digest,
        raw_json=json.dumps(public),
    )
    session.add(row)

    if update_node_inventory:
        target = node
        if target is None:
            target = await get_node(session, accepted.node_id)
        if target is not None:
            inventory = {
                "fabric_report": public,
                "ib_devices": public["ib_devices"],
                "ib_rate_gbps": accepted.ib_rate_gbps,
                "gpu_topo_sha256": accepted.gpu_topo_sha256,
                "report_digest": accepted.report_digest,
                "has_ib": bool(accepted.ib_devices),
            }
            has_ib, rate = _extract_ib_flags(inventory)
            target.inventory_json = json.dumps(inventory)
            target.has_ib = 1 if has_ib else 0
            target.ib_rate_gbps = rate if rate is not None else accepted.ib_rate_gbps
            target.updated_at = utc_now()

    await session.commit()
    await session.refresh(row)
    return row


def _sim_report_for_node(
    node: Node,
    *,
    seed: int,
    topo_variant: str = "pack",
    collected_at: datetime | None = None,
) -> FabricReport:
    """Build a sim FabricReport bound to a real marketplace node id."""

    gpus = max(1, int(node.gpu_count or 1))
    # Derive a stable per-node index from id for IB guids when possible.
    index = abs(hash(node.id)) % 1000
    matrix = synthetic_nvlink_topo_matrix(gpus)
    if topo_variant != "pack":
        matrix = matrix + f"#variant={topo_variant}:seed={seed}:node={node.id}\n"

    from datetime import UTC

    # Identical seed + variant + node → identical collected_at for digest stability.
    when = collected_at or datetime(
        2026,
        7,
        1,
        (seed + index) % 24,
        seed % 60,
        0,
        tzinfo=UTC,
    )

    return build_fabric_report(
        node_id=node.id,
        collected_at=when,
        ib_devices=synthetic_ib_devices(node_index=index % 64, count=1, rate_gbps=200.0),
        gpu_gpu_topo_matrix=matrix,
        numa_map={f"gpu{g}": g % 2 for g in range(gpus)},
        nccl_version="sim-2.21.5",
        eth_ifaces=["eth0", "lo"],
        gpu_count=gpus,
        source="sim",
    )


async def fabric_scan_node(
    session: AsyncSession,
    *,
    node_id: str,
    source: Literal["sim", "scan", "inject", "manual"] = "sim",
    seed: int = 0,
    topo_variant: str = "pack",
    report: FabricReport | None = None,
) -> FabricReport:
    """Run fabric-scan on a node and persist the accepted FabricReport (VAL-FAB-018).

    Missing node → FabricReportError 404. In sim mode, synthetic IB/NVLink is used.
    Digest changes only when topology / seed / variant changes for that node.
    """

    node = await get_node(session, node_id)
    if node is None:
        raise FabricReportError(
            "node_not_found",
            "node not found",
            status_code=404,
        )

    if report is not None:
        if report.node_id != node_id:
            raise FabricReportError(
                "node_id_mismatch",
                "report.node_id does not match path node_id",
                status_code=422,
            )
        built = report
    elif source == "sim":
        # Reuse seed inventory graphs so multi-node planning still sees IB fabric.
        _ = seed_sim_inventory(seed=seed, node_count=1, gpus_per_node=max(1, node.gpu_count))
        built = _sim_report_for_node(node, seed=seed, topo_variant=topo_variant)
    else:
        # Generic scan path without HCA: eth-only matrix still carried digests.
        gpus = max(1, int(node.gpu_count or 1))
        built = build_fabric_report(
            node_id=node.id,
            collected_at=utc_now(),
            ib_devices=[],
            gpu_gpu_topo_matrix=synthetic_nvlink_topo_matrix(gpus),
            numa_map={f"gpu{g}": g % 2 for g in range(gpus)},
            nccl_version=None,
            eth_ifaces=["eth0"],
            gpu_count=gpus,
            source=source,
        )

    try:
        accepted = validate_accepted_report(built, gpu_count=int(node.gpu_count or built.gpu_count))
    except ValueError as exc:
        raise FabricReportError(
            "fabric_report_invalid",
            str(exc),
            status_code=422,
        ) from exc

    # If an identical digest already exists as latest, return it without dup row spam.
    latest = await get_latest_fabric_report(session, node_id)
    if latest is not None and latest.report_digest == accepted.report_digest:
        # Still refresh inventory denormalization to "usable".
        inventory = {
            "fabric_report": accepted.to_public(),
            "ib_devices": accepted.to_public()["ib_devices"],
            "ib_rate_gbps": accepted.ib_rate_gbps,
            "gpu_topo_sha256": accepted.gpu_topo_sha256,
            "report_digest": accepted.report_digest,
            "has_ib": bool(accepted.ib_devices),
        }
        has_ib, rate = _extract_ib_flags(inventory)
        node.inventory_json = json.dumps(inventory)
        node.has_ib = 1 if has_ib else 0
        node.ib_rate_gbps = rate if rate is not None else accepted.ib_rate_gbps
        node.updated_at = utc_now()
        await session.commit()
        return accepted

    await persist_fabric_report(session, accepted, node=node, update_node_inventory=True)
    return accepted


__all__ = [
    "FabricReportError",
    "fabric_scan_node",
    "get_latest_fabric_report",
    "list_fabric_reports",
    "persist_fabric_report",
    "report_to_public",
]
