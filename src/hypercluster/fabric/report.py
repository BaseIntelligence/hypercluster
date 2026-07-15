"""Job-level fabric report bundling and digests (architecture §8.1 / VAL-FAB-024).

Bundles per-node digests for multi-node jobs so fabric-report completeness matches
rankmap |unique node_ids|.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from hypercluster.fabric.discovery import (
    DIGEST_PREFIX,
    FabricReport,
    canonical_json,
    compute_report_digest,
)

BUNDLE_VERSION = "fabric-report-bundle.v1"


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def unique_node_ids_from_rankmap(rankmap: list[dict[str, Any]] | list[Any]) -> list[str]:
    """Stable unique node_id order as first-seen in rankmap."""

    seen: list[str] = []
    for item in rankmap:
        if isinstance(item, dict):
            nid = str(item.get("node_id") or "")
        else:
            nid = str(getattr(item, "node_id", "") or "")
        if nid and nid not in seen:
            seen.append(nid)
    return seen


def node_digest_entries(
    *,
    node_ids: list[str],
    reports: list[FabricReport] | None = None,
    fallback_digest_by_node: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """One entry per participating node with report_digest when known."""

    by_id: dict[str, FabricReport] = {}
    for report in reports or []:
        by_id[report.node_id] = report
    fallback = fallback_digest_by_node or {}
    entries: list[dict[str, Any]] = []
    for nid in node_ids:
        found = by_id.get(nid)
        if found is not None:
            entries.append(
                {
                    "node_id": nid,
                    "report_digest": found.report_digest,
                    "ib_devices": [d.to_public() for d in found.ib_devices],
                    "gpu_topo_sha256": found.gpu_topo_sha256,
                    "ib_rate_gbps": found.ib_rate_gbps,
                    "source": found.source,
                }
            )
        else:
            dig = fallback.get(nid) or (
                DIGEST_PREFIX
                + hashlib.sha256(f"missing-report:{nid}".encode()).hexdigest()
            )
            entries.append(
                {
                    "node_id": nid,
                    "report_digest": dig,
                    "ib_devices": [],
                    "gpu_topo_sha256": None,
                    "ib_rate_gbps": None,
                    "source": "missing",
                }
            )
    return entries


def compute_bundle_digest(nodes: list[dict[str, Any]], *, job_id: str) -> str:
    body = {
        "bundle_version": BUNDLE_VERSION,
        "job_id": job_id,
        "nodes": [
            {"node_id": n["node_id"], "report_digest": n["report_digest"]} for n in nodes
        ],
    }
    return DIGEST_PREFIX + hashlib.sha256(canonical_json(body).encode()).hexdigest()


def bundle_job_fabric_report(
    *,
    job_id: str,
    attempt_id: str | None,
    rankmap: list[dict[str, Any]] | list[Any],
    fabric_mode: str = "auto",
    world_size: int = 1,
    nnodes: int = 1,
    reports: list[FabricReport] | None = None,
    fallback_digest_by_node: dict[str, str] | None = None,
    nccl_version: str | None = "sim-2.21.5",
    collected_at: datetime | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a multi-node fabric view if |nodes| matches |nodes in rankmap| (VAL-FAB-024).

    Returns a dict suitable for JobFabricReport.raw_json + public GET.
    """

    when = collected_at or datetime.now(UTC)
    node_ids = unique_node_ids_from_rankmap(rankmap)
    # If rankmap empty, fall back to synthetic sim-node ids from nnodes.
    if not node_ids:
        node_ids = [f"sim-node-{i}" for i in range(max(1, int(nnodes)))]

    nodes = node_digest_entries(
        node_ids=node_ids,
        reports=reports,
        fallback_digest_by_node=fallback_digest_by_node,
    )
    # Aggregate IB devices / rates from member digests.
    ib_devices: list[dict[str, Any]] = []
    rates: list[float] = []
    for entry in nodes:
        for d in entry.get("ib_devices") or []:
            device = dict(d)
            device.setdefault("node_id", entry["node_id"])
            ib_devices.append(device)
        rate = entry.get("ib_rate_gbps")
        if rate is not None:
            rates.append(float(rate))
        # Also from nested ib_devices rates when ib_rate missing.
        for d in entry.get("ib_devices") or []:
            if "rate_gbps" in d:
                rates.append(float(d["rate_gbps"]))

    topo_parts = sorted(
        f"{n['node_id']}:{n.get('gpu_topo_sha256') or ''}:{n['report_digest']}" for n in nodes
    )
    gpu_topo_sha256 = hashlib.sha256("|".join(topo_parts).encode()).hexdigest()
    bundle_digest = compute_bundle_digest(nodes, job_id=job_id)

    body: dict[str, Any] = {
        "job_id": job_id,
        "attempt_id": attempt_id,
        "nnodes": int(nnodes),
        "world_size": int(world_size),
        "fabric_mode": fabric_mode,
        "bundle_version": BUNDLE_VERSION,
        "nodes": nodes,
        "node_count": len(nodes),
        "node_ids": list(node_ids),
        "ib_devices": ib_devices,
        "ib_rate_gbps": max(rates) if rates else None,
        "gpu_topo_sha256": gpu_topo_sha256,
        "numa_map": {f"gpu{i}": i % 2 for i in range(int(world_size))},
        "nccl_version": nccl_version,
        "eth_ifaces": ["lo", "eth0"],
        "collected_at": _iso_z(when),
        "report_digest": bundle_digest,
        "fabric_report_digest": bundle_digest,
    }
    if extra:
        body.update(extra)
    return body


def validate_bundle_completeness(
    bundle: dict[str, Any],
    rankmap: list[dict[str, Any]] | list[Any],
) -> bool:
    """True when |nodes in report| matches |unique nodes in rankmap|."""

    expected = set(unique_node_ids_from_rankmap(rankmap))
    if not expected:
        return True
    nodes = bundle.get("nodes") or []
    reported = {str(n.get("node_id")) for n in nodes if isinstance(n, dict)}
    return expected == reported and len(reported) == len(expected)


def report_digest_from_bundle(bundle: dict[str, Any]) -> str:
    digest = bundle.get("report_digest") or bundle.get("fabric_report_digest")
    if isinstance(digest, str) and digest.startswith(DIGEST_PREFIX):
        return digest
    # Recompute occasionally when missing.
    nodes = bundle.get("nodes") or []
    job_id = str(bundle.get("job_id") or "")
    return compute_bundle_digest(nodes, job_id=job_id)


# Re-export for callers that want to digest a single FabricReport still.
__all__ = [
    "BUNDLE_VERSION",
    "bundle_job_fabric_report",
    "compute_bundle_digest",
    "compute_report_digest",
    "node_digest_entries",
    "report_digest_from_bundle",
    "unique_node_ids_from_rankmap",
    "validate_bundle_completeness",
]
