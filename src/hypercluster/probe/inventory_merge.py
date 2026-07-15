"""Merge GpuHostEvidence measurements into node inventory_json (VAL-GPU-027).

Pure helpers: API/domain layers call these after a pass-all (or fail) probe so
``inventory_json`` gains ``gpu_uuids`` and ``gpu_probe_status`` without a fifth
scoring factor or Verda adapter.
"""

from __future__ import annotations

import json
from typing import Any

from hypercluster.probe.types import GpuHostEvidence

GPU_PROBE_STATUS_VERIFIED = "verified"
GPU_PROBE_STATUS_FAILED = "failed"
GPU_PROBE_STATUS_ERROR = "error"
GPU_PROBE_STATUS_NONE = "none"


def probe_status_label(evidence: GpuHostEvidence) -> str:
    if evidence.status == "passed":
        return GPU_PROBE_STATUS_VERIFIED
    if evidence.status == "error":
        return GPU_PROBE_STATUS_ERROR
    return GPU_PROBE_STATUS_FAILED


def merge_probe_into_inventory(
    existing: dict[str, Any] | str | None,
    evidence: GpuHostEvidence,
) -> dict[str, Any]:
    """Return inventory dict with measured UUID list + probe status fields.

    On **passed** evidence, always set ``gpu_uuids`` from measured inventory
    (VAL-GPU-027). Failed/error runs also stamp ``gpu_probe_status`` so callers
    can surface dirty silicon without inventing UUIDs from empty measured sets.
    """

    inv = _coerce_inventory(existing)
    inv["gpu_probe_status"] = probe_status_label(evidence)
    inv["gpu_probe_evidence_id"] = evidence.id
    inv["gpu_probe_transport"] = evidence.transport
    if evidence.failure_code:
        inv["gpu_probe_failure_code"] = evidence.failure_code
    else:
        inv.pop("gpu_probe_failure_code", None)

    uuids = evidence.measured.uuid_set()
    if evidence.status == "passed" or uuids:
        inv["gpu_uuids"] = list(uuids)
        inv["measured_gpu_count"] = evidence.measured.gpu_count
        if evidence.measured.gpus:
            names = [g.name for g in evidence.measured.gpus if g.name]
            if names:
                inv["measured_gpu_model"] = names[0]
            drivers = [g.driver_version for g in evidence.measured.gpus if g.driver_version]
            if drivers:
                inv["driver_version"] = drivers[0]
        digests = evidence.digests.model_dump(mode="json")
        inv["gpu_probe_digests"] = {k: v for k, v in digests.items() if v is not None}
    return inv


def inventory_json_dumps(inventory: dict[str, Any]) -> str:
    return json.dumps(inventory, sort_keys=True, separators=(",", ":"), default=str)


def apply_probe_to_node_fields(
    *,
    inventory_json: str | None,
    evidence: GpuHostEvidence,
) -> dict[str, Any]:
    """Build a field patch suitable for converting a Node row after probe.

    Returns keys: ``inventory_json`` (str), ``inventory`` (dict), and when pass
    yields optional measured gpu_model / gpu_count suggestions (callers decide
    whether to overwrite claim columns).
    """

    merged = merge_probe_into_inventory(inventory_json, evidence)
    patch: dict[str, Any] = {
        "inventory": merged,
        "inventory_json": inventory_json_dumps(merged),
        "gpu_probe_status": merged["gpu_probe_status"],
        "gpu_uuids": list(merged.get("gpu_uuids") or []),
    }
    if evidence.status == "passed" and evidence.measured.gpu_count > 0:
        patch["measured_gpu_count"] = evidence.measured.gpu_count
        if evidence.measured.gpus:
            patch["measured_gpu_model"] = evidence.measured.gpus[0].name
    return patch


def _coerce_inventory(existing: dict[str, Any] | str | None) -> dict[str, Any]:
    if existing is None:
        return {}
    if isinstance(existing, dict):
        return dict(existing)
    if isinstance(existing, str):
        text = existing.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return {"_raw_inventory": existing}
        if isinstance(data, dict):
            return dict(data)
        return {"_raw_inventory": data}
    return {}


__all__ = [
    "GPU_PROBE_STATUS_ERROR",
    "GPU_PROBE_STATUS_FAILED",
    "GPU_PROBE_STATUS_NONE",
    "GPU_PROBE_STATUS_VERIFIED",
    "apply_probe_to_node_fields",
    "inventory_json_dumps",
    "merge_probe_into_inventory",
    "probe_status_label",
]
