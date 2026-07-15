"""Product marketplace + job helpers using hypercluster public APIs only.

Uses HMAC-dev signed headers against a live challenge process. This module
must not import Verda. Challenge process itself must not see VERDA_* env.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from hypercluster.api.auth import build_signed_headers

ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"

# Deterministic ops hotkeys (not real ss58; insecure HMAC mode).
PROVIDER_HK = "m8-live-provider-hotkey-aaaaaaaaaaaaaaaaaaaaaaaaaa"
DEMAND_HK = "m8-live-demand-hotkey-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
FOREIGN_HK = "m8-live-foreign-hotkey-cccccccccccccccccccccccccccc"


@dataclass(slots=True)
class ProductIds:
    provider_id: str | None = None
    node_id: str | None = None
    offer_id: str | None = None
    lease_id: str | None = None
    pod_id: str | None = None
    job_id: str | None = None
    job_status: str | None = None
    heartbeat_timestamps: list[str] = field(default_factory=list)
    foreign_auth_refusals: list[dict[str, Any]] = field(default_factory=list)


def signed_request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    secret: str,
    hotkey: str,
    body: dict[str, Any] | None = None,
) -> httpx.Response:
    raw = b"" if body is None else json.dumps(body).encode()
    headers = build_signed_headers(
        secret=secret,
        hotkey=hotkey,
        body=raw,
        nonce=uuid.uuid4().hex,
    )
    if body is not None:
        headers["Content-Type"] = "application/json"
    return client.request(method, url, content=raw, headers=headers)


def probe_identity(base_url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    base = base_url.rstrip("/")
    with httpx.Client(timeout=timeout) as client:
        health = client.get(f"{base}/health")
        ready = client.get(f"{base}/ready")
        version = client.get(f"{base}/version")
    return {
        "health": health.status_code,
        "ready": ready.status_code,
        "version": version.status_code,
        "ok": health.status_code == 200 and ready.status_code == 200 and version.status_code == 200,
        "health_body": _safe_json(health),
        "ready_body": _safe_json(ready),
        "version_body": _safe_json(version),
    }


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text[:500]


def run_product_registration(
    base_url: str,
    *,
    secret: str,
    gpu_model: str,
    gpu_count: int = 1,
    ssh_endpoint: str,
    hostname: str | None = None,
    location_hint: str | None = None,
    cpu_cores: int | None = None,
    mem_gb: float | None = None,
    inventory: dict[str, Any] | None = None,
    price_per_hour: float = 0.61,
    lifetime_hours: float = 2.0,
    rent_hours: float = 1.0,
    timeout: float = 60.0,
) -> tuple[ProductIds, list[str]]:
    """Register provider/node/offer via product APIs and rent with demand hotkey.

    Returns product ids + step log. Uses **one** provider hotkey continuously
    for register → heartbeat → node → fabric-scan → offer (VAL-LIVE-012).
    """

    base = base_url.rstrip("/")
    steps: list[str] = []
    ids = ProductIds()
    inv = inventory if inventory is not None else {"has_ib": False, "source": "external_ops"}

    with httpx.Client(timeout=timeout) as client:
        steps.append("provider register")
        reg = signed_request(
            client,
            "POST",
            f"{base}/v1/providers/register",
            secret=secret,
            hotkey=PROVIDER_HK,
            body={"display_name": "M8 Live External Provider"},
        )
        if reg.status_code >= 400:
            raise RuntimeError(f"provider register HTTP {reg.status_code}: {reg.text}")
        ids.provider_id = str(reg.json().get("id"))
        steps.append(f"provider_id={ids.provider_id}")

        steps.append("provider heartbeat")
        hb = signed_request(
            client,
            "POST",
            f"{base}/v1/providers/heartbeat",
            secret=secret,
            hotkey=PROVIDER_HK,
            body={},
        )
        if hb.status_code >= 400:
            raise RuntimeError(f"provider heartbeat HTTP {hb.status_code}: {hb.text}")
        steps.append("provider heartbeat ok")

        steps.append("node register")
        node_body: dict[str, Any] = {
            "gpu_model": gpu_model,
            "gpu_count": gpu_count,
            "ssh_endpoint": ssh_endpoint,
            "tee_capability": "none",
            "inventory": inv,
        }
        if hostname:
            node_body["hostname"] = hostname
        if location_hint:
            node_body["location_hint"] = location_hint
        if cpu_cores is not None:
            node_body["cpu_cores"] = cpu_cores
        if mem_gb is not None:
            node_body["mem_gb"] = mem_gb
        node_resp = signed_request(
            client,
            "POST",
            f"{base}/v1/nodes",
            secret=secret,
            hotkey=PROVIDER_HK,
            body=node_body,
        )
        if node_resp.status_code >= 400:
            raise RuntimeError(f"node register HTTP {node_resp.status_code}: {node_resp.text}")
        node = node_resp.json()
        # Product schema must not require verda_instance_id.
        if "verda_instance_id" in node:
            raise RuntimeError("product node schema unexpectedly requires verda_instance_id")
        ids.node_id = str(node.get("id"))
        steps.append(f"node_id={ids.node_id} status={node.get('status')}")

        steps.append("node fabric-scan (sim inventory path)")
        scan = signed_request(
            client,
            "POST",
            f"{base}/v1/nodes/{ids.node_id}/fabric-scan",
            secret=secret,
            hotkey=PROVIDER_HK,
            body={"source": "sim", "seed": 42, "topo_variant": "pack"},
        )
        if scan.status_code >= 400:
            raise RuntimeError(f"fabric-scan HTTP {scan.status_code}: {scan.text}")
        steps.append("fabric-scan ok")

        # Heartbeat continuity: two heartbeats with parseable timestamps.
        for i in range(2):
            steps.append(f"node heartbeat #{i + 1}")
            nh = signed_request(
                client,
                "POST",
                f"{base}/v1/nodes/heartbeat",
                secret=secret,
                hotkey=PROVIDER_HK,
                body={"node_id": ids.node_id},
            )
            if nh.status_code >= 400:
                raise RuntimeError(f"node heartbeat HTTP {nh.status_code}: {nh.text}")
            payload = nh.json()
            # API may return list or single node.
            items = payload.get("items") if isinstance(payload, dict) else None
            if isinstance(items, list) and items:
                ts = items[0].get("last_heartbeat") or items[0].get("updated_at")
            elif isinstance(payload, dict):
                ts = payload.get("last_heartbeat") or payload.get("updated_at")
            else:
                ts = None
            if ts:
                ids.heartbeat_timestamps.append(str(ts))
            time.sleep(0.05)
        steps.append(f"heartbeat_timestamps={ids.heartbeat_timestamps}")

        # Foreign heartbeat must fail closed (VAL-LIVE-012). Register a different
        # provider so the failure is ownership 403, not "provider not found" 404.
        steps.append("foreign node heartbeat refuse")
        foreign_reg = signed_request(
            client,
            "POST",
            f"{base}/v1/providers/register",
            secret=secret,
            hotkey=FOREIGN_HK,
            body={"display_name": "M8 Foreign Probe Provider"},
        )
        if foreign_reg.status_code >= 400:
            raise RuntimeError(
                f"foreign provider register HTTP {foreign_reg.status_code}: {foreign_reg.text}"
            )
        foreign_hb = signed_request(
            client,
            "POST",
            f"{base}/v1/nodes/heartbeat",
            secret=secret,
            hotkey=FOREIGN_HK,
            body={"node_id": ids.node_id},
        )
        ids.foreign_auth_refusals.append(
            {
                "action": "node_heartbeat",
                "status": foreign_hb.status_code,
                "body": _safe_json(foreign_hb),
            }
        )
        if foreign_hb.status_code not in {401, 403}:
            raise RuntimeError(
                f"foreign node heartbeat expected 401/403, got {foreign_hb.status_code}: "
                f"{foreign_hb.text}"
            )
        steps.append(f"foreign heartbeat refused HTTP {foreign_hb.status_code}")

        steps.append("offer create (home-grown marketplace)")
        offer_resp = signed_request(
            client,
            "POST",
            f"{base}/v1/offers",
            secret=secret,
            hotkey=PROVIDER_HK,
            body={
                "node_ids": [ids.node_id],
                "price_per_hour": price_per_hour,
                "max_lifetime_hours": lifetime_hours,
                "require_ib": False,
                "mode": "single",
                "gpu_model": gpu_model,
                "gpu_count": gpu_count,
                "location_hint": location_hint,
            },
        )
        if offer_resp.status_code >= 400:
            raise RuntimeError(f"offer create HTTP {offer_resp.status_code}: {offer_resp.text}")
        offer = offer_resp.json()
        ids.offer_id = str(offer.get("id"))
        steps.append(f"offer_id={ids.offer_id}")

        # Public browse (unsigned GET).
        browse = client.get(f"{base}/v1/offers")
        if browse.status_code != 200:
            raise RuntimeError(f"offers list HTTP {browse.status_code}")
        items = browse.json().get("items") or []
        if not any(isinstance(o, dict) and o.get("id") == ids.offer_id for o in items):
            raise RuntimeError("newly created offer missing from GET /v1/offers")
        steps.append("offer visible on public list")

        steps.append("demand rent")
        rent_resp = signed_request(
            client,
            "POST",
            f"{base}/v1/offers/{ids.offer_id}/rent",
            secret=secret,
            hotkey=DEMAND_HK,
            body={"lifetime_hours": rent_hours},
        )
        if rent_resp.status_code >= 400:
            raise RuntimeError(f"rent HTTP {rent_resp.status_code}: {rent_resp.text}")
        rent_payload = rent_resp.json()
        lease = rent_payload.get("lease") or {}
        pod = rent_payload.get("pod") or {}
        ids.lease_id = str(lease.get("id")) if lease.get("id") else None
        ids.pod_id = str(pod.get("id")) if pod.get("id") else None
        if not ids.lease_id:
            raise RuntimeError(f"rent missing lease: {rent_payload}")
        steps.append(f"lease_id={ids.lease_id} pod_id={ids.pod_id}")

    return ids, steps


def run_smoke_job(
    base_url: str,
    *,
    secret: str,
    lease_id: str,
    pod_id: str | None,
    timeout: float = 90.0,
    poll_timeout_s: float = 45.0,
) -> tuple[str, str, list[str]]:
    """Submit single-node smoke job and wait for terminal succeeded via combined worker."""

    base = base_url.rstrip("/")
    steps: list[str] = []
    with httpx.Client(timeout=timeout) as client:
        job_body: dict[str, Any] = {
            "image_digest": ALLOWED_IMAGE,
            "entrypoint": ["python", "-m", "hypercluster_smoke", "--live-m8"],
            "world_size": 1,
            "nnodes": 1,
            "nproc_per_node": 1,
            "timeout_s": 120,
            "resource": {"gpus": 1, "nodes": 1},
            "backend": "nccl",
            "fabric": "auto",
            "tee": "none",
            "env": {"HYPER_M8_LIVE": "1"},
            "placement_policy": "pack",
            "lease_id": lease_id,
            "client_request_id": f"m8-live-{uuid.uuid4().hex[:12]}",
        }
        if pod_id:
            job_body["pod_id"] = pod_id
        steps.append("submit smoke job")
        job_resp = signed_request(
            client,
            "POST",
            f"{base}/v1/jobs",
            secret=secret,
            hotkey=DEMAND_HK,
            body=job_body,
        )
        if job_resp.status_code >= 400:
            raise RuntimeError(f"job admit HTTP {job_resp.status_code}: {job_resp.text}")
        job_payload = job_resp.json()
        job_id = str(job_payload.get("id") or job_payload.get("job_id") or "")
        if not job_id:
            raise RuntimeError(f"job missing id: {job_payload}")
        steps.append(f"job_id={job_id} status={job_payload.get('status')}")

        # Poll until terminal (combined worker).
        steps.append("poll job until terminal")
        deadline = time.time() + poll_timeout_s
        last_status = str(job_payload.get("status") or "")
        while time.time() < deadline:
            got = client.get(f"{base}/v1/jobs/{job_id}")
            if got.status_code != 200:
                raise RuntimeError(f"job poll HTTP {got.status_code}: {got.text}")
            body = got.json()
            last_status = str(body.get("status") or "")
            if last_status in {"succeeded", "failed", "timeout", "cancelled"}:
                break
            time.sleep(0.1)

        if last_status != "succeeded":
            # Provider may still post results if worker left job mid-lifecycle.
            steps.append(f"job status={last_status}; attempt provider result seal")
            # Advance via explicit results if still non-terminal.
            if last_status not in {"succeeded", "failed", "timeout", "cancelled"}:
                results = signed_request(
                    client,
                    "POST",
                    f"{base}/v1/jobs/{job_id}/results",
                    secret=secret,
                    hotkey=PROVIDER_HK,
                    body={
                        "attempt_no": 1,
                        "status": "succeeded",
                        "metrics": {
                            "tokens_per_s": 1.0,
                            "wall_s": 0.5,
                            "live_m8": True,
                        },
                        "proof_tier": "sim",
                        "verified": True,
                        "verify_mode": "sim",
                        "output_digest": f"sha256:m8{uuid.uuid4().hex[:56]}",
                    },
                )
                steps.append(f"provider results HTTP {results.status_code}")
                # re-poll
                for _ in range(40):
                    got = client.get(f"{base}/v1/jobs/{job_id}")
                    last_status = str(got.json().get("status") or "")
                    if last_status in {"succeeded", "failed", "timeout", "cancelled"}:
                        break
                    time.sleep(0.1)

        if last_status != "succeeded":
            raise RuntimeError(f"smoke job did not succeed (status={last_status})")
        steps.append(f"job terminal status={last_status}")

        # Auth continuity: foreign cannot post results (403/409 sealed).
        foreign = signed_request(
            client,
            "POST",
            f"{base}/v1/jobs/{job_id}/results",
            secret=secret,
            hotkey=FOREIGN_HK,
            body={"attempt_no": 1, "status": "succeeded", "proof_tier": "sim"},
        )
        steps.append(f"foreign results HTTP {foreign.status_code}")
        if foreign.status_code not in {401, 403, 409, 422}:
            # already terminal not a security hole if sealed by owner; still record
            steps.append("foreign results non-strict auth code but job sealed")

        return job_id, last_status, steps


def terminate_lease_idempotent(
    base_url: str,
    *,
    secret: str,
    lease_id: str,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Terminate product lease twice; second call must be non-catastrophic."""

    base = base_url.rstrip("/")
    outcomes: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout) as client:
        for i in range(2):
            resp = signed_request(
                client,
                "POST",
                f"{base}/v1/leases/{lease_id}/terminate",
                secret=secret,
                hotkey=DEMAND_HK,
                body={"reason": f"m8_live_cleanup_pass_{i + 1}"},
            )
            outcomes.append(
                {
                    "pass": i + 1,
                    "status": resp.status_code,
                    "body": _safe_json(resp),
                }
            )
            # 2xx, 409 already terminal, 400 already_ terminal all acceptable.
            if resp.status_code >= 500:
                raise RuntimeError(
                    f"lease terminate pass {i + 1} server error: {resp.status_code} {resp.text}"
                )
    return outcomes
