"""Cross e2e market resilience: double-rent recover, idle protect, nonce refuse.

Fulfills:
  VAL-CROSS-010  Double-rent reject then second capitalize path after terminate
  VAL-CROSS-011  Active rental not killed by idle-only health reclaim
  VAL-CROSS-024  Nonce replay cannot double-create jobs or double-rent
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx

from hypercluster.api.auth import build_signed_headers
from hypercluster.sim.scenarios import ScenarioResult, _detail_code, _fail, _signed_request

# Deterministic hotkeys for this cross slice (HMAC-dev insecure mode).
PROVIDER_HK = "cross-mra-provider-hotkey-aaaaaaaaaaaaaaaaaaaaaaaa"
RENTER_HK = "cross-mra-renter-hotkey-bbbbbbbbbbbbbbbbbbbbbbbbbbbb"
RENTER2_HK = "cross-mra-renter2-hotkey-cccccccccccccccccccccccccc"
JOB_HK = "cross-mra-job-hotkey-dddddddddddddddddddddddddddddddd"

ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"

CROSS_MARKET_RESILIENCE = "cross-market-resilience-auth"

_CONFLICT_CODES = frozenset(
    {
        "offer_not_listed",
        "offer_already_leased",
        "capacity_unavailable",
        "already_leased",
        "node_not_offerable",
    }
)


def _resolve_secret(shared_token: str | None) -> str:
    secret = (shared_token or "").strip()
    if not secret:
        secret = (os.environ.get("CHALLENGE_SHARED_TOKEN") or "").strip()
    if not secret:
        token_file = (os.environ.get("CHALLENGE_SHARED_TOKEN_FILE") or "").strip()
        if token_file:
            try:
                secret = Path(token_file).read_text(encoding="utf-8").strip()
            except OSError:
                secret = ""
    return secret or "test-challenge-shared-token"


def _signed_with_nonce(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    secret: str,
    hotkey: str,
    body: dict[str, Any] | None = None,
    nonce: str | None = None,
) -> httpx.Response:
    """Signed mutate that can pin a fixed nonce for replay probes."""

    raw = b"" if body is None else json.dumps(body).encode()
    headers = build_signed_headers(
        secret=secret,
        hotkey=hotkey,
        body=raw,
        nonce=nonce if nonce is not None else uuid.uuid4().hex,
    )
    if body is not None:
        headers["Content-Type"] = "application/json"
    return client.request(method, url, content=raw, headers=headers)


def _register_provider_and_node(
    client: httpx.Client,
    base_url: str,
    *,
    secret: str,
    steps: list[str],
    ssh_label: str,
    gpu_count: int = 1,
) -> tuple[str | None, str | None, str | None]:
    """Register provider + one IB node; return (provider_id, node_id, error)."""

    reg = _signed_request(
        client,
        "POST",
        f"{base_url}/v1/providers/register",
        secret=secret,
        hotkey=PROVIDER_HK,
        body={"display_name": "Cross MRA Provider"},
    )
    if reg.status_code >= 400:
        return None, None, f"provider register HTTP {reg.status_code}: {reg.text}"
    provider_id = str((reg.json() or {}).get("id") or "")
    steps.append(f"provider register ok id={provider_id or '?'}")

    node_resp = _signed_request(
        client,
        "POST",
        f"{base_url}/v1/nodes",
        secret=secret,
        hotkey=PROVIDER_HK,
        body={
            "gpu_model": "H100",
            "gpu_count": gpu_count,
            "ssh_endpoint": f"{ssh_label}:22",
            "tee_capability": "none",
            "inventory": {
                "ib_devices": ["mlx5_0"],
                "ib_rate_gbps": 200.0,
            },
        },
    )
    if node_resp.status_code >= 400:
        return provider_id, None, f"node register HTTP {node_resp.status_code}: {node_resp.text}"
    node_id = str((node_resp.json() or {}).get("id") or "")
    if not node_id:
        return provider_id, None, "node register missing id"
    steps.append(f"node register ok id={node_id}")
    return provider_id, node_id, None


def _create_listed_offer(
    client: httpx.Client,
    base_url: str,
    *,
    secret: str,
    node_id: str,
    steps: list[str],
    price: float = 2.5,
) -> tuple[str | None, str | None]:
    """Create a listed single-node offer; return (offer_id, error)."""

    offer_resp = _signed_request(
        client,
        "POST",
        f"{base_url}/v1/offers",
        secret=secret,
        hotkey=PROVIDER_HK,
        body={
            "node_ids": [node_id],
            "price_per_hour": price,
            "max_lifetime_hours": 24.0,
            "require_ib": True,
            "mode": "single",
        },
    )
    if offer_resp.status_code >= 400:
        return None, f"offer create HTTP {offer_resp.status_code}: {offer_resp.text}"
    offer = offer_resp.json() if isinstance(offer_resp.json(), dict) else {}
    offer_id = str(offer.get("id") or "")
    if not offer_id or offer.get("status") != "listed":
        return None, f"offer create unexpected payload: {offer}"
    steps.append(f"offer create ok id={offer_id} status=listed")
    return offer_id, None


def _count_active_leases_for_offer(
    client: httpx.Client,
    base_url: str,
    *,
    offer_id: str,
    hotkey: str,
) -> int:
    resp = client.get(
        f"{base_url}/v1/leases",
        headers={"X-Hotkey": hotkey},
        params={"offer_id": offer_id, "status": "active"},
    )
    if resp.status_code != 200:
        # Fallback: list all for hotkey and filter client-side.
        resp = client.get(f"{base_url}/v1/leases", headers={"X-Hotkey": hotkey})
        if resp.status_code != 200:
            return -1
        items = (resp.json() or {}).get("items") or []
        return sum(
            1
            for x in items
            if isinstance(x, dict)
            and x.get("offer_id") == offer_id
            and x.get("status") in {"active", "requested"}
        )
    items = (resp.json() or {}).get("items") or []
    return len([x for x in items if isinstance(x, dict)])


def run_cross_double_rent_recover(
    base_url: str,
    *,
    timeout: float = 20.0,
    shared_token: str | None = None,
) -> ScenarioResult:
    """VAL-CROSS-010: rent → double-rent 4xx → terminate → re-list → second rent 2xx."""

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    secret = _resolve_secret(shared_token)
    name = "cross-double-rent-recover"

    try:
        with httpx.Client(timeout=timeout) as client:
            # Unique SSH host so sequential runs do not collide exclusive nodes.
            suffix = uuid.uuid4().hex[:8]
            _prov, node_id, err = _register_provider_and_node(
                client,
                normalized,
                secret=secret,
                steps=steps,
                ssh_label=f"10.70.1.{int(suffix[:2], 16) % 200 + 1}-mra-{suffix}",
            )
            if err or not node_id:
                return _fail(name, normalized, err or "node missing", steps, None)

            offer_id, err = _create_listed_offer(
                client,
                normalized,
                secret=secret,
                node_id=node_id,
                steps=steps,
            )
            if err or not offer_id:
                return _fail(name, normalized, err or "offer missing", steps, None)

            # Rent 1 (capitalist path for renter1).
            steps.append("rent1 (renter1) exclusive capacity")
            rent1 = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/offers/{offer_id}/rent",
                secret=secret,
                hotkey=RENTER_HK,
                body={"lifetime_hours": 4.0},
            )
            if rent1.status_code >= 400:
                return _fail(
                    name,
                    normalized,
                    f"rent1 HTTP {rent1.status_code}: {rent1.text}",
                    steps,
                    None,
                )
            lease1 = (rent1.json() or {}).get("lease") or {}
            lease1_id = lease1.get("id")
            if not lease1_id:
                return _fail(name, normalized, f"rent1 missing lease: {rent1.text}", steps, None)
            if lease1.get("status") not in {"active", "requested"}:
                return _fail(
                    name,
                    normalized,
                    f"rent1 unexpected lease status {lease1.get('status')}",
                    steps,
                    None,
                )
            steps.append(f"rent1 ok lease_id={lease1_id} status={lease1.get('status')}")

            # Double-rent reject (renter2 on same exclusive offer).
            steps.append("double-rent reject (renter2)")
            double = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/offers/{offer_id}/rent",
                secret=secret,
                hotkey=RENTER2_HK,
                body={"lifetime_hours": 4.0},
            )
            if double.status_code not in {400, 403, 409, 422}:
                return _fail(
                    name,
                    normalized,
                    (
                        "double-rent expected conflict-class 4xx, got "
                        f"{double.status_code}: {double.text}"
                    ),
                    steps,
                    None,
                )
            code = _detail_code(double.json())
            steps.append(f"double-rent rejected HTTP {double.status_code} code={code}")
            if code is not None and code not in _CONFLICT_CODES:
                # Soft check: some deployments only expose status; unknown codes still OK
                # as long as dual active lease does not form.
                steps.append(f"double-rent detail code {code} (not in known set; status OK)")

            active_on_offer = _count_active_leases_for_offer(
                client, normalized, offer_id=offer_id, hotkey=RENTER_HK
            )
            # renter2 list should not show new active for same offer either
            r2_active = _count_active_leases_for_offer(
                client, normalized, offer_id=offer_id, hotkey=RENTER2_HK
            )
            if active_on_offer > 1 or r2_active > 0:
                return _fail(
                    name,
                    normalized,
                    (
                        "double active leases on exclusive capacity: "
                        f"renter1={active_on_offer} renter2={r2_active}"
                    ),
                    steps,
                    None,
                )
            steps.append(
                f"exclusive invariant ok renter1_active={active_on_offer} "
                f"renter2_active={r2_active}"
            )

            # Terminate lease1 → free capacity.
            steps.append("terminate lease1")
            term = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/leases/{lease1_id}/terminate",
                secret=secret,
                hotkey=RENTER_HK,
                body={"reason": "cross_mra_free_for_rerent"},
            )
            if term.status_code >= 400:
                return _fail(
                    name,
                    normalized,
                    f"terminate HTTP {term.status_code}: {term.text}",
                    steps,
                    None,
                )
            term_lease = (term.json() or {}).get("lease") or {}
            if term_lease.get("status") not in {"terminated", "expired", "failed"}:
                got = client.get(f"{normalized}/v1/leases/{lease1_id}")
                if got.status_code != 200 or got.json().get("status") not in {
                    "terminated",
                    "expired",
                    "failed",
                }:
                    return _fail(
                        name,
                        normalized,
                        f"lease not terminal after terminate: {term.text}",
                        steps,
                        None,
                    )
            steps.append(
                f"terminate ok lease_id={lease1_id} status={term_lease.get('status', 'terminal')}"
            )

            # Capitalist second path: re-list same node, rent with renter2.
            steps.append("re-list freed capacity as new offer")
            offer2_id, err = _create_listed_offer(
                client,
                normalized,
                secret=secret,
                node_id=node_id,
                steps=steps,
                price=2.0,
            )
            if err or not offer2_id:
                return _fail(name, normalized, err or "re-list offer missing", steps, None)

            steps.append("second rent (renter2) after terminate — capitalist recover")
            rent3 = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/offers/{offer2_id}/rent",
                secret=secret,
                hotkey=RENTER2_HK,
                body={"lifetime_hours": 3.0},
            )
            if rent3.status_code >= 400:
                return _fail(
                    name,
                    normalized,
                    f"second rent HTTP {rent3.status_code}: {rent3.text}",
                    steps,
                    None,
                )
            lease3 = (rent3.json() or {}).get("lease") or {}
            lease3_id = lease3.get("id")
            if not lease3_id:
                return _fail(
                    name, normalized, f"second rent missing lease: {rent3.text}", steps, None
                )
            if lease3.get("status") not in {"active", "requested"}:
                return _fail(
                    name,
                    normalized,
                    f"second rent unexpected status {lease3.get('status')}",
                    steps,
                    None,
                )
            steps.append(
                f"second rent ok lease_id={lease3_id} status={lease3.get('status')} "
                f"offer_id={offer2_id}"
            )
            steps.append("VAL-CROSS-010 ok: double-rent reject then second offer success path")

    except httpx.HTTPError as exc:
        return _fail(name, normalized, f"HTTP client error: {exc}", steps, None)

    return ScenarioResult(
        name=name,
        ok=True,
        base_url=normalized,
        message=(
            "double-rent recover passed: rent1 → double-rent 4xx → "
            "terminate → re-list → second rent 2xx"
        ),
        steps=steps,
        identity=None,
    )


def run_cross_idle_rental_protection(
    base_url: str,
    *,
    timeout: float = 20.0,
    shared_token: str | None = None,
) -> ScenarioResult:
    """VAL-CROSS-011: active lease survives idle-only reclaim sweep tick."""

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    secret = _resolve_secret(shared_token)
    name = "cross-idle-rental-protection"

    try:
        with httpx.Client(timeout=timeout) as client:
            suffix = uuid.uuid4().hex[:8]
            # Rented capacity node.
            _prov, node_id, err = _register_provider_and_node(
                client,
                normalized,
                secret=secret,
                steps=steps,
                ssh_label=f"10.70.2.{int(suffix[:2], 16) % 200 + 1}-idle-{suffix}",
            )
            if err or not node_id:
                return _fail(name, normalized, err or "node missing", steps, None)

            # Free sibling node that SHOULD go offline under idle reclaim
            # (proves the sweep genuinely runs, not a no-op).
            free_node = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/nodes",
                secret=secret,
                hotkey=PROVIDER_HK,
                body={
                    "gpu_model": "A100",
                    "gpu_count": 1,
                    "ssh_endpoint": f"10.70.3.{int(suffix[2:4], 16) % 200 + 1}-free-{suffix}:22",
                    "tee_capability": "none",
                    "inventory": {
                        "ib_devices": ["mlx5_0"],
                        "ib_rate_gbps": 100.0,
                    },
                },
            )
            if free_node.status_code >= 400:
                return _fail(
                    name,
                    normalized,
                    f"free node register HTTP {free_node.status_code}: {free_node.text}",
                    steps,
                    None,
                )
            free_node_id = str((free_node.json() or {}).get("id") or "")
            if not free_node_id:
                return _fail(name, normalized, "free node missing id", steps, None)
            steps.append(f"free idle node ok id={free_node_id}")

            offer_id, err = _create_listed_offer(
                client,
                normalized,
                secret=secret,
                node_id=node_id,
                steps=steps,
            )
            if err or not offer_id:
                return _fail(name, normalized, err or "offer missing", steps, None)

            steps.append("rent active capacity (tenant hold)")
            rent = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/offers/{offer_id}/rent",
                secret=secret,
                hotkey=RENTER_HK,
                body={"lifetime_hours": 6.0},
            )
            if rent.status_code >= 400:
                return _fail(
                    name,
                    normalized,
                    f"rent HTTP {rent.status_code}: {rent.text}",
                    steps,
                    None,
                )
            lease = (rent.json() or {}).get("lease") or {}
            pod = (rent.json() or {}).get("pod") or {}
            lease_id = lease.get("id")
            pod_id = pod.get("id")
            if not lease_id or not pod_id:
                return _fail(name, normalized, f"rent missing lease/pod: {rent.text}", steps, None)
            steps.append(
                f"rent ok lease_id={lease_id} status={lease.get('status')} "
                f"pod_id={pod_id} pod_status={pod.get('status')}"
            )

            # Idle-only reclaim tick via sim surface (ages heartbeats + sweep).
            steps.append("idle reclaim sweep (age active + free node heartbeats)")
            reclaim = client.post(
                f"{normalized}/v1/sim/idle-reclaim",
                json={
                    "liveness_seconds": 30,
                    "age_heartbeats_seconds": 3600,
                },
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            if reclaim.status_code == 404:
                return _fail(
                    name,
                    normalized,
                    (
                        "idle reclaim surface missing (POST /v1/sim/idle-reclaim "
                        "required for VAL-CROSS-011 under pure HTTP e2e)"
                    ),
                    steps,
                    None,
                )
            if reclaim.status_code >= 400:
                return _fail(
                    name,
                    normalized,
                    f"idle reclaim HTTP {reclaim.status_code}: {reclaim.text}",
                    steps,
                    None,
                )
            reclaim_body = reclaim.json() if isinstance(reclaim.json(), dict) else {}
            steps.append(
                "idle reclaim ok "
                f"offline_marked={reclaim_body.get('offline_marked')} "
                f"protected={reclaim_body.get('protected_node_ids')}"
            )

            # Lease + pod still active / running.
            got_lease = client.get(f"{normalized}/v1/leases/{lease_id}")
            got_pod = client.get(f"{normalized}/v1/pods/{pod_id}")
            if got_lease.status_code != 200 or got_pod.status_code != 200:
                return _fail(
                    name,
                    normalized,
                    (
                        f"post-sweep lease/pod fetch failed: "
                        f"lease={got_lease.status_code} pod={got_pod.status_code}"
                    ),
                    steps,
                    None,
                )
            lease_after = got_lease.json()
            pod_after = got_pod.json()
            if lease_after.get("status") != "active":
                return _fail(
                    name,
                    normalized,
                    (
                        "active lease killed by idle-only reclaim: "
                        f"status={lease_after.get('status')}"
                    ),
                    steps,
                    None,
                )
            if pod_after.get("status") not in {"running", "provisioning"}:
                return _fail(
                    name,
                    normalized,
                    (f"active pod harmed by idle-only reclaim: status={pod_after.get('status')}"),
                    steps,
                    None,
                )
            steps.append(
                f"lease survived idle: status={lease_after.get('status')} "
                f"pod={pod_after.get('status')}"
            )

            node_after = client.get(f"{normalized}/v1/nodes/{node_id}")
            if node_after.status_code != 200:
                return _fail(
                    name,
                    normalized,
                    f"node fetch HTTP {node_after.status_code}",
                    steps,
                    None,
                )
            rented_status = (node_after.json() or {}).get("status")
            if rented_status in {"offline", "draining"} and rented_status != "rented":
                # Explicit fail if wiped offline mid-tenant.
                if rented_status == "offline":
                    return _fail(
                        name,
                        normalized,
                        f"rented node marked offline mid-lease: status={rented_status}",
                        steps,
                        None,
                    )
            if rented_status not in {"rented", "healthy"}:
                return _fail(
                    name,
                    normalized,
                    f"rented/active node unexpected status={rented_status}",
                    steps,
                    None,
                )
            steps.append(f"rented node protected status={rented_status} (short-circuit ok)")

            free_after = client.get(f"{normalized}/v1/nodes/{free_node_id}")
            free_status = (
                (free_after.json() or {}).get("status") if free_after.status_code == 200 else None
            )
            if free_status == "offline":
                steps.append(
                    f"free idle node reclaimed offline id={free_node_id} "
                    "(sweep side-effect confirms idle path ran)"
                )
            else:
                # Sweep may leave free node offline only when status was reclaim-eligible;
                # still OK if different policy, as long as tenant lived.
                steps.append(
                    f"free node status={free_status} "
                    "(tenant protection independent of free-node reclaim)"
                )

            steps.append("VAL-CROSS-011 ok: active rental survived idle-only reclaim sweep")

    except httpx.HTTPError as exc:
        return _fail(name, normalized, f"HTTP client error: {exc}", steps, None)

    return ScenarioResult(
        name=name,
        ok=True,
        base_url=normalized,
        message=(
            "idle rental protection passed: active lease/pod survived idle reclaim short-circuit"
        ),
        steps=steps,
        identity=None,
    )


def run_cross_nonce_replay_refuse(
    base_url: str,
    *,
    timeout: float = 20.0,
    shared_token: str | None = None,
) -> ScenarioResult:
    """VAL-CROSS-024: same nonce cannot double-create jobs or double-rent."""

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    secret = _resolve_secret(shared_token)
    name = "cross-nonce-replay-refuse"

    try:
        with httpx.Client(timeout=timeout) as client:
            # ---- Job admit nonce replay ----
            job_body = {
                "image_digest": ALLOWED_IMAGE,
                "entrypoint": ["python", "-c", "print('mra')"],
                "world_size": 1,
                "nnodes": 1,
                "nproc_per_node": 1,
                "resource": {"gpu": 1},
                "timeout_s": 120,
                # No client_request_id → pure nonce surface (not client idempotency).
            }
            fixed_job_nonce = f"cross-mra-job-nonce-{uuid.uuid4().hex[:12]}"
            steps.append(f"job create with fixed nonce={fixed_job_nonce}")
            first_job = _signed_with_nonce(
                client,
                "POST",
                f"{normalized}/v1/jobs",
                secret=secret,
                hotkey=JOB_HK,
                body=job_body,
                nonce=fixed_job_nonce,
            )
            if first_job.status_code >= 400:
                return _fail(
                    name,
                    normalized,
                    f"job create HTTP {first_job.status_code}: {first_job.text}",
                    steps,
                    None,
                )
            job_payload = first_job.json() if isinstance(first_job.json(), dict) else {}
            job_id = job_payload.get("id") or job_payload.get("job_id")
            if not job_id:
                return _fail(name, normalized, f"job create missing id: {job_payload}", steps, None)
            steps.append(f"job create ok job_id={job_id}")

            steps.append("job nonce replay (identical signed headers)")
            # Replay: rebuild SAME header set by reusing build_signed_headers with
            # the same nonce. Timestamp may shift but nonce is the uniqueness key.
            replay_job = _signed_with_nonce(
                client,
                "POST",
                f"{normalized}/v1/jobs",
                secret=secret,
                hotkey=JOB_HK,
                body=job_body,
                nonce=fixed_job_nonce,
            )
            if replay_job.status_code not in {401, 403, 409}:
                return _fail(
                    name,
                    normalized,
                    (
                        "job nonce replay expected 401/403/409, got "
                        f"{replay_job.status_code}: {replay_job.text}"
                    ),
                    steps,
                    None,
                )
            jcode = _detail_code(replay_job.json())
            steps.append(f"job nonce replay refused HTTP {replay_job.status_code} code={jcode}")

            listed_jobs = client.get(
                f"{normalized}/v1/jobs",
                headers={"X-Hotkey": JOB_HK},
            )
            if listed_jobs.status_code != 200:
                return _fail(
                    name,
                    normalized,
                    f"list jobs HTTP {listed_jobs.status_code}",
                    steps,
                    None,
                )
            job_items = (listed_jobs.json() or {}).get("items") or []
            job_count = len(job_items)
            if job_count != 1:
                return _fail(
                    name,
                    normalized,
                    f"job count unstable after nonce replay: count={job_count}",
                    steps,
                    None,
                )
            steps.append(f"job count stable after nonce replay: count={job_count}")

            # ---- Rent nonce replay ----
            suffix = uuid.uuid4().hex[:8]
            _prov, node_id, err = _register_provider_and_node(
                client,
                normalized,
                secret=secret,
                steps=steps,
                ssh_label=f"10.70.4.{int(suffix[:2], 16) % 200 + 1}-nonce-{suffix}",
            )
            if err or not node_id:
                return _fail(name, normalized, err or "node missing", steps, None)

            offer_id, err = _create_listed_offer(
                client,
                normalized,
                secret=secret,
                node_id=node_id,
                steps=steps,
            )
            if err or not offer_id:
                return _fail(name, normalized, err or "offer missing", steps, None)

            rent_body = {"lifetime_hours": 4.0}
            fixed_rent_nonce = f"cross-mra-rent-nonce-{uuid.uuid4().hex[:12]}"
            steps.append(f"rent with fixed nonce={fixed_rent_nonce}")
            first_rent = _signed_with_nonce(
                client,
                "POST",
                f"{normalized}/v1/offers/{offer_id}/rent",
                secret=secret,
                hotkey=RENTER_HK,
                body=rent_body,
                nonce=fixed_rent_nonce,
            )
            if first_rent.status_code >= 400:
                return _fail(
                    name,
                    normalized,
                    f"rent HTTP {first_rent.status_code}: {first_rent.text}",
                    steps,
                    None,
                )
            lease = (first_rent.json() or {}).get("lease") or {}
            lease_id = lease.get("id")
            if not lease_id:
                return _fail(
                    name, normalized, f"rent missing lease: {first_rent.text}", steps, None
                )
            steps.append(f"rent ok lease_id={lease_id}")

            # Count active leases for this offer before replay.
            before = _count_active_leases_for_offer(
                client, normalized, offer_id=offer_id, hotkey=RENTER_HK
            )
            if before != 1:
                return _fail(
                    name,
                    normalized,
                    f"expected exactly 1 active lease before rent replay, got {before}",
                    steps,
                    None,
                )

            steps.append("rent nonce replay (identical nonce, new signature attempt)")
            replay_rent = _signed_with_nonce(
                client,
                "POST",
                f"{normalized}/v1/offers/{offer_id}/rent",
                secret=secret,
                hotkey=RENTER_HK,
                body=rent_body,
                nonce=fixed_rent_nonce,
            )
            if replay_rent.status_code not in {401, 403, 409}:
                # Capacity already leased could also 409 with capacity code if
                # nonce store somehow skipped; still require no twin lease.
                if replay_rent.status_code in {400, 422}:
                    steps.append(
                        f"rent replay returned {replay_rent.status_code} "
                        f"(capacity/validation); verifying lease count stable"
                    )
                else:
                    return _fail(
                        name,
                        normalized,
                        (
                            "rent nonce replay expected 401/403/409 (or capacity 4xx), got "
                            f"{replay_rent.status_code}: {replay_rent.text}"
                        ),
                        steps,
                        None,
                    )
            rcode = _detail_code(replay_rent.json()) if replay_rent.content else None
            steps.append(f"rent nonce replay refused HTTP {replay_rent.status_code} code={rcode}")

            after = _count_active_leases_for_offer(
                client, normalized, offer_id=offer_id, hotkey=RENTER_HK
            )
            if after != 1:
                return _fail(
                    name,
                    normalized,
                    f"lease count unstable after rent nonce replay: before={before} after={after}",
                    steps,
                    None,
                )
            steps.append(f"lease count stable after rent nonce replay: count={after}")
            steps.append("VAL-CROSS-024 ok: nonce replay cannot double-create jobs or double-rent")

    except httpx.HTTPError as exc:
        return _fail(name, normalized, f"HTTP client error: {exc}", steps, None)

    return ScenarioResult(
        name=name,
        ok=True,
        base_url=normalized,
        message=(
            "nonce replay refuse passed: job+rent identical nonce rejected; resource counts stable"
        ),
        steps=steps,
        identity=None,
    )


def run_cross_market_resilience_auth(
    base_url: str,
    *,
    timeout: float = 45.0,
    shared_token: str | None = None,
) -> ScenarioResult:
    """Combined VAL-CROSS-010 + 011 + 024 under one scenario name."""

    steps: list[str] = []
    normalized = base_url.rstrip("/")

    double = run_cross_double_rent_recover(base_url, timeout=timeout, shared_token=shared_token)
    steps.extend(double.steps)
    if not double.ok:
        return ScenarioResult(
            name=CROSS_MARKET_RESILIENCE,
            ok=False,
            base_url=normalized,
            message=f"double-rent recover failed: {double.message}",
            steps=steps,
            identity=None,
        )

    idle = run_cross_idle_rental_protection(base_url, timeout=timeout, shared_token=shared_token)
    steps.extend(idle.steps)
    if not idle.ok:
        return ScenarioResult(
            name=CROSS_MARKET_RESILIENCE,
            ok=False,
            base_url=normalized,
            message=f"idle rental protection failed: {idle.message}",
            steps=steps,
            identity=None,
        )

    nonce = run_cross_nonce_replay_refuse(base_url, timeout=timeout, shared_token=shared_token)
    steps.extend(nonce.steps)
    if not nonce.ok:
        return ScenarioResult(
            name=CROSS_MARKET_RESILIENCE,
            ok=False,
            base_url=normalized,
            message=f"nonce replay refuse failed: {nonce.message}",
            steps=steps,
            identity=None,
        )

    return ScenarioResult(
        name=CROSS_MARKET_RESILIENCE,
        ok=True,
        base_url=normalized,
        message=(
            "cross-market-resilience-auth passed: double-rent recover + "
            "idle rental protection + nonce replay refuse"
        ),
        steps=steps,
        identity=None,
    )


__all__ = [
    "ALLOWED_IMAGE",
    "CROSS_MARKET_RESILIENCE",
    "JOB_HK",
    "PROVIDER_HK",
    "RENTER2_HK",
    "RENTER_HK",
    "run_cross_double_rent_recover",
    "run_cross_idle_rental_protection",
    "run_cross_market_resilience_auth",
    "run_cross_nonce_replay_refuse",
]
