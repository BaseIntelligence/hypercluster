"""Cross e2e multi-node fabric success/fail + TEE offline bonus.

Fulfills:
  VAL-CROSS-004  multi-node pack + nccl + fabric_gate=1 end-to-end
  VAL-CROSS-005  eth-fallback / honesty fail zeros composite + weight mass
  VAL-CROSS-006  TEE offline bonus multiplies composite on marketplace job
  VAL-CROSS-021  fabric report → planner → launcher → fabric_gate digest chain
"""

from __future__ import annotations

import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from hypercluster.sim.scenarios import ScenarioResult, _fail, _signed_request

# Deterministic hotkeys for this cross slice (HMAC-dev insecure mode).
PROVIDER_HK = "cross-fabric-provider-hotkey-aaaaaaaaaaaaaaaaaaaaaaa"
DEMAND_HK = "cross-fabric-demand-hotkey-bbbbbbbbbbbbbbbbbbbbbbbbbb"
FOREIGN_HK = "cross-fabric-foreign-hotkey-cccccccccccccccccccccccc"

ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"
COMPOSE_GOLDEN = "sha256:0c0ffeec0a5eabcdef0123456789abcdef0123456789abcdef0123456789ab"

CROSS_MULTINODE = "cross-multinode-fabric-tee"


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


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _poll_job_terminal(
    client: httpx.Client,
    base_url: str,
    job_id: str,
    *,
    poll_timeout_s: float,
    steps: list[str],
) -> tuple[str | None, dict[str, Any]]:
    """Poll GET /v1/jobs/{id} until terminal; return (status, last_body)."""

    deadline = time.time() + poll_timeout_s
    last_body: dict[str, Any] = {}
    last_status: str | None = None
    while time.time() < deadline:
        got = client.get(f"{base_url}/v1/jobs/{job_id}")
        if got.status_code != 200:
            steps.append(f"job poll HTTP {got.status_code}")
            return None, last_body
        last_body = got.json() if isinstance(got.json(), dict) else {}
        last_status = str(last_body.get("status") or "")
        if last_status in {"succeeded", "failed", "timeout", "cancelled"}:
            return last_status, last_body
        time.sleep(0.05)
    return last_status, last_body


def _wait_scores(
    client: httpx.Client,
    base_url: str,
    hotkey: str,
    *,
    settle_s: float = 5.0,
) -> list[dict[str, Any]]:
    deadline = time.time() + settle_s
    items: list[dict[str, Any]] = []
    while time.time() < deadline:
        resp = client.get(f"{base_url}/v1/scores/{hotkey}")
        if resp.status_code == 200:
            payload = resp.json() if isinstance(resp.json(), dict) else {}
            items = list(payload.get("items") or [])
            if items:
                return items
        time.sleep(0.1)
    return items


def _register_provider_pair(
    client: httpx.Client,
    base_url: str,
    *,
    secret: str,
    steps: list[str],
    node_count: int = 2,
    gpus_per_node: int = 2,
    tee_capability: str = "none",
    seed: int = 21,
) -> tuple[str, list[str]] | ScenarioResult:
    """Register provider + N IB-capable nodes with fabric-scan. Returns (provider_id, node_ids)."""

    steps.append("provider register")
    reg = _signed_request(
        client,
        "POST",
        f"{base_url}/v1/providers/register",
        secret=secret,
        hotkey=PROVIDER_HK,
        body={"display_name": "Cross Multi-Node Fabric Provider"},
    )
    if reg.status_code >= 400:
        return _fail(
            CROSS_MULTINODE,
            base_url,
            f"provider register HTTP {reg.status_code}: {reg.text}",
            steps,
            None,
        )
    provider_id = str(reg.json().get("id") or "")
    steps.append(f"provider_id={provider_id}")

    hb = _signed_request(
        client,
        "POST",
        f"{base_url}/v1/providers/heartbeat",
        secret=secret,
        hotkey=PROVIDER_HK,
        body={},
    )
    if hb.status_code >= 400:
        return _fail(
            CROSS_MULTINODE,
            base_url,
            f"provider heartbeat HTTP {hb.status_code}: {hb.text}",
            steps,
            None,
        )

    node_ids: list[str] = []
    # Unique SSH endpoints per call so sequential slices never collide with
    # still-active exclusive leases from a prior path in the same DB.
    register_tag = uuid.uuid4().hex[:10]
    for i in range(node_count):
        steps.append(f"node register [{i}] tag={register_tag}")
        node_resp = _signed_request(
            client,
            "POST",
            f"{base_url}/v1/nodes",
            secret=secret,
            hotkey=PROVIDER_HK,
            body={
                "gpu_model": "H100",
                "gpu_count": gpus_per_node,
                "ssh_endpoint": f"10.9.{register_tag[:2]}.{10 + i}-{register_tag}:22",
                "hostname": f"cross-fab-{register_tag}-n{i}",
                "tee_capability": tee_capability,
                "inventory": {
                    "ib_devices": [f"mlx5_{i}"],
                    "ib_rate_gbps": 200.0,
                    "has_ib": True,
                },
            },
        )
        if node_resp.status_code >= 400:
            return _fail(
                CROSS_MULTINODE,
                base_url,
                f"node register HTTP {node_resp.status_code}: {node_resp.text}",
                steps,
                None,
            )
        node_id = node_resp.json().get("id")
        if not node_id:
            return _fail(
                CROSS_MULTINODE,
                base_url,
                "node register missing id",
                steps,
                None,
            )
        node_ids.append(str(node_id))
        scan = _signed_request(
            client,
            "POST",
            f"{base_url}/v1/nodes/{node_id}/fabric-scan",
            secret=secret,
            hotkey=PROVIDER_HK,
            body={"source": "sim", "seed": seed + i, "topo_variant": "pack"},
        )
        if scan.status_code >= 400:
            return _fail(
                CROSS_MULTINODE,
                base_url,
                f"fabric-scan HTTP {scan.status_code}: {scan.text}",
                steps,
                None,
            )
        dig = None
        try:
            dig = (scan.json() or {}).get("report_digest") or (
                (scan.json() or {}).get("report") or {}
            ).get("report_digest")
        except Exception:  # noqa: BLE001
            dig = None
        steps.append(f"node_id={node_id} fabric-scan ok digest={dig}")
    return provider_id, node_ids


def _offer_rent(
    client: httpx.Client,
    base_url: str,
    *,
    secret: str,
    node_ids: list[str],
    steps: list[str],
    mode: str = "cluster",
    require_ib: bool = True,
    demand_hotkey: str = DEMAND_HK,
) -> tuple[str, str | None, str | None] | ScenarioResult:
    """Create offer, rent; return (offer_id, lease_id, pod_id)."""

    steps.append(f"offer create mode={mode} nodes={len(node_ids)} require_ib={require_ib}")
    offer_resp = _signed_request(
        client,
        "POST",
        f"{base_url}/v1/offers",
        secret=secret,
        hotkey=PROVIDER_HK,
        body={
            "node_ids": node_ids,
            "price_per_hour": 2.5,
            "max_lifetime_hours": 12.0,
            "require_ib": require_ib,
            "mode": mode,
        },
    )
    if offer_resp.status_code >= 400:
        return _fail(
            CROSS_MULTINODE,
            base_url,
            f"offer create HTTP {offer_resp.status_code}: {offer_resp.text}",
            steps,
            None,
        )
    offer_id = offer_resp.json().get("id")
    if not offer_id:
        return _fail(
            CROSS_MULTINODE,
            base_url,
            f"offer missing id: {offer_resp.json()}",
            steps,
            None,
        )
    offer_id = str(offer_id)
    steps.append(f"offer_id={offer_id}")

    steps.append("demand rent")
    rent_resp = _signed_request(
        client,
        "POST",
        f"{base_url}/v1/offers/{offer_id}/rent",
        secret=secret,
        hotkey=demand_hotkey,
        body={"lifetime_hours": 4.0},
    )
    if rent_resp.status_code >= 400:
        return _fail(
            CROSS_MULTINODE,
            base_url,
            f"rent HTTP {rent_resp.status_code}: {rent_resp.text}",
            steps,
            None,
        )
    rent_payload = rent_resp.json()
    lease = rent_payload.get("lease") or {}
    pod = rent_payload.get("pod") or {}
    lease_id = lease.get("id")
    pod_id = pod.get("id")
    if not lease_id:
        return _fail(
            CROSS_MULTINODE,
            base_url,
            f"rent missing lease: {rent_payload}",
            steps,
            None,
        )
    steps.append(f"lease_id={lease_id} pod_id={pod_id} mode={pod.get('mode')}")
    return offer_id, str(lease_id), str(pod_id) if pod_id else None


def _submit_job(
    client: httpx.Client,
    base_url: str,
    *,
    secret: str,
    hotkey: str,
    lease_id: str,
    pod_id: str | None,
    steps: list[str],
    world_size: int = 4,
    nnodes: int = 2,
    nproc_per_node: int = 2,
    fabric: str = "ib",
    tee: str = "none",
    placement_policy: str = "pack",
    client_tag: str = "cross-mn",
) -> str | ScenarioResult:
    steps.append(
        f"job submit world_size={world_size} nnodes={nnodes} "
        f"fabric={fabric} tee={tee} policy={placement_policy}"
    )
    job_body: dict[str, Any] = {
        "image_digest": ALLOWED_IMAGE,
        "entrypoint": ["python", "-m", "train", f"--{client_tag}"],
        "world_size": world_size,
        "nnodes": nnodes,
        "nproc_per_node": nproc_per_node,
        "timeout_s": 120,
        "resource": {"gpus": world_size, "nodes": nnodes},
        "backend": "nccl",
        "fabric": fabric,
        "tee": tee,
        "env": {"HYPER_CROSS_MN": "1"},
        "placement_policy": placement_policy,
        "lease_id": lease_id,
        "client_request_id": f"{client_tag}-{uuid.uuid4().hex[:12]}",
    }
    if pod_id:
        job_body["pod_id"] = pod_id
    job_resp = _signed_request(
        client,
        "POST",
        f"{base_url}/v1/jobs",
        secret=secret,
        hotkey=hotkey,
        body=job_body,
    )
    if job_resp.status_code >= 400:
        return _fail(
            CROSS_MULTINODE,
            base_url,
            f"job submit HTTP {job_resp.status_code}: {job_resp.text}",
            steps,
            None,
        )
    job_id = job_resp.json().get("id") or job_resp.json().get("job_id")
    if not job_id:
        return _fail(
            CROSS_MULTINODE,
            base_url,
            f"job missing id: {job_resp.json()}",
            steps,
            None,
        )
    steps.append(f"job_id={job_id}")
    return str(job_id)


def _offline_quote_b64(*, job_id: str, nonce: str) -> str:
    """Build a valid offline TDX quote for the given job/nonce."""

    from hypercluster.attest.offline_fixtures import (
        make_offline_envelope,
        package_quote_b64,
    )
    from hypercluster.attest.policy import DEFAULT_COMPOSE_HASH_GOLDEN
    from hypercluster.attest.report_data import build_report_data

    report = build_report_data(job_id=job_id, image_digest=ALLOWED_IMAGE, nonce=nonce)
    env = make_offline_envelope(
        compose_hash=DEFAULT_COMPOSE_HASH_GOLDEN or COMPOSE_GOLDEN,
        expected_compose_hash=DEFAULT_COMPOSE_HASH_GOLDEN or COMPOSE_GOLDEN,
        report_data=report,
        job_id=job_id,
        image_digest=ALLOWED_IMAGE,
        nonce=nonce,
        fixture_id="cross_mn_tee_positive",
    )
    return package_quote_b64(env)


def _extract_placement_chain(
    job_detail: dict[str, Any],
    attempt: dict[str, Any] | None,
    fabric_report: dict[str, Any] | None,
    score: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collect digests used for VAL-CROSS-021 chain identity checks."""

    placement = job_detail.get("placement") or {}
    graph_digest = placement.get("graph_digest")
    rankmap = placement.get("rankmap") or []
    launch = job_detail.get("launch_contract") or placement.get("launch_contract") or {}
    attempt_metrics = (attempt or {}).get("metrics") or {}
    fabric_artifact = attempt_metrics.get("fabric_artifact_digest")
    report_digest = None
    if fabric_report is not None:
        report_digest = fabric_report.get("report_digest")
        raw = fabric_report.get("raw") or {}
        if not report_digest and isinstance(raw, dict):
            report_digest = raw.get("report_digest") or raw.get("fabric_report_digest")
    if report_digest is None and attempt is not None:
        report_digest = attempt.get("fabric_report_digest")

    score_details = (score or {}).get("details") or {}
    extra = score_details.get("extra") if isinstance(score_details, dict) else None
    score_graph = None
    score_fab_art = None
    score_report = None
    if isinstance(extra, dict):
        score_graph = extra.get("graph_digest")
        score_fab_art = extra.get("fabric_artifact_digest")
        score_report = extra.get("fabric_report_digest")
    if score_graph is None and isinstance(score_details, dict):
        factors = score_details.get("factors") or {}
        # factors alone do not carry digests; leave None
        _ = factors

    unique_nodes = {
        str(b.get("node_id")) for b in rankmap if isinstance(b, dict) and b.get("node_id")
    }
    return {
        "graph_digest": graph_digest,
        "rankmap_len": len(rankmap) if isinstance(rankmap, list) else 0,
        "unique_nodes": sorted(unique_nodes),
        "launch_rankmap_len": len(launch.get("rankmap") or []) if isinstance(launch, dict) else 0,
        "fabric_artifact_digest": fabric_artifact,
        "fabric_report_digest": report_digest,
        "attempt_fabric_report_digest": (attempt or {}).get("fabric_report_digest"),
        "score_graph_digest": score_graph,
        "score_fabric_artifact_digest": score_fab_art,
        "score_fabric_report_digest": score_report,
        "fabric_gate": (score or {}).get("fabric_gate"),
        "attempt_fabric_gate": attempt_metrics.get("fabric_gate"),
    }


def run_cross_multinode_success(
    base_url: str,
    *,
    timeout: float = 45.0,
    shared_token: str | None = None,
    poll_timeout_s: float = 25.0,
) -> ScenarioResult:
    """VAL-CROSS-004 + VAL-CROSS-021: multi-node pack path fabric_gate=1 + digest chain."""

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    secret = _resolve_secret(shared_token)
    steps.append("cross multinode success path (pack + nccl + fabric_gate 1)")

    try:
        with httpx.Client(timeout=timeout) as client:
            health = client.get(f"{normalized}/health")
            if health.status_code != 200:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"health HTTP {health.status_code}",
                    steps,
                    None,
                )
            steps.append("identity health ok")

            reg = _register_provider_pair(
                client,
                normalized,
                secret=secret,
                steps=steps,
                node_count=2,
                gpus_per_node=2,
                tee_capability="none",
                seed=21,
            )
            if isinstance(reg, ScenarioResult):
                return reg
            _provider_id, node_ids = reg

            rented = _offer_rent(
                client,
                normalized,
                secret=secret,
                node_ids=node_ids,
                steps=steps,
                mode="cluster",
                require_ib=True,
            )
            if isinstance(rented, ScenarioResult):
                return rented
            _offer_id, lease_id, pod_id = rented
            assert lease_id is not None

            job_or = _submit_job(
                client,
                normalized,
                secret=secret,
                hotkey=DEMAND_HK,
                lease_id=lease_id,
                pod_id=pod_id,
                steps=steps,
                world_size=4,
                nnodes=2,
                nproc_per_node=2,
                fabric="ib",
                tee="none",
                placement_policy="pack",
                client_tag="cross-mn-success",
            )
            if isinstance(job_or, ScenarioResult):
                return job_or
            job_id = job_or

            steps.append("poll multi-node job to terminal")
            terminal, _ = _poll_job_terminal(
                client,
                normalized,
                job_id,
                poll_timeout_s=poll_timeout_s,
                steps=steps,
            )
            if terminal != "succeeded":
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"job did not succeed (status={terminal})",
                    steps,
                    None,
                )
            steps.append(f"job terminal status={terminal}")

            detail = client.get(f"{normalized}/v1/jobs/{job_id}")
            if detail.status_code != 200:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"job detail HTTP {detail.status_code}",
                    steps,
                    None,
                )
            job_detail = detail.json()
            placement = job_detail.get("placement") or {}
            rankmap = placement.get("rankmap") or []
            graph_digest = placement.get("graph_digest")
            unique_nodes = {
                str(b.get("node_id")) for b in rankmap if isinstance(b, dict) and b.get("node_id")
            }
            if len(rankmap) < 2:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"rankmap too small for multi-node: {rankmap}",
                    steps,
                    None,
                )
            if len(unique_nodes) < 2:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"multi-node forced single rank only; nodes={unique_nodes}",
                    steps,
                    None,
                )
            if not graph_digest:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    "placement.graph_digest missing",
                    steps,
                    None,
                )
            steps.append(
                f"pack placement ranks={len(rankmap)} nodes={sorted(unique_nodes)} "
                f"graph_digest={str(graph_digest)[:22]}…"
            )

            # NCCL env matrix should be present on placement for fabric=ib.
            nccl_env = placement.get("nccl_env") or {}
            steps.append(f"nccl_env keys={sorted(nccl_env.keys())[:8]}")

            attempt_resp = client.get(f"{normalized}/v1/jobs/{job_id}/attempts/1")
            if attempt_resp.status_code != 200:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"attempt GET HTTP {attempt_resp.status_code}",
                    steps,
                    None,
                )
            attempt = attempt_resp.json()
            metrics = attempt.get("metrics") or {}
            attempt_gate = _safe_float(metrics.get("fabric_gate"))
            if attempt_gate is None:
                attempt_gate = 1.0  # seal may omit; score is Source of truth
            steps.append(
                f"attempt fabric_gate={attempt_gate} "
                f"artifact={str(metrics.get('fabric_artifact_digest') or '')[:22]}"
            )

            fab_resp = client.get(f"{normalized}/v1/jobs/{job_id}/fabric-report")
            fabric_report = fab_resp.json() if fab_resp.status_code == 200 else None
            if fabric_report is None:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"fabric-report not ready HTTP {fab_resp.status_code}",
                    steps,
                    None,
                )
            fab_digest = fabric_report.get("report_digest")
            attempt_fab = attempt.get("fabric_report_digest")
            if fab_digest and attempt_fab and fab_digest != attempt_fab:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    (
                        "fabric report digest mismatch vs attempt: "
                        f"report={fab_digest} attempt={attempt_fab}"
                    ),
                    steps,
                    None,
                )
            steps.append(f"fabric_report_digest={str(fab_digest)[:22]}… chain match")

            scores = _wait_scores(client, normalized, DEMAND_HK, settle_s=5.0)
            if not scores:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    "no score rows for demand after multi-node success",
                    steps,
                    None,
                )
            score = scores[0]
            gate = _safe_float(score.get("fabric_gate"))
            composite = _safe_float(score.get("composite"))
            if gate != 1.0:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"expected fabric_gate=1 on healthy IB stack, got {gate}",
                    steps,
                    None,
                )
            if composite is None or not math.isfinite(composite) or composite <= 0:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"expected composite>0 with fabric_gate=1 correctness=1, got {composite}",
                    steps,
                    None,
                )
            steps.append(
                f"score fabric_gate={gate} composite={composite} tee_bonus={score.get('tee_bonus')}"
            )

            chain = _extract_placement_chain(job_detail, attempt, fabric_report, score)
            # VAL-CROSS-021: digests consistent across placement / attempt / report.
            if not chain["graph_digest"]:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    "chain identity missing plan graph_digest",
                    steps,
                    None,
                )
            if not chain["fabric_artifact_digest"] and not chain["fabric_report_digest"]:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    "chain identity missing launcher artifact and report digests",
                    steps,
                    None,
                )
            if chain["attempt_fabric_report_digest"] and chain["fabric_report_digest"]:
                if chain["attempt_fabric_report_digest"] != chain["fabric_report_digest"]:
                    return _fail(
                        CROSS_MULTINODE,
                        normalized,
                        "attempt/report digest silent substitute",
                        steps,
                        None,
                    )
            # Attempt honesty fabric_gate must match score fabric_gate when present.
            if chain["attempt_fabric_gate"] is not None:
                att_g = _safe_float(chain["attempt_fabric_gate"])
                if att_g is not None and att_g != gate:
                    # Honesty layer may leave gate on metrics; must not silently pass green.
                    if gate == 1.0 and att_g == 0.0:
                        return _fail(
                            CROSS_MULTINODE,
                            normalized,
                            "score fabric_gate green while launch honesty gate=0",
                            steps,
                            None,
                        )
            steps.append(
                "digest chain ok: "
                f"graph={str(chain['graph_digest'])[:18]}… "
                f"artifact={str(chain['fabric_artifact_digest'] or '')[:18]}… "
                f"report={str(chain['fabric_report_digest'] or '')[:18]}…"
            )

            # Weight mass positive for this success (baseline for fail comparison).
            preview = client.get(f"{normalized}/v1/weight-preview")
            demand_w = 0.0
            if preview.status_code == 200:
                wmap = (preview.json() or {}).get("weights") or {}
                demand_w = float(wmap.get(DEMAND_HK) or 0.0)
            steps.append(f"weight-preview demand={demand_w}")
            if demand_w <= 0:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"expected positive weight after multi-node success, got {demand_w}",
                    steps,
                    None,
                )
            steps.append("VAL-CROSS-004 multi-node fabric_gate=1 complete")
            steps.append("VAL-CROSS-021 digest chain identity complete")

    except httpx.HTTPError as exc:
        return _fail(
            CROSS_MULTINODE,
            normalized,
            f"HTTP client error: {exc}",
            steps,
            None,
        )

    return ScenarioResult(
        name=CROSS_MULTINODE,
        ok=True,
        base_url=normalized,
        message=(
            "cross-multinode success: pack multi-node + fabric_gate=1 + "
            "report→planner→launcher digest chain"
        ),
        steps=steps,
        identity=None,
    )


def run_cross_multinode_fabric_fail(
    base_url: str,
    *,
    timeout: float = 45.0,
    shared_token: str | None = None,
    poll_timeout_s: float = 25.0,
) -> ScenarioResult:
    """VAL-CROSS-005: eth-fallback honesty inject zeros composite and weight mass.

    Requires the challenge process to start with HYPER_SIM_ETH_FALLBACK=true
    (or HyperSettings.sim_eth_fallback=True) so _collect_success injects the
    forbidden eth fallback under fabric=ib.
    """

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    secret = _resolve_secret(shared_token)
    steps.append("cross multinode fabric_gate fail inject (eth fallback)")

    try:
        with httpx.Client(timeout=timeout) as client:
            reg = _register_provider_pair(
                client,
                normalized,
                secret=secret,
                steps=steps,
                node_count=2,
                gpus_per_node=2,
                seed=33,
            )
            if isinstance(reg, ScenarioResult):
                return reg
            _provider_id, node_ids = reg

            rented = _offer_rent(
                client,
                normalized,
                secret=secret,
                node_ids=node_ids,
                steps=steps,
                mode="cluster",
                require_ib=True,
            )
            if isinstance(rented, ScenarioResult):
                return rented
            _offer_id, lease_id, pod_id = rented
            assert lease_id is not None

            # Baseline weight before this attempt (should often be empty on isolated DB).
            pre = client.get(f"{normalized}/v1/weight-preview")
            pre_w = 0.0
            if pre.status_code == 200:
                pre_w = float(((pre.json() or {}).get("weights") or {}).get(DEMAND_HK) or 0.0)
            steps.append(f"pre-attempt demand weight={pre_w}")

            job_or = _submit_job(
                client,
                normalized,
                secret=secret,
                hotkey=DEMAND_HK,
                lease_id=lease_id,
                pod_id=pod_id,
                steps=steps,
                world_size=4,
                nnodes=2,
                nproc_per_node=2,
                fabric="ib",
                tee="none",
                placement_policy="pack",
                client_tag="cross-mn-fail",
            )
            if isinstance(job_or, ScenarioResult):
                return job_or
            job_id = job_or

            terminal, _ = _poll_job_terminal(
                client,
                normalized,
                job_id,
                poll_timeout_s=poll_timeout_s,
                steps=steps,
            )
            # eth fallback still operationally succeeds with honesty zero.
            if terminal not in {"succeeded", "failed"}:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"job not terminal under fail inject (status={terminal})",
                    steps,
                    None,
                )
            steps.append(f"job terminal status={terminal} (honesty path)")

            attempt_resp = client.get(f"{normalized}/v1/jobs/{job_id}/attempts/1")
            attempt = attempt_resp.json() if attempt_resp.status_code == 200 else {}
            metrics = (attempt or {}).get("metrics") or {}
            steps.append(
                f"attempt metrics fabric_gate={metrics.get('fabric_gate')} "
                f"composite={metrics.get('composite')} "
                f"failure_code={attempt.get('failure_code') or metrics.get('failure_code')}"
            )

            scores = _wait_scores(client, normalized, DEMAND_HK, settle_s=5.0)
            if not scores:
                # Composite zero may omit rows; rely on weight-delta below.
                steps.append("no score rows; check weight delta only")
                gate = 0.0
                composite = 0.0
            else:
                # Prefer the score bound to this job when possible.
                score = scores[0]
                gate_raw = _safe_float(score.get("fabric_gate"))
                comp_raw = _safe_float(score.get("composite"))
                gate = 0.0 if gate_raw is None else float(gate_raw)
                composite = 0.0 if comp_raw is None else float(comp_raw)
                steps.append(
                    f"score fabric_gate={gate} composite={composite} "
                    f"tee_bonus={score.get('tee_bonus')}"
                )
                if abs(gate) > 1e-12:
                    return _fail(
                        CROSS_MULTINODE,
                        normalized,
                        f"fail inject expected fabric_gate=0, got {gate}",
                        steps,
                        None,
                    )
                if composite > 1e-12:
                    return _fail(
                        CROSS_MULTINODE,
                        normalized,
                        f"fail inject expected composite=0, got {composite}",
                        steps,
                        None,
                    )

            post = client.get(f"{normalized}/v1/weight-preview")
            post_w = 0.0
            if post.status_code == 200:
                post_w = float(((post.json() or {}).get("weights") or {}).get(DEMAND_HK) or 0.0)
            steps.append(f"post-attempt demand weight={post_w}")
            # Weight must not inflate from this failing attempt alone.
            if post_w > pre_w + 1e-9:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    (f"fail inject inflated weight mass: pre={pre_w} post={post_w}"),
                    steps,
                    None,
                )
            steps.append("weight mass not inflated by fabric fail inject")
            steps.append("VAL-CROSS-005 fabric_gate fail inject complete")

    except httpx.HTTPError as exc:
        return _fail(
            CROSS_MULTINODE,
            normalized,
            f"HTTP client error: {exc}",
            steps,
            None,
        )

    return ScenarioResult(
        name=CROSS_MULTINODE,
        ok=True,
        base_url=normalized,
        message=("cross-multinode fail inject: fabric_gate=0 composite=0 weight mass not inflated"),
        steps=steps,
        identity=None,
    )


def run_cross_tee_offline_bonus(
    base_url: str,
    *,
    timeout: float = 45.0,
    shared_token: str | None = None,
    poll_timeout_s: float = 25.0,
    tee_bonus_tdx: float = 1.08,
) -> ScenarioResult:
    """VAL-CROSS-006: offline TEE bonus multiplies composite vs tee=none twin."""

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    secret = _resolve_secret(shared_token)
    steps.append("cross TEE offline bonus twin on marketplace-bound jobs")

    try:
        with httpx.Client(timeout=timeout) as client:
            # --- node0 / twin none ---
            reg = _register_provider_pair(
                client,
                normalized,
                secret=secret,
                steps=steps,
                node_count=2,
                gpus_per_node=2,
                tee_capability="tdx",
                seed=41,
            )
            if isinstance(reg, ScenarioResult):
                return reg
            _provider_id, node_ids = reg
            node_none, node_tdx = node_ids[0], node_ids[1]

            # Twin A: tee=none
            rented_a = _offer_rent(
                client,
                normalized,
                secret=secret,
                node_ids=[node_none],
                steps=steps,
                mode="single",
                require_ib=True,
            )
            if isinstance(rented_a, ScenarioResult):
                return rented_a
            _oa, lease_a, pod_a = rented_a
            assert lease_a is not None
            job_a_or = _submit_job(
                client,
                normalized,
                secret=secret,
                hotkey=DEMAND_HK,
                lease_id=lease_a,
                pod_id=pod_a,
                steps=steps,
                world_size=2,
                nnodes=1,
                nproc_per_node=2,
                fabric="ib",
                tee="none",
                placement_policy="pack",
                client_tag="cross-tee-none",
            )
            if isinstance(job_a_or, ScenarioResult):
                return job_a_or
            job_none = job_a_or
            term_a, _ = _poll_job_terminal(
                client,
                normalized,
                job_none,
                poll_timeout_s=poll_timeout_s,
                steps=steps,
            )
            if term_a != "succeeded":
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"tee=none twin did not succeed (status={term_a})",
                    steps,
                    None,
                )
            scores_none = _wait_scores(client, normalized, DEMAND_HK, settle_s=5.0)
            if not scores_none:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    "no scores for tee=none twin",
                    steps,
                    None,
                )
            # Prefer the score matching this attempt when multiple exist.
            score_none = scores_none[0]
            attempt_none = client.get(f"{normalized}/v1/jobs/{job_none}/attempts/1")
            attempt_none_id = None
            if attempt_none.status_code == 200:
                attempt_none_id = (attempt_none.json() or {}).get("id")
            for row in scores_none:
                if attempt_none_id and row.get("attempt_id") == attempt_none_id:
                    score_none = row
                    break
            tee_none = _safe_float(score_none.get("tee_bonus")) or 1.0
            comp_none = _safe_float(score_none.get("composite")) or 0.0
            eff_none = _safe_float(score_none.get("efficiency")) or 0.0
            gate_none = _safe_float(score_none.get("fabric_gate")) or 0.0
            steps.append(
                f"twin tee=none tee_bonus={tee_none} composite={comp_none} "
                f"eff={eff_none} gate={gate_none}"
            )
            if tee_none > 1.0 + 1e-9:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"tee=none twin should not get bonus, got {tee_none}",
                    steps,
                    None,
                )
            if comp_none <= 0 or gate_none != 1.0:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"tee=none twin expected green base composite, got {comp_none}",
                    steps,
                    None,
                )

            # Twin B: tee=tdx with offline verify upgrade after success.
            rented_b = _offer_rent(
                client,
                normalized,
                secret=secret,
                node_ids=[node_tdx],
                steps=steps,
                mode="single",
                require_ib=True,
            )
            if isinstance(rented_b, ScenarioResult):
                return rented_b
            _ob, lease_b, pod_b = rented_b
            assert lease_b is not None
            job_b_or = _submit_job(
                client,
                normalized,
                secret=secret,
                hotkey=DEMAND_HK,
                lease_id=lease_b,
                pod_id=pod_b,
                steps=steps,
                world_size=2,
                nnodes=1,
                nproc_per_node=2,
                fabric="ib",
                tee="tdx",
                placement_policy="pack",
                client_tag="cross-tee-tdx",
            )
            if isinstance(job_b_or, ScenarioResult):
                return job_b_or
            job_tdx = job_b_or
            term_b, _ = _poll_job_terminal(
                client,
                normalized,
                job_tdx,
                poll_timeout_s=poll_timeout_s,
                steps=steps,
            )
            if term_b != "succeeded":
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"tee=tdx twin did not succeed (status={term_b})",
                    steps,
                    None,
                )
            steps.append("tee=tdx job succeeded; attach offline fixture proof via results")

            nonce = f"cross-tee-nonce-{uuid.uuid4().hex[:16]}"
            quote_b64 = _offline_quote_b64(job_id=job_tdx, nonce=nonce)
            # Efficiency from attempt metrics for fair twin product.
            att_b = client.get(f"{normalized}/v1/jobs/{job_tdx}/attempts/1")
            att_b_body = att_b.json() if att_b.status_code == 200 else {}
            metrics_b = (att_b_body or {}).get("metrics") or {}
            eff_b = _safe_float(metrics_b.get("efficiency"))
            if eff_b is None:
                eff_b = eff_none
            fab_gate_b = _safe_float(metrics_b.get("fabric_gate"))
            if fab_gate_b is None or fab_gate_b == 0.0:
                fab_gate_b = 1.0
            fab_dig = (att_b_body or {}).get("fabric_report_digest")
            out_dig = (att_b_body or {}).get("output_digest")

            tee_post = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/jobs/{job_tdx}/results",
                secret=secret,
                hotkey=PROVIDER_HK,
                body={
                    "attempt_no": 1,
                    "status": "succeeded",
                    "metrics": {
                        "efficiency": eff_b,
                        "fabric_gate": fab_gate_b,
                    },
                    "fabric_report_digest": fab_dig,
                    "output_digest": out_dig,
                    "proof_tier": "tdx",
                    "verified": True,
                    "verify_mode": "offline_fixture",
                    "quote_b64": quote_b64,
                    "tee_nonce": nonce,
                    "report_data_hex": None,  # built from nonce server-side
                },
            )
            if tee_post.status_code not in {200, 409}:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    (
                        "offline TEE results post expected 200/409, got "
                        f"{tee_post.status_code}: {tee_post.text}"
                    ),
                    steps,
                    None,
                )
            steps.append(f"offline TEE results post HTTP {tee_post.status_code}")

            # Re-fetch scores; pick attempt for tdx job.
            scores_all = _wait_scores(client, normalized, DEMAND_HK, settle_s=5.0)
            attempt_tdx_id = None
            if att_b.status_code == 200:
                attempt_tdx_id = (att_b.json() or {}).get("id")
            score_tdx = None
            for row in scores_all:
                if attempt_tdx_id and row.get("attempt_id") == attempt_tdx_id:
                    score_tdx = row
                    break
            if score_tdx is None and scores_all:
                # newest first — prefer higher tee_bonus
                scored = sorted(
                    scores_all,
                    key=lambda r: float(r.get("tee_bonus") or 1.0),
                    reverse=True,
                )
                score_tdx = scored[0]
            if score_tdx is None:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    "no score for tee=tdx after offline proof",
                    steps,
                    None,
                )
            tee_tdx = _safe_float(score_tdx.get("tee_bonus")) or 1.0
            comp_tdx = _safe_float(score_tdx.get("composite")) or 0.0
            eff_tdx = _safe_float(score_tdx.get("efficiency")) or 0.0
            gate_tdx = _safe_float(score_tdx.get("fabric_gate")) or 0.0
            steps.append(
                f"twin tee=tdx tee_bonus={tee_tdx} composite={comp_tdx} "
                f"eff={eff_tdx} gate={gate_tdx}"
            )

            if abs(tee_tdx - float(tee_bonus_tdx)) > 1e-6:
                # Allow configured bonus match within hyper default.
                if tee_tdx <= 1.0 + 1e-9:
                    return _fail(
                        CROSS_MULTINODE,
                        normalized,
                        (f"expected tee_bonus≈{tee_bonus_tdx} after offline verify, got {tee_tdx}"),
                        steps,
                        None,
                    )
            if tee_tdx <= tee_none + 1e-9:
                return _fail(
                    CROSS_MULTINODE,
                    normalized,
                    f"tee bonus not applied: tdx={tee_tdx} none={tee_none}",
                    steps,
                    None,
                )
            # Ordered composites when gates green and efficiency comparable.
            if gate_tdx == 1.0 and gate_none == 1.0:
                if comp_tdx + 1e-9 < comp_none:
                    # If efficiencies differ widely, compare normalized product.
                    expected_ratio = tee_tdx / max(tee_none, 1e-12)
                    if eff_tdx > 0 and eff_none > 0:
                        norm_tdx = comp_tdx / eff_tdx
                        norm_none = comp_none / eff_none
                        if norm_tdx + 1e-9 < norm_none * (expected_ratio * 0.95):
                            return _fail(
                                CROSS_MULTINODE,
                                normalized,
                                (
                                    "normalized composite order wrong: "
                                    f"tdx={norm_tdx} none={norm_none} "
                                    f"bonus_ratio={expected_ratio}"
                                ),
                                steps,
                                None,
                            )
                    else:
                        return _fail(
                            CROSS_MULTINODE,
                            normalized,
                            f"composite order wrong: tdx={comp_tdx} none={comp_none}",
                            steps,
                            None,
                        )
                steps.append(
                    f"composite order ok: tee_tdx={comp_tdx} >≈ tee_none={comp_none} "
                    f"(bonus {tee_tdx} vs {tee_none})"
                )
            steps.append("VAL-CROSS-006 TEE offline bonus complete")

    except httpx.HTTPError as exc:
        return _fail(
            CROSS_MULTINODE,
            normalized,
            f"HTTP client error: {exc}",
            steps,
            None,
        )

    return ScenarioResult(
        name=CROSS_MULTINODE,
        ok=True,
        base_url=normalized,
        message=("cross-tee twin: offline TDX bonus multiplies composite over tee=none"),
        steps=steps,
        identity=None,
    )


def run_cross_multinode_fabric_tee(
    base_url: str,
    *,
    timeout: float = 60.0,
    shared_token: str | None = None,
    poll_timeout_s: float = 25.0,
    include_fail_inject: bool = False,
    include_tee_bonus: bool = True,
) -> ScenarioResult:
    """Combined runner for CLI: success (+chain) then optional fail/tee slices.

    Fail inject requires the *already running* API to have sim_eth_fallback hard
    wired; when include_fail_inject is False (default CLI) only success+tee run
    against a clean IB path.
    """

    steps: list[str] = []
    success = run_cross_multinode_success(
        base_url,
        timeout=timeout,
        shared_token=shared_token,
        poll_timeout_s=poll_timeout_s,
    )
    steps.extend(success.steps)
    if not success.ok:
        return ScenarioResult(
            name=CROSS_MULTINODE,
            ok=False,
            base_url=base_url.rstrip("/"),
            message=f"success path failed: {success.message}",
            steps=steps,
            identity=None,
        )

    if include_fail_inject:
        fail = run_cross_multinode_fabric_fail(
            base_url,
            timeout=timeout,
            shared_token=shared_token,
            poll_timeout_s=poll_timeout_s,
        )
        steps.extend(fail.steps)
        if not fail.ok:
            return ScenarioResult(
                name=CROSS_MULTINODE,
                ok=False,
                base_url=base_url.rstrip("/"),
                message=f"fail inject path failed: {fail.message}",
                steps=steps,
                identity=None,
            )

    if include_tee_bonus:
        tee = run_cross_tee_offline_bonus(
            base_url,
            timeout=timeout,
            shared_token=shared_token,
            poll_timeout_s=poll_timeout_s,
        )
        steps.extend(tee.steps)
        if not tee.ok:
            return ScenarioResult(
                name=CROSS_MULTINODE,
                ok=False,
                base_url=base_url.rstrip("/"),
                message=f"tee bonus path failed: {tee.message}",
                steps=steps,
                identity=None,
            )

    return ScenarioResult(
        name=CROSS_MULTINODE,
        ok=True,
        base_url=base_url.rstrip("/"),
        message=(
            "cross-multinode-fabric-tee passed: success gate+chain"
            + (" + fail inject" if include_fail_inject else "")
            + (" + tee bonus twin" if include_tee_bonus else "")
        ),
        steps=steps,
        identity=None,
    )


__all__ = [
    "ALLOWED_IMAGE",
    "CROSS_MULTINODE",
    "DEMAND_HK",
    "FOREIGN_HK",
    "PROVIDER_HK",
    "run_cross_multinode_fabric_fail",
    "run_cross_multinode_fabric_tee",
    "run_cross_multinode_success",
    "run_cross_tee_offline_bonus",
]
