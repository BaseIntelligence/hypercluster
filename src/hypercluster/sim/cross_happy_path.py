"""Cross-area sim happy path: marketplace → rent → job → score → weights.

Fulfills VAL-CROSS-001/002/003/008/009/013 under pure local sim (no Verda).
"""

from __future__ import annotations

import math
import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from hypercluster.no_verda import VERDA_HOST_MARKERS
from hypercluster.sim.identity import IdentityReport, probe_identity_gates
from hypercluster.sim.scenarios import ScenarioResult, _fail, _signed_request

# Deterministic hotkeys for the cross happy path (HMAC-dev insecure mode).
PROVIDER_HK = "cross-happy-provider-hotkey-aaaaaaaaaaaaaaaaaaaaaa"
DEMAND_HK = "cross-happy-demand-hotkey-bbbbbbbbbbbbbbbbbbbbbbbb"
FOREIGN_HK = "cross-happy-foreign-hotkey-cccccccccccccccccccccccc"

ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"

# Host substrings that must never appear in outbound requests during pure sim.
# Shared with hypercluster.no_verda (VAL-LIVE-001/011 + VAL-CROSS-013).
_VERDA_HOST_MARKERS = VERDA_HOST_MARKERS


@dataclass(slots=True)
class EgressRecord:
    """Single observed outbound request host (scheme+netloc)."""

    method: str
    url: str
    host: str


@dataclass(slots=True)
class EgressTrace:
    """Accumulator for hosts touched during a pure-sim e2e run (VAL-CROSS-013)."""

    requests: list[EgressRecord] = field(default_factory=list)

    def record(self, method: str, url: str) -> None:
        parsed = urlparse(url)
        host = (parsed.hostname or parsed.netloc or "").lower()
        self.requests.append(EgressRecord(method=method, url=url, host=host))

    def verda_hits(self) -> list[EgressRecord]:
        hits: list[EgressRecord] = []
        for rec in self.requests:
            blob = f"{rec.host} {rec.url}".lower()
            if any(marker in blob for marker in _VERDA_HOST_MARKERS):
                hits.append(rec)
        return hits

    @property
    def verda_clean(self) -> bool:
        return not self.verda_hits()


@contextmanager
def capture_httpx_egress(trace: EgressTrace) -> Iterator[EgressTrace]:
    """Patch httpx Client/AsyncClient.send to record destination hosts.

    Used so pure sim e2e can prove zero Verda egress without pcap.
    """

    original_client_send = httpx.Client.send
    original_async_send = httpx.AsyncClient.send

    def _sync_send(self: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any) -> Any:
        trace.record(str(request.method), str(request.url))
        return original_client_send(self, request, *args, **kwargs)

    async def _async_send(
        self: httpx.AsyncClient, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> Any:
        trace.record(str(request.method), str(request.url))
        return await original_async_send(self, request, *args, **kwargs)

    httpx.Client.send = _sync_send  # type: ignore[method-assign]
    httpx.AsyncClient.send = _async_send  # type: ignore[method-assign]
    try:
        yield trace
    finally:
        httpx.Client.send = original_client_send  # type: ignore[method-assign]
        httpx.AsyncClient.send = original_async_send  # type: ignore[method-assign]


def _resolve_secret(shared_token: str | None) -> str:
    secret = (shared_token or "").strip()
    if not secret:
        secret = (os.environ.get("CHALLENGE_SHARED_TOKEN") or "").strip()
    if not secret:
        token_file = (os.environ.get("CHALLENGE_SHARED_TOKEN_FILE") or "").strip()
        if token_file:
            try:
                from pathlib import Path

                secret = Path(token_file).read_text(encoding="utf-8").strip()
            except OSError:
                secret = ""
    return secret or "test-challenge-shared-token"


def probe_baseline_identity(
    base_url: str,
    *,
    timeout: float = 5.0,
) -> tuple[IdentityReport, list[str], dict[str, int]]:
    """VAL-CROSS-001: health + version + ready all 200 on a fresh green start."""

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    codes: dict[str, int] = {}
    steps.append("baseline identity triangle: /health /version /ready")
    report = probe_identity_gates(normalized, timeout=timeout)
    codes["health"] = int(report.health_http_status or 0)
    codes["ready"] = int(report.ready_http_status or 0)

    try:
        with httpx.Client(timeout=timeout) as client:
            ver = client.get(f"{normalized}/version")
            codes["version"] = ver.status_code
            if ver.status_code != 200:
                report.errors.append(f"version HTTP {ver.status_code}")
                report.ok = False
            else:
                body = ver.json()
                if isinstance(body, dict):
                    slug = body.get("challenge_slug") or body.get("slug")
                    role = body.get("role")
                    if slug not in {None, "hypercluster"} and slug != "hypercluster":
                        report.errors.append(f"version slug={slug!r}")
                        report.ok = False
                    if role is not None and role != "challenge":
                        report.errors.append(f"version role={role!r}")
                        report.ok = False
                    cv = body.get("challenge_version")
                    if cv is not None and report.version and str(cv) != str(report.version):
                        report.errors.append(
                            f"version challenge_version={cv!r} != health.version={report.version!r}"
                        )
                        report.ok = False
                steps.append(f"GET /version → {ver.status_code}")
    except httpx.HTTPError as exc:
        codes["version"] = 0
        report.errors.append(f"version request failed: {exc}")
        report.ok = False

    if codes.get("health") != 200 or codes.get("ready") != 200 or codes.get("version") != 200:
        report.ok = False
        if codes.get("health") != 200:
            report.errors.append(f"baseline health code={codes.get('health')}")
        if codes.get("ready") != 200:
            report.errors.append(f"baseline ready code={codes.get('ready')}")
        if codes.get("version") != 200:
            report.errors.append(f"baseline version code={codes.get('version')}")

    if report.ok:
        steps.append(
            f"baseline green health={codes['health']} "
            f"version={codes['version']} ready={codes['ready']}"
        )
    return report, steps, codes


def run_cross_happy_path(
    base_url: str,
    *,
    timeout: float = 45.0,
    shared_token: str | None = None,
    poll_timeout_s: float = 20.0,
    identity_probe: Callable[..., IdentityReport] = probe_identity_gates,
) -> ScenarioResult:
    """Full marketplace→rent→job→score→weights causal chain (VAL-CROSS-002/003).

    Also exercises:
      - VAL-CROSS-001 baseline identity
      - VAL-CROSS-008 provider auth continuity (results + privileged legs)
      - VAL-CROSS-009 demand auth continuity (list+cancel isolation)
      - VAL-CROSS-013 zero Verda egress via httpx host trace
    """

    del identity_probe  # baseline uses full triangle; keep kw for parity
    normalized = base_url.rstrip("/")
    steps: list[str] = []
    ids: dict[str, str | None] = {
        "provider_id": None,
        "node_id": None,
        "offer_id": None,
        "lease_id": None,
        "pod_id": None,
        "job_id": None,
        "score_id": None,
        "attempt_id": None,
    }
    secret = _resolve_secret(shared_token)
    egress = EgressTrace()

    with capture_httpx_egress(egress):
        report, baseline_steps, codes = probe_baseline_identity(
            normalized, timeout=min(timeout, 5.0)
        )
        steps.extend(baseline_steps)
        if not report.ok:
            return ScenarioResult(
                name="cross-happy-path",
                ok=False,
                base_url=normalized,
                message=(
                    "cross-happy-path failed: baseline identity not green "
                    f"(codes={codes}; errors={'; '.join(report.errors)})"
                ),
                steps=steps,
                identity=report,
            )

        try:
            with httpx.Client(timeout=timeout) as client:
                # ----- Provider register + heartbeat -----
                steps.append("provider register")
                reg = _signed_request(
                    client,
                    "POST",
                    f"{normalized}/v1/providers/register",
                    secret=secret,
                    hotkey=PROVIDER_HK,
                    body={"display_name": "Cross Happy Path Provider"},
                )
                if reg.status_code >= 400:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"provider register HTTP {reg.status_code}: {reg.text}",
                        steps,
                        report,
                    )
                provider_id = reg.json().get("id")
                ids["provider_id"] = provider_id
                steps.append(f"provider_id={provider_id}")

                steps.append("provider heartbeat")
                hb = _signed_request(
                    client,
                    "POST",
                    f"{normalized}/v1/providers/heartbeat",
                    secret=secret,
                    hotkey=PROVIDER_HK,
                    body={},
                )
                if hb.status_code >= 400:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"provider heartbeat HTTP {hb.status_code}: {hb.text}",
                        steps,
                        report,
                    )
                steps.append("provider heartbeat ok")

                # ----- Node + fabric scan -----
                steps.append("node register")
                node_resp = _signed_request(
                    client,
                    "POST",
                    f"{normalized}/v1/nodes",
                    secret=secret,
                    hotkey=PROVIDER_HK,
                    body={
                        "gpu_model": "H100",
                        "gpu_count": 2,
                        "ssh_endpoint": "10.8.8.8:22",
                        "tee_capability": "none",
                        "inventory": {
                            "ib_devices": ["mlx5_0"],
                            "ib_rate_gbps": 200.0,
                        },
                    },
                )
                if node_resp.status_code >= 400:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"node register HTTP {node_resp.status_code}: {node_resp.text}",
                        steps,
                        report,
                    )
                node_id = node_resp.json().get("id")
                if not node_id:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        "node register missing id",
                        steps,
                        report,
                    )
                ids["node_id"] = str(node_id)
                steps.append(f"node_id={node_id}")

                steps.append("node fabric-scan (sim)")
                scan = _signed_request(
                    client,
                    "POST",
                    f"{normalized}/v1/nodes/{node_id}/fabric-scan",
                    secret=secret,
                    hotkey=PROVIDER_HK,
                    body={"source": "sim", "seed": 11, "topo_variant": "pack"},
                )
                if scan.status_code >= 400:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"fabric-scan HTTP {scan.status_code}: {scan.text}",
                        steps,
                        report,
                    )
                steps.append("fabric-scan ok")

                # ----- Offer + rent -----
                steps.append("offer create")
                offer_resp = _signed_request(
                    client,
                    "POST",
                    f"{normalized}/v1/offers",
                    secret=secret,
                    hotkey=PROVIDER_HK,
                    body={
                        "node_ids": [node_id],
                        "price_per_hour": 1.25,
                        "max_lifetime_hours": 12.0,
                        "require_ib": True,
                        "mode": "single",
                    },
                )
                if offer_resp.status_code >= 400:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"offer create HTTP {offer_resp.status_code}: {offer_resp.text}",
                        steps,
                        report,
                    )
                offer = offer_resp.json()
                offer_id = offer.get("id")
                if not offer_id:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"offer missing id: {offer}",
                        steps,
                        report,
                    )
                ids["offer_id"] = str(offer_id)
                steps.append(f"offer_id={offer_id}")

                steps.append("demand rent")
                rent_resp = _signed_request(
                    client,
                    "POST",
                    f"{normalized}/v1/offers/{offer_id}/rent",
                    secret=secret,
                    hotkey=DEMAND_HK,
                    body={"lifetime_hours": 4.0},
                )
                if rent_resp.status_code >= 400:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"rent HTTP {rent_resp.status_code}: {rent_resp.text}",
                        steps,
                        report,
                    )
                rent_payload = rent_resp.json()
                lease = rent_payload.get("lease") or {}
                pod = rent_payload.get("pod") or {}
                lease_id = lease.get("id")
                pod_id = pod.get("id")
                if not lease_id:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"rent missing lease: {rent_payload}",
                        steps,
                        report,
                    )
                ids["lease_id"] = str(lease_id)
                ids["pod_id"] = str(pod_id) if pod_id else None
                steps.append(f"lease_id={lease_id} pod_id={pod_id}")

                # Foreign terminate must fail (privileged security leg).
                steps.append("foreign terminate refuse")
                foreign_term = _signed_request(
                    client,
                    "POST",
                    f"{normalized}/v1/leases/{lease_id}/terminate",
                    secret=secret,
                    hotkey=FOREIGN_HK,
                    body={"reason": "cross_auth_probe"},
                )
                if foreign_term.status_code not in {401, 403}:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        (
                            "foreign terminate expected 401/403, got "
                            f"{foreign_term.status_code}: {foreign_term.text}"
                        ),
                        steps,
                        report,
                    )
                steps.append(f"foreign terminate refused HTTP {foreign_term.status_code}")

                # ----- Demand job submit bound to lease -----
                steps.append("demand job submit (lease-bound)")
                job_body = {
                    "image_digest": ALLOWED_IMAGE,
                    "entrypoint": ["python", "-m", "train", "--cross-happy"],
                    "world_size": 2,
                    "nnodes": 1,
                    "nproc_per_node": 2,
                    "timeout_s": 120,
                    "resource": {"gpus": 2, "nodes": 1},
                    "backend": "nccl",
                    "fabric": "auto",
                    "tee": "none",
                    "env": {"HYPER_CROSS": "1"},
                    "placement_policy": "pack",
                    "lease_id": lease_id,
                    "client_request_id": f"cross-happy-{uuid.uuid4().hex[:12]}",
                }
                if pod_id:
                    job_body["pod_id"] = pod_id
                job_resp = _signed_request(
                    client,
                    "POST",
                    f"{normalized}/v1/jobs",
                    secret=secret,
                    hotkey=DEMAND_HK,
                    body=job_body,
                )
                if job_resp.status_code >= 400:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"job submit HTTP {job_resp.status_code}: {job_resp.text}",
                        steps,
                        report,
                    )
                job_payload = job_resp.json()
                job_id = job_payload.get("id") or job_payload.get("job_id")
                if not job_id:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"job missing id: {job_payload}",
                        steps,
                        report,
                    )
                ids["job_id"] = str(job_id)
                steps.append(f"job_id={job_id} status={job_payload.get('status')}")

                # Demand list isolation: own job visible; foreign sees empty (or not ours).
                steps.append("demand list scoped")
                # List is sig-optional but scoped by X-Hotkey.
                own_list = client.get(
                    f"{normalized}/v1/jobs",
                    headers={"X-Hotkey": DEMAND_HK},
                )
                if own_list.status_code != 200:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"demand list HTTP {own_list.status_code}",
                        steps,
                        report,
                    )
                own_items = own_list.json().get("items") or []
                if not any(isinstance(j, dict) and j.get("id") == job_id for j in own_items):
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        "demand list missing own job",
                        steps,
                        report,
                    )
                foreign_list = client.get(
                    f"{normalized}/v1/jobs",
                    headers={"X-Hotkey": FOREIGN_HK},
                )
                if foreign_list.status_code != 200:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"foreign list HTTP {foreign_list.status_code}",
                        steps,
                        report,
                    )
                foreign_items = foreign_list.json().get("items") or []
                if any(isinstance(j, dict) and j.get("id") == job_id for j in foreign_items):
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        "foreign list incorrectly showed demand job",
                        steps,
                        report,
                    )
                steps.append("demand list isolation ok")

                # Foreign cancel must fail (VAL-CROSS-009).
                steps.append("foreign cancel refuse")
                foreign_cancel = _signed_request(
                    client,
                    "POST",
                    f"{normalized}/v1/jobs/{job_id}/cancel",
                    secret=secret,
                    hotkey=FOREIGN_HK,
                    body={"reason": "cross_auth_probe"},
                )
                if foreign_cancel.status_code not in {401, 403}:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        (
                            "foreign cancel expected 401/403, got "
                            f"{foreign_cancel.status_code}: {foreign_cancel.text}"
                        ),
                        steps,
                        report,
                    )
                steps.append(f"foreign cancel refused HTTP {foreign_cancel.status_code}")

                # Poll job to terminal (combined worker advances sim lifecycle).
                steps.append("poll job until terminal")
                terminal_status = None
                deadline = time.time() + poll_timeout_s
                last_status = job_payload.get("status")
                while time.time() < deadline:
                    got = client.get(f"{normalized}/v1/jobs/{job_id}")
                    if got.status_code != 200:
                        return _fail(
                            "cross-happy-path",
                            normalized,
                            f"job poll HTTP {got.status_code}: {got.text}",
                            steps,
                            report,
                        )
                    last_status = got.json().get("status")
                    if last_status in {
                        "succeeded",
                        "failed",
                        "timeout",
                        "cancelled",
                    }:
                        terminal_status = last_status
                        break
                    time.sleep(0.05)
                if terminal_status != "succeeded":
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"job did not succeed (last_status={last_status})",
                        steps,
                        report,
                    )
                steps.append(f"job terminal status={terminal_status}")

                # If results not sealed by worker yet, provider posts them (auth path).
                steps.append("provider post results (auth continuity)")
                attempt_get = client.get(f"{normalized}/v1/jobs/{job_id}/attempts/1")
                has_attempt = attempt_get.status_code == 200
                if has_attempt:
                    attempt_body = attempt_get.json()
                    ids["attempt_id"] = (
                        str(attempt_body.get("id") or attempt_body.get("attempt_id") or "") or None
                    )

                # Foreign results must be refused even post-success (403 or 409 sealed).
                foreign_results = _signed_request(
                    client,
                    "POST",
                    f"{normalized}/v1/jobs/{job_id}/results",
                    secret=secret,
                    hotkey=FOREIGN_HK,
                    body={
                        "attempt_no": 1,
                        "status": "succeeded",
                        "metrics": {"efficiency": 9.0, "fabric_gate": 1.0},
                        "proof_tier": "sim",
                        "verified": True,
                        "verify_mode": "sim",
                    },
                )
                if foreign_results.status_code not in {401, 403, 409}:
                    # 409 only if already sealed; foreign still should not re-score.
                    # Prefer hard 403 from auth layer.
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        (
                            "foreign results expected 401/403/409, got "
                            f"{foreign_results.status_code}: {foreign_results.text}"
                        ),
                        steps,
                        report,
                    )
                steps.append(f"foreign results refused HTTP {foreign_results.status_code}")

                # Provider happy path post (idempotent if already terminal).
                provider_results = _signed_request(
                    client,
                    "POST",
                    f"{normalized}/v1/jobs/{job_id}/results",
                    secret=secret,
                    hotkey=PROVIDER_HK,
                    body={
                        "attempt_no": 1,
                        "status": "succeeded",
                        "metrics": {
                            "efficiency": 5.0,
                            "fabric_gate": 1.0,
                            "allreduce_gbps": 12.0,
                        },
                        "fabric_report_digest": "sha256:" + ("ab" * 32),
                        "output_digest": "sha256:" + ("cd" * 32),
                        "proof_tier": "sim",
                        "verified": True,
                        "verify_mode": "sim",
                    },
                )
                if provider_results.status_code not in {200, 409}:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        (
                            "provider results expected 200/409, got "
                            f"{provider_results.status_code}: {provider_results.text}"
                        ),
                        steps,
                        report,
                    )
                steps.append(f"provider results ok HTTP {provider_results.status_code}")

                # ----- Score + leaderboard + weights (causal 1:1) -----
                steps.append("scores for demand hotkey")
                # Small settle for async score flush.
                score_deadline = time.time() + 5.0
                score_items: list[dict[str, Any]] = []
                while time.time() < score_deadline:
                    scores_resp = client.get(f"{normalized}/v1/scores/{DEMAND_HK}")
                    if scores_resp.status_code != 200:
                        return _fail(
                            "cross-happy-path",
                            normalized,
                            f"scores HTTP {scores_resp.status_code}: {scores_resp.text}",
                            steps,
                            report,
                        )
                    score_items = scores_resp.json().get("items") or []
                    if score_items:
                        break
                    time.sleep(0.1)
                if not score_items:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        "no score rows for demand hotkey after successful job",
                        steps,
                        report,
                    )
                first_score = score_items[0]
                raw_score_id = first_score.get("id") or first_score.get("score_id") or ""
                ids["score_id"] = str(raw_score_id) or None
                composite = float(first_score.get("composite") or 0.0)
                if not math.isfinite(composite) or composite <= 0:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"expected positive finite composite, got {composite}",
                        steps,
                        report,
                    )
                steps.append(
                    f"score_id={ids['score_id']} composite={composite} count={len(score_items)}"
                )

                steps.append("leaderboard mass for demand hotkey")
                board = client.get(f"{normalized}/v1/leaderboard")
                if board.status_code != 200:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"leaderboard HTTP {board.status_code}: {board.text}",
                        steps,
                        report,
                    )
                board_items = board.json().get("items") or []
                demand_row = next(
                    (
                        row
                        for row in board_items
                        if isinstance(row, dict) and row.get("hotkey") == DEMAND_HK
                    ),
                    None,
                )
                if demand_row is None:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"demand hotkey missing from leaderboard: {board_items}",
                        steps,
                        report,
                    )
                mass = float(
                    demand_row.get("aggregate")
                    or demand_row.get("mass")
                    or demand_row.get("composite")
                    or demand_row.get("score")
                    or 0.0
                )
                if not math.isfinite(mass) or mass <= 0:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"leaderboard mass not positive for demand: {demand_row}",
                        steps,
                        report,
                    )
                steps.append(f"leaderboard demand aggregate={mass}")

                steps.append("weight-preview causal 1:1")
                preview = client.get(f"{normalized}/v1/weight-preview")
                if preview.status_code != 200:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"weight-preview HTTP {preview.status_code}: {preview.text}",
                        steps,
                        report,
                    )
                body = preview.json()
                wmap = body.get("weights") if isinstance(body, dict) else None
                if not isinstance(wmap, dict):
                    wmap = {}
                # Positive weight for demand; foreign must not gain mass without attempts.
                demand_w = float(wmap.get(DEMAND_HK) or 0.0)
                if not math.isfinite(demand_w) or demand_w < 0:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"illegal demand weight {demand_w}",
                        steps,
                        report,
                    )
                if demand_w <= 0:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"demand hotkey missing positive weight: {wmap}",
                        steps,
                        report,
                    )
                foreign_w = float(wmap.get(FOREIGN_HK) or 0.0)
                if foreign_w > 0:
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"foreign hotkey gained weight without scores: {foreign_w}",
                        steps,
                        report,
                    )
                # Sole causal: every positive weight key must be from this chain's parties.
                positive_keys = [
                    k for k, v in wmap.items() if _safe_float(v) is not None and float(v) > 0
                ]
                allowed_keys = {DEMAND_HK, PROVIDER_HK}
                unexpected = [k for k in positive_keys if k not in allowed_keys]
                if unexpected:
                    # Soft note only if leftover scores from shared DB — cross fixture
                    # uses isolated DB, so unexpected is fail.
                    return _fail(
                        "cross-happy-path",
                        normalized,
                        f"unexpected positive weight hotkeys: {unexpected}",
                        steps,
                        report,
                    )
                steps.append(f"weights demand={demand_w} positive_keys={positive_keys}")

                # Threaded IDs summary (VAL-CROSS-002 evidence).
                timeline = (
                    f"timeline provider_id={ids['provider_id']} node_id={ids['node_id']} "
                    f"offer_id={ids['offer_id']} lease_id={ids['lease_id']} "
                    f"pod_id={ids['pod_id']} job_id={ids['job_id']} "
                    f"score_id={ids['score_id']} weights_keys={positive_keys}"
                )
                steps.append(timeline)

        except httpx.HTTPError as exc:
            return _fail(
                "cross-happy-path",
                normalized,
                f"HTTP client error: {exc}",
                steps,
                report,
            )

    # ----- VAL-CROSS-013: no Verda during pure sim -----
    verda_hits = egress.verda_hits()
    steps.append(f"egress requests={len(egress.requests)} verda_hits={len(verda_hits)}")
    if verda_hits:
        sample = ", ".join(f"{h.method} {h.url}" for h in verda_hits[:5])
        return ScenarioResult(
            name="cross-happy-path",
            ok=False,
            base_url=normalized,
            message=f"VERDA egress observed during pure sim e2e: {sample}",
            steps=steps,
            identity=report,
        )
    steps.append("no Verda egress during pure sim e2e")
    steps.append("cross-happy-path complete")

    return ScenarioResult(
        name="cross-happy-path",
        ok=True,
        base_url=normalized,
        message=(
            "cross-happy-path passed: baseline identity + marketplace→rent→job"
            "→score→weights causal chain; auth continuity; no Verda egress"
        ),
        steps=steps,
        identity=report,
    )


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ALLOWED_IMAGE",
    "DEMAND_HK",
    "FOREIGN_HK",
    "PROVIDER_HK",
    "EgressRecord",
    "EgressTrace",
    "capture_httpx_egress",
    "probe_baseline_identity",
    "run_cross_happy_path",
]
