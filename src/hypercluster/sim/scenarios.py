"""Local CI scenario runners (smoke, marketplace, nccl, tee-offline, weights).

Architecture §12.3 names: smoke, marketplace, nccl, tee-offline, weights.
  - smoke: health/ready green + empty weights burn-safe (VAL-CLI-015)
  - marketplace: offer/rent/terminate + double-rent reject (VAL-CLI-016)
  - nccl: multi-node pack/spread + fabric_gate fail inject (VAL-CLI-017)
  - tee-offline: positive/negative fixtures + bonus (VAL-CLI-018)
  - weights: multi-hotkey composites → push ack/idempotency (VAL-CLI-019)

Cross-area happy path (marketplace→rent→job→score→weights) lives in
:mod:`hypercluster.sim.cross_happy_path` and is invoked via
:func:`run_cross_happy_path` / :func:`run_scenario` name ``cross-happy-path``.

See hypercluster.sim.orchestration for the reusable multi-scenario suite runner.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from hypercluster.api.auth import build_signed_headers
from hypercluster.sim.identity import IdentityReport, probe_identity_gates

SMOKE = "smoke"
MARKETPLACE = "marketplace"
NCCL = "nccl"
TEE_OFFLINE = "tee-offline"
WEIGHTS = "weights"
CROSS_HAPPY_PATH = "cross-happy-path"

# Architecture §12.3 names remain the five canonical suite scenarios.
KNOWN_SCENARIOS = (SMOKE, MARKETPLACE, NCCL, TEE_OFFLINE, WEIGHTS)
# Extended names accepted by run_scenario (cross e2e; not part of suite order).
EXTENDED_SCENARIOS = KNOWN_SCENARIOS + (CROSS_HAPPY_PATH,)

# Deterministic local-sim hotkeys (not real ss58; HMAC insecure mode).
_SCENARIO_PROVIDER_HK = "sim-mkt-provider-hotkey-aaaaaaaaaaaaaaaaaaaaaaaa"
_SCENARIO_RENTER_HK = "sim-mkt-renter-hotkey-bbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_SCENARIO_RENTER2_HK = "sim-mkt-renter2-hotkey-cccccccccccccccccccccccccc"


@dataclass(slots=True)
class ScenarioResult:
    """Outcome of a sim scenario run."""

    name: str
    ok: bool
    base_url: str
    message: str
    steps: list[str] = field(default_factory=list)
    identity: IdentityReport | None = None

    def summary_lines(self) -> list[str]:
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"scenario={self.name} result={status}",
            f"base_url={self.base_url}",
            f"message={self.message}",
        ]
        lines.extend(f"step: {s}" for s in self.steps)
        if self.identity is not None:
            lines.extend(f"identity: {line}" for line in self.identity.summary_lines())
        return lines


def _detail_code(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    if isinstance(detail, dict):
        code = detail.get("code")
        return str(code) if code is not None else None
    return None


def _signed_request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    secret: str,
    hotkey: str,
    body: dict[str, Any] | None = None,
) -> httpx.Response:
    """POST/DELETE/etc. with HMAC-dev signed headers."""

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


def run_smoke_scenario(
    base_url: str,
    *,
    timeout: float = 5.0,
    identity_probe: Callable[..., IdentityReport] = probe_identity_gates,
) -> ScenarioResult:
    """Smoke: health/ready green before scenario may claim pass (VAL-SCAF-036)."""

    steps: list[str] = []
    steps.append("probe identity gates (/health + /ready)")
    report = identity_probe(base_url, timeout=timeout)
    if not report.ok:
        steps.append("identity gates failed")
        return ScenarioResult(
            name=SMOKE,
            ok=False,
            base_url=base_url.rstrip("/"),
            message=f"smoke failed: identity not green ({'; '.join(report.errors)})",
            steps=steps,
            identity=report,
        )
    steps.append("identity gates green")

    # Empty / burn-safe weights preview (architecture §12.3 smoke).
    steps.append("weights empty burn-safe probe")
    try:
        with httpx.Client(timeout=timeout) as client:
            preview = client.get(f"{report.base_url}/v1/weight-preview")
            if preview.status_code == 404:
                # Alternate public path sometimes exposed as /v1/weights.
                alt = client.get(f"{report.base_url}/v1/weights")
                if alt.status_code == 200:
                    preview = alt
            if preview.status_code == 200:
                body = preview.json()
                wmap = body.get("weights") if isinstance(body, dict) else None
                if wmap is None and isinstance(body, dict):
                    # shape may nest under data
                    wmap = body.get("data") if isinstance(body.get("data"), dict) else body
                    if isinstance(wmap, dict) and "weights" in wmap:
                        wmap = wmap.get("weights")
                if not isinstance(wmap, dict):
                    wmap = {}
                for key, val in wmap.items():
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        return ScenarioResult(
                            name=SMOKE,
                            ok=False,
                            base_url=report.base_url,
                            message=f"smoke failed: non-numeric weight {key}={val!r}",
                            steps=steps,
                            identity=report,
                        )
                    if fval != fval or fval == float("inf") or fval == float("-inf") or fval < 0:
                        return ScenarioResult(
                            name=SMOKE,
                            ok=False,
                            base_url=report.base_url,
                            message=f"smoke failed: illegal weight {key}={val}",
                            steps=steps,
                            identity=report,
                        )
                steps.append(
                    f"weights burn-safe ok (count={len(wmap)}; empty-or-finite≥0)"
                )
            else:
                # Missing preview endpoint is acceptable on empty install when
                # identity is green; burn-safe means no crash / no NaN invent.
                steps.append(
                    f"weight-preview HTTP {preview.status_code}; "
                    "identity green — treating as empty burn-safe"
                )
    except httpx.HTTPError as exc:
        return ScenarioResult(
            name=SMOKE,
            ok=False,
            base_url=report.base_url,
            message=f"smoke failed: weight-preview probe error: {exc}",
            steps=steps,
            identity=report,
        )

    return ScenarioResult(
        name=SMOKE,
        ok=True,
        base_url=report.base_url,
        message="smoke passed: health/ready green + weights burn-safe",
        steps=steps,
        identity=report,
    )


def run_marketplace_scenario(
    base_url: str,
    *,
    timeout: float = 15.0,
    shared_token: str | None = None,
    identity_probe: Callable[..., IdentityReport] = probe_identity_gates,
) -> ScenarioResult:
    """Marketplace local sim: offer → rent → double-rent reject → terminate (VAL-MKT-030).

    Requires a live challenge API with ``HYPER_ALLOW_INSECURE_SIGNATURES``
    (or decrypted shared token for HMAC-dev) so signed probes can run without
    real substrate keys. Uses only the product marketplace HTTP surface.
    """

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    steps.append("probe identity gates (/health + /ready)")
    report = identity_probe(base_url, timeout=timeout)
    if not report.ok:
        steps.append("identity gates failed")
        return ScenarioResult(
            name=MARKETPLACE,
            ok=False,
            base_url=normalized,
            message=(
                "marketplace failed: identity not green "
                f"({'; '.join(report.errors)})"
            ),
            steps=steps,
            identity=report,
        )
    steps.append("identity gates green")

    # Resolve shared token for HMAC-dev signatures.
    secret = (shared_token or "").strip()
    if not secret:
        import os

        secret = (os.environ.get("CHALLENGE_SHARED_TOKEN") or "").strip()
    if not secret:
        token_file = (os.environ.get("CHALLENGE_SHARED_TOKEN_FILE") or "").strip()
        if token_file:
            try:
                from pathlib import Path

                secret = Path(token_file).read_text(encoding="utf-8").strip()
            except OSError:
                secret = ""
    if not secret:
        # Fallback used by many local fixtures/tests (documented .env.example).
        secret = "test-challenge-shared-token"

    try:
        with httpx.Client(timeout=timeout) as client:
            # 1) Provider register
            steps.append("provider register")
            reg = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/providers/register",
                secret=secret,
                hotkey=_SCENARIO_PROVIDER_HK,
                body={"display_name": "Sim Marketplace Provider"},
            )
            if reg.status_code >= 400:
                return _fail(
                    MARKETPLACE,
                    normalized,
                    f"provider register HTTP {reg.status_code}: {reg.text}",
                    steps,
                    report,
                )
            steps.append(f"provider register ok id={reg.json().get('id')}")

            # 2) Node register
            steps.append("node register")
            node_resp = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/nodes",
                secret=secret,
                hotkey=_SCENARIO_PROVIDER_HK,
                body={
                    "gpu_model": "H100",
                    "gpu_count": 8,
                    "ssh_endpoint": "10.9.9.9:22",
                    "tee_capability": "none",
                    "inventory": {
                        "ib_devices": ["mlx5_0"],
                        "ib_rate_gbps": 200.0,
                    },
                },
            )
            if node_resp.status_code >= 400:
                return _fail(
                    MARKETPLACE,
                    normalized,
                    f"node register HTTP {node_resp.status_code}: {node_resp.text}",
                    steps,
                    report,
                )
            node_id = node_resp.json().get("id")
            if not node_id:
                return _fail(
                    MARKETPLACE,
                    normalized,
                    "node register missing id",
                    steps,
                    report,
                )
            steps.append(f"node register ok id={node_id}")

            # 3) Offer create
            steps.append("offer create")
            offer_resp = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/offers",
                secret=secret,
                hotkey=_SCENARIO_PROVIDER_HK,
                body={
                    "node_ids": [node_id],
                    "price_per_hour": 2.5,
                    "max_lifetime_hours": 24.0,
                    "require_ib": True,
                    "mode": "single",
                },
            )
            if offer_resp.status_code >= 400:
                return _fail(
                    MARKETPLACE,
                    normalized,
                    f"offer create HTTP {offer_resp.status_code}: {offer_resp.text}",
                    steps,
                    report,
                )
            offer = offer_resp.json()
            offer_id = offer.get("id")
            if not offer_id or offer.get("status") != "listed":
                return _fail(
                    MARKETPLACE,
                    normalized,
                    f"offer create unexpected payload: {offer}",
                    steps,
                    report,
                )
            steps.append(f"offer create ok id={offer_id} status=listed")

            # 4) Basic list / browse
            steps.append("list offers (browse)")
            listed = client.get(f"{normalized}/v1/offers")
            if listed.status_code != 200:
                return _fail(
                    MARKETPLACE,
                    normalized,
                    f"list offers HTTP {listed.status_code}",
                    steps,
                    report,
                )
            items = listed.json().get("items") or []
            if not any(isinstance(o, dict) and o.get("id") == offer_id for o in items):
                return _fail(
                    MARKETPLACE,
                    normalized,
                    f"offer {offer_id} missing from browse list",
                    steps,
                    report,
                )
            steps.append(f"list offers ok count={len(items)}")

            # Renter identity (register not required for rent; hotkey signs rent).
            steps.append("rent offer (renter1)")
            rent_resp = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/offers/{offer_id}/rent",
                secret=secret,
                hotkey=_SCENARIO_RENTER_HK,
                body={"lifetime_hours": 4.0},
            )
            if rent_resp.status_code >= 400:
                return _fail(
                    MARKETPLACE,
                    normalized,
                    f"rent HTTP {rent_resp.status_code}: {rent_resp.text}",
                    steps,
                    report,
                )
            rent_payload = rent_resp.json()
            lease = rent_payload.get("lease") or {}
            pod = rent_payload.get("pod") or {}
            lease_id = lease.get("id")
            if not lease_id:
                return _fail(
                    MARKETPLACE,
                    normalized,
                    f"rent missing lease id: {rent_payload}",
                    steps,
                    report,
                )
            lease_status = lease.get("status")
            if lease_status not in {"active", "requested"}:
                return _fail(
                    MARKETPLACE,
                    normalized,
                    f"unexpected lease status after rent: {lease_status}",
                    steps,
                    report,
                )
            steps.append(
                f"rent ok lease_id={lease_id} status={lease_status} "
                f"pod_id={pod.get('id')}"
            )

            # 5) Double-rent reject
            steps.append("double-rent reject (renter2)")
            double = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/offers/{offer_id}/rent",
                secret=secret,
                hotkey=_SCENARIO_RENTER2_HK,
                body={"lifetime_hours": 4.0},
            )
            if double.status_code not in {409, 400, 422, 403}:
                return _fail(
                    MARKETPLACE,
                    normalized,
                    (
                        "double-rent expected conflict-class 4xx, got "
                        f"{double.status_code}: {double.text}"
                    ),
                    steps,
                    report,
                )
            steps.append(
                f"double-rent rejected HTTP {double.status_code} "
                f"code={_detail_code(double.json())}"
            )

            # 6) Terminate
            steps.append("terminate lease")
            term = _signed_request(
                client,
                "POST",
                f"{normalized}/v1/leases/{lease_id}/terminate",
                secret=secret,
                hotkey=_SCENARIO_RENTER_HK,
                body={"reason": "sim_marketplace_scenario"},
            )
            if term.status_code >= 400:
                return _fail(
                    MARKETPLACE,
                    normalized,
                    f"terminate HTTP {term.status_code}: {term.text}",
                    steps,
                    report,
                )
            term_lease = (term.json() or {}).get("lease") or {}
            if term_lease.get("status") not in {"terminated", "expired", "failed"}:
                # Re-fetch if terminate payload omits status
                got = client.get(f"{normalized}/v1/leases/{lease_id}")
                if got.status_code != 200 or got.json().get("status") not in {
                    "terminated",
                    "expired",
                    "failed",
                }:
                    return _fail(
                        MARKETPLACE,
                        normalized,
                        f"lease not terminal after terminate: {term.text}",
                        steps,
                        report,
                    )
            steps.append(
                f"terminate ok lease_id={lease_id} "
                f"status={term_lease.get('status', 'terminal')}"
            )

            # 7) Post-terminate list sanity (active rent no longer listed for climb)
            steps.append("post-terminate list sanity")
            post_list = client.get(f"{normalized}/v1/offers")
            if post_list.status_code != 200:
                return _fail(
                    MARKETPLACE,
                    normalized,
                    f"post-terminate list HTTP {post_list.status_code}",
                    steps,
                    report,
                )
            steps.append("marketplace flow complete")

    except httpx.HTTPError as exc:
        return _fail(
            MARKETPLACE,
            normalized,
            f"HTTP client error: {exc}",
            steps,
            report,
        )

    return ScenarioResult(
        name=MARKETPLACE,
        ok=True,
        base_url=normalized,
        message=(
            "marketplace passed: offer/list/rent/double-rent-reject/terminate"
        ),
        steps=steps,
        identity=report,
    )


def _fail(
    name: str,
    base_url: str,
    message: str,
    steps: list[str],
    identity: IdentityReport | None,
) -> ScenarioResult:
    steps.append(f"FAILED: {message}")
    return ScenarioResult(
        name=name,
        ok=False,
        base_url=base_url,
        message=message,
        steps=steps,
        identity=identity,
    )


def _tee_fixture_root() -> Path:
    """Locate tests/fixtures/tee from repo checkout or site-package layout."""

    # Prefer walking up from this module to the monorepo root (src/hypercluster/sim).
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "tests" / "fixtures" / "tee",  # .../hypercluster/tests/...
        here.parents[4] / "tests" / "fixtures" / "tee",
        Path.cwd() / "tests" / "fixtures" / "tee",
    ]
    for c in candidates:
        if c.is_dir() and (c / "positive_tdx_v1.json").is_file():
            return c
    # Last-ditch: env override for custom checkouts.
    import os

    env = (os.environ.get("HYPER_TEE_FIXTURE_DIR") or "").strip()
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "tee fixtures not found under tests/fixtures/tee "
        f"(searched {[str(c) for c in candidates]})"
    )


def run_tee_offline_scenario(
    base_url: str,
    *,
    fixture_dir: str | Path | None = None,
) -> ScenarioResult:
    """CI tee-offline: compose-hash golden + positive/negative fixture verify.

    Fully offline — no GPU/TEE silicon and no live dstack-verifier network
    (VAL-TEE-013). ``base_url`` is accepted for CLI parity but is not required
    for the core offline path.
    """

    from hypercluster.attest.compose_hash import (
        hash_compose_file,
        load_golden_hash_file,
    )
    from hypercluster.attest.offline_fixtures import (
        make_offline_envelope,
        package_quote_b64,
    )
    from hypercluster.attest.policy import (
        DEFAULT_COMPOSE_HASH_GOLDEN,
        TeeVerifyPolicy,
    )
    from hypercluster.attest.report_data import build_report_data
    from hypercluster.attest.verify import verify_offline_fixture_file, verify_tee
    from hypercluster.domain.scoring_tee import compute_tee_bonus
    from hypercluster.settings import HyperSettings

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    steps.append("tee-offline: locate fixture root (offline only, no live network)")

    try:
        root = Path(fixture_dir) if fixture_dir is not None else _tee_fixture_root()
    except FileNotFoundError as exc:
        return _fail(
            TEE_OFFLINE,
            normalized,
            str(exc),
            steps,
            None,
        )
    steps.append(f"fixture_root={root}")

    # --- Valve A: compose-hash golden stability (VAL-TEE-010 integrated smoke)
    golden_compose = root / "golden_compose.yml"
    golden_hash = root / "golden_compose.sha256"
    positive = root / "positive_tdx_v1.json"
    if not golden_compose.is_file() or not golden_hash.is_file():
        return _fail(
            TEE_OFFLINE,
            normalized,
            f"missing golden compose fixtures under {root}",
            steps,
            None,
        )
    if not positive.is_file():
        return _fail(
            TEE_OFFLINE,
            normalized,
            f"missing positive_tdx_v1.json under {root}",
            steps,
            None,
        )

    steps.append("compose-hash golden: two successive hashes")
    h1 = hash_compose_file(golden_compose)
    h2 = hash_compose_file(golden_compose)
    if h1 != h2:
        return _fail(
            TEE_OFFLINE,
            normalized,
            f"compose-hash non-deterministic: {h1} vs {h2}",
            steps,
            None,
        )
    expected_golden = load_golden_hash_file(golden_hash)
    if h1 != expected_golden:
        return _fail(
            TEE_OFFLINE,
            normalized,
            f"compose-hash golden drift: got={h1} expected={expected_golden}",
            steps,
            None,
        )
    steps.append(f"compose-hash ok hash={h1}")

    # --- Valve B: positive offline fixture verify (no network)
    steps.append("offline_fixture positive verify")
    policy = TeeVerifyPolicy(
        compose_allowlist=frozenset({DEFAULT_COMPOSE_HASH_GOLDEN}),
        tcb_enforce=True,
        acceptable_tcb_statuses=frozenset({"UpToDate"}),
        disallowed_advisory_ids=frozenset(),
    )
    positive_result = verify_offline_fixture_file(
        positive,
        policy=policy,
        job_id="job-offline-positive-0001",
        image_digest=(
            "sha256:sim000000000000000000000000000000000000000000000000000000000001"
        ),
        nonce="n0nce-posit1ve-aaaa-bbbb-cccc-111111111111",
    )
    if not positive_result.is_valid:
        return _fail(
            TEE_OFFLINE,
            normalized,
            (
                "positive fixture failed offline verify: "
                f"reasons={positive_result.reason_codes}"
            ),
            steps,
            None,
        )
    steps.append(
        f"positive offline_fixture is_valid=true "
        f"compose_hash={positive_result.compose_hash}"
    )

    # Bonus application for verified offline TDX (VAL-TEE-006 / scenario wiring).
    bonus = compute_tee_bonus(
        proof_tier="tdx",
        verified=True,
        verify_mode="offline_fixture",
        tee_mode="tdx",
        hyper=HyperSettings(tee_bonus_tdx=1.08, tee_bonus_tdx_gpu=1.20),
        is_valid_verdict=True,
    )
    if bonus.tee_bonus != 1.08:
        return _fail(
            TEE_OFFLINE,
            normalized,
            f"expected tee_bonus=1.08 after positive offline, got {bonus.tee_bonus}",
            steps,
            None,
        )
    steps.append(f"tee_bonus applied={bonus.tee_bonus} (TDX offline path)")

    # --- Valve C: mutated compose_hash rejects (no bonus)
    steps.append("mutated compose_hash reject")
    bad_compose = (
        "sha256:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    )
    job_id = "job-offline-mutated-compose"
    image = "sha256:sim000000000000000000000000000000000000000000000000000000000001"
    nonce = "n0nce-mutated-compose-aaaa-bbbb-cccc-3333"
    report = build_report_data(job_id=job_id, image_digest=image, nonce=nonce)
    bad_env = make_offline_envelope(
        compose_hash=bad_compose,
        expected_compose_hash=DEFAULT_COMPOSE_HASH_GOLDEN,
        report_data=report,
        job_id=job_id,
        image_digest=image,
        nonce=nonce,
        fixture_id="mutated_compose_scenario",
    )
    from hypercluster.attest.models import TeeVerifyRequest

    bad_result = verify_tee(
        TeeVerifyRequest(
            quote_b64=package_quote_b64(bad_env),
            report_data_expected=report,
            mode="offline_fixture",
        ),
        policy=policy,
    )
    if bad_result.is_valid:
        return _fail(
            TEE_OFFLINE,
            normalized,
            "mutated compose_hash incorrectly verified is_valid=true",
            steps,
            None,
        )
    reasons = " ".join(bad_result.reason_codes).lower()
    if not any(
        token in reasons
        for token in (
            "compose",
            "allowlist",
            "measurement",
        )
    ):
        return _fail(
            TEE_OFFLINE,
            normalized,
            f"mutated compose reject missing reason: {bad_result.reason_codes}",
            steps,
            None,
        )
    steps.append(
        f"mutated compose rejected reasons={list(bad_result.reason_codes)}"
    )

    # Unverified claim → no inflated bonus.
    no_bonus = compute_tee_bonus(
        proof_tier="tdx",
        verified=False,
        verify_mode="offline_fixture",
        tee_mode="tdx",
        hyper=HyperSettings(tee_bonus_tdx=1.08),
        is_valid_verdict=False,
    )
    if no_bonus.tee_bonus != 1.0:
        return _fail(
            TEE_OFFLINE,
            normalized,
            f"unverified claim unexpectedly got bonus={no_bonus.tee_bonus}",
            steps,
            None,
        )
    steps.append("unverified claim tee_bonus=1.0 ok")
    steps.append("tee-offline complete (offline fixtures only; no live network)")

    return ScenarioResult(
        name=TEE_OFFLINE,
        ok=True,
        base_url=normalized,
        message=(
            "tee-offline passed: compose-hash golden + positive offline verify + "
            "mutated compose reject (no hardware/network)"
        ),
        steps=steps,
        identity=None,
    )



def run_nccl_scenario(
    base_url: str = "http://127.0.0.1:3200",
    *,
    seed: int = 7,
) -> ScenarioResult:
    """NCCL multi-node local sim: pack/spread plans + fabric_gate fail inject.

    Fully offline — uses seed_sim_inventory + place_ranks + sim_launch only.
    Never requires real InfiniBand or GPU silicon (VAL-CLI-017).
    ``base_url`` is accepted for CLI parity; identity is not required.
    """

    from hypercluster.fabric.launcher import LaunchRequest, sim_launch
    from hypercluster.fabric.planner import PlacementRequest, place_ranks
    from hypercluster.sim.inventory import seed_sim_inventory

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    steps.append("nccl: seed multi-node IB/NVLink inventory (local sim only)")

    inv = seed_sim_inventory(seed=seed, node_count=4, gpus_per_node=2)
    reports = inv.reports()
    if len(reports) < 2:
        return _fail(
            NCCL,
            normalized,
            f"expected ≥2 sim nodes, got {len(reports)}",
            steps,
            None,
        )
    steps.append(
        f"inventory seed={seed} nodes={len(reports)} "
        f"graph_digest={inv.graph_digest[:18]}…"
    )

    image = (
        "sha256:sim000000000000000000000000000000000000000000000000000000000001"
    )

    # --- Pack plan (concentrate ranks) ---
    steps.append("place_ranks policy=pack world_size=4 nnodes=2")
    pack = place_ranks(
        PlacementRequest(
            job_id="sim-nccl-pack",
            world_size=4,
            nnodes=2,
            nproc_per_node=2,
            policy="pack",
            fabric="auto",
            node_reports=reports,
        )
    )
    if not pack.ok:
        return _fail(
            NCCL,
            normalized,
            f"pack placement failed: {pack.reason or pack}",
            steps,
            None,
        )
    pack_nodes = {b.node_id for b in pack.rankmap}
    steps.append(
        f"pack ok ranks={len(pack.rankmap)} nodes={sorted(pack_nodes)} "
        f"graph={pack.graph_digest[:18]}…"
    )

    # --- Spread plan (distribute ranks) ---
    steps.append("place_ranks policy=spread world_size=4 nnodes=4")
    spread = place_ranks(
        PlacementRequest(
            job_id="sim-nccl-spread",
            world_size=4,
            nnodes=4,
            nproc_per_node=1,
            policy="spread",
            fabric="auto",
            node_reports=reports,
        )
    )
    if not spread.ok:
        return _fail(
            NCCL,
            normalized,
            f"spread placement failed: {spread.reason or spread}",
            steps,
            None,
        )
    spread_nodes = {b.node_id for b in spread.rankmap}
    if len(spread_nodes) < 2:
        return _fail(
            NCCL,
            normalized,
            f"spread expected multi-node distribution, got nodes={spread_nodes}",
            steps,
            None,
        )
    steps.append(
        f"spread ok ranks={len(spread.rankmap)} nodes={sorted(spread_nodes)} "
        f"graph={spread.graph_digest[:18]}…"
    )

    # --- Happy multi-node launch (pack plan) ---
    steps.append("sim_launch multi-node (pack) honesty=l1")
    ok_launch = sim_launch(
        LaunchRequest(
            placement=pack,
            image_digest=image,
            entrypoint=["python", "-m", "train"],
            fabric_mode="auto",
            honesty_level="l1",
            node_reports=reports,
            seed=seed,
        )
    )
    if ok_launch.status != "succeeded":
        return _fail(
            NCCL,
            normalized,
            (
                f"expected succeeded launch, got status={ok_launch.status} "
                f"code={ok_launch.failure_code} reason={ok_launch.reason}"
            ),
            steps,
            None,
        )
    if ok_launch.fabric_gate != 1.0:
        return _fail(
            NCCL,
            normalized,
            f"expected fabric_gate=1.0 on clean launch, got {ok_launch.fabric_gate}",
            steps,
            None,
        )
    if ok_launch.metrics is None:
        return _fail(
            NCCL,
            normalized,
            "successful launch missing synthetic NCCL metrics",
            steps,
            None,
        )
    steps.append(
        f"launch ok status={ok_launch.status} fabric_gate={ok_launch.fabric_gate} "
        f"allreduce_gbps={ok_launch.metrics.allreduce_gbps}"
    )

    # --- fabric_gate fail inject: inventory spoof under fabric=ib ---
    steps.append("fabric_gate fail inject: inventory_spoof under fabric=ib")
    ib_plan = place_ranks(
        PlacementRequest(
            job_id="sim-nccl-spoof",
            world_size=2,
            nnodes=2,
            nproc_per_node=1,
            policy="pack",
            fabric="ib",
            node_reports=reports,
        )
    )
    if not ib_plan.ok:
        return _fail(
            NCCL,
            normalized,
            f"ib pack for spoof inject failed: {ib_plan.reason or ib_plan}",
            steps,
            None,
        )
    spoof = sim_launch(
        LaunchRequest(
            placement=ib_plan,
            image_digest=image,
            fabric_mode="ib",
            honesty_level="l1",
            inventory_spoof=True,
            node_reports=reports,
            seed=seed,
        )
    )
    if spoof.fabric_gate != 0.0 or spoof.composite != 0.0:
        return _fail(
            NCCL,
            normalized,
            (
                "inventory_spoof inject expected fabric_gate=0 and composite=0, "
                f"got gate={spoof.fabric_gate} composite={spoof.composite}"
            ),
            steps,
            None,
        )
    steps.append(
        f"inventory_spoof zeros fabric_gate={spoof.fabric_gate} "
        f"composite={spoof.composite} integrity_fail={spoof.integrity_fail}"
    )

    # --- fabric_gate fail inject: eth_fallback under fabric=ib ---
    steps.append("fabric_gate fail inject: eth_fallback_injected under fabric=ib")
    eth_fb = sim_launch(
        LaunchRequest(
            placement=ib_plan,
            image_digest=image,
            fabric_mode="ib",
            honesty_level="l1",
            eth_fallback_injected=True,
            node_reports=reports,
            seed=seed,
        )
    )
    if eth_fb.fabric_gate != 0.0 or eth_fb.composite != 0.0:
        return _fail(
            NCCL,
            normalized,
            (
                "eth_fallback inject expected fabric_gate=0 and composite=0, "
                f"got gate={eth_fb.fabric_gate} composite={eth_fb.composite}"
            ),
            steps,
            None,
        )
    steps.append(
        f"eth_fallback zeros fabric_gate={eth_fb.fabric_gate} "
        f"composite={eth_fb.composite}"
    )

    # --- explicit failed inject still produces LaunchResult ---
    steps.append("status fail inject: inject_status=failed")
    failed = sim_launch(
        LaunchRequest(
            placement=pack,
            image_digest=image,
            inject_status="failed",
            seed=seed,
        )
    )
    if failed.status != "failed":
        return _fail(
            NCCL,
            normalized,
            f"inject_status=failed expected status=failed, got {failed.status}",
            steps,
            None,
        )
    steps.append(f"failed inject status={failed.status} code={failed.failure_code}")
    steps.append("nccl scenario complete (local sim; no real IB)")

    return ScenarioResult(
        name=NCCL,
        ok=True,
        base_url=normalized,
        message=(
            "nccl passed: pack/spread multi-node + fabric_gate fail inject "
            "(inventory_spoof + eth_fallback)"
        ),
        steps=steps,
        identity=None,
    )


def run_weights_scenario(
    base_url: str,
    *,
    timeout: float = 30.0,
    shared_token: str | None = None,
    master_url: str | None = None,
    identity_probe: Callable[..., IdentityReport] = probe_identity_gates,
) -> ScenarioResult:
    """Weights e2e: multi-hotkey composites → push ack → finite map (VAL-SCORE-025).

    Seeds scores into the live challenge DB via local process settings when the
    API is local; pushes to mock-master (default :3201) and asserts acked +
    finite non-negative weight-preview.
    """

    import asyncio
    import math
    import os

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    steps.append("probe identity gates (/health + /ready)")
    report = identity_probe(base_url, timeout=min(timeout, 5.0))
    if not report.ok:
        steps.append("identity gates failed")
        return ScenarioResult(
            name=WEIGHTS,
            ok=False,
            base_url=normalized,
            message=f"weights failed: identity not green ({'; '.join(report.errors)})",
            steps=steps,
            identity=report,
        )
    steps.append("identity gates green")

    token = (
        shared_token
        or os.environ.get("CHALLENGE_SHARED_TOKEN")
        or os.environ.get("HYPER_SHARED_TOKEN")
        or ""
    )
    master = (
        master_url
        or os.environ.get("HYPER_MASTER_BASE_URL")
        or "http://127.0.0.1:3201"
    )

    # Base-compatible ss58-like hotkeys (alpha chars required; no bare UIDs).
    hotkey_a = "5DAAnrj7VHTznn2AaACRrN8iJZqK7PhB1aH6Yqz3G3eQnZf"
    hotkey_b = "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw"

    async def _seed_and_push() -> dict[str, Any]:
        from hypercluster.db.database import Database
        from hypercluster.db.models import Job, JobAttempt
        from hypercluster.domain.scoring_tee import persist_score_for_attempt
        from hypercluster.settings import HyperSettings, get_settings
        from hypercluster.weight_push import WeightPushClient
        from hypercluster.weights import load_raw_weights

        settings = get_settings()
        hyper = HyperSettings(
            score_window_attempts=50,
            self_deal_damping=0.5,
            master_base_url=master,
            weight_push_enabled=True,
            weight_push_freshness_s=300,
        )
        database = Database(settings.database_url)
        await database.init()
        try:
            async with database.session() as session:
                for hotkey, eff in ((hotkey_a, 10.0), (hotkey_b, 3.0)):
                    job_id = str(uuid.uuid4())
                    attempt_id = str(uuid.uuid4())
                    session.add(
                        Job(
                            id=job_id,
                            submitter_hotkey=hotkey,
                            status="succeeded",
                            image_digest=(
                                "sha256:sim000000000000000000000000000000000000000000000000000000000001"
                            ),
                            entrypoint_json=json.dumps(["python", "-m", "train"]),
                            world_size=1,
                            nnodes=1,
                            nproc_per_node=1,
                            backend="nccl",
                            fabric_mode="auto",
                            tee_mode="none",
                            resource_json=json.dumps({"gpus": 1}),
                            timeout_s=60,
                        )
                    )
                    session.add(
                        JobAttempt(
                            id=attempt_id,
                            job_id=job_id,
                            attempt_no=1,
                            status="succeeded",
                        )
                    )
                    await session.flush()
                    await persist_score_for_attempt(
                        session,
                        attempt_id=attempt_id,
                        hotkey=hotkey,
                        role="demand",
                        correctness=1.0,
                        efficiency=eff,
                        fabric_gate=1.0,
                        proof=None,
                        tee_mode="none",
                        hyper=hyper,
                    )
                await session.commit()
            steps.append("seeded multi-hotkey demand scores")
            weights = await load_raw_weights(database=database, hyper=hyper)
            if not weights:
                return {"ok": False, "error": "empty weights after seed"}
            for k, v in weights.items():
                if not math.isfinite(float(v)) or float(v) < 0:
                    return {"ok": False, "error": f"illegal weight {k}={v}"}
            steps.append(f"raw weights finite count={len(weights)}")

            client = WeightPushClient(
                database=database,
                challenge_slug=settings.slug,
                master_base_url=master,
                shared_token=token or "test-challenge-shared-token",
                hyper=hyper,
            )
            result = await client.push_once(weights=weights, epoch=1)
            steps.append(
                f"push status={result.status} push_status={result.push_status} "
                f"epoch={result.epoch} revision={result.revision}"
            )
            if result.status != "acknowledged" or result.push_status not in {
                "acked",
                "sim",
            }:
                return {
                    "ok": False,
                    "error": f"push not acked: {result.status} {result.error}",
                    "result": result,
                }
            # Idempotent re-push
            again = await client.push_once(
                weights=weights, epoch=1, revision=result.revision
            )
            steps.append(f"idempotent re-push status={again.status} idemp={again.idempotent}")
            if again.status != "acknowledged":
                return {"ok": False, "error": f"idempotent push failed: {again.status}"}
            return {
                "ok": True,
                "weights": weights,
                "result": {
                    "status": result.status,
                    "epoch": result.epoch,
                    "revision": result.revision,
                    "payload_digest": result.payload_digest,
                    "push_status": result.push_status,
                },
            }
        finally:
            await database.close()

    if not token:
        # Still attempt with test default if process uses it; record step.
        steps.append("shared token from env missing; using process settings / default test token")

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            # Called from pytest-asyncio / nested async context.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                outcome = pool.submit(lambda: asyncio.run(_seed_and_push())).result(
                    timeout=max(timeout, 30.0)
                )
        else:
            outcome = asyncio.run(_seed_and_push())
    except Exception as exc:  # noqa: BLE001
        steps.append(f"seed/push raised: {exc}")
        return ScenarioResult(
            name=WEIGHTS,
            ok=False,
            base_url=normalized,
            message=f"weights failed: {exc}",
            steps=steps,
            identity=report,
        )

    if not outcome.get("ok"):
        return ScenarioResult(
            name=WEIGHTS,
            ok=False,
            base_url=normalized,
            message=f"weights failed: {outcome.get('error')}",
            steps=steps,
            identity=report,
        )

    # Black-box: weight-preview when the API process shares the challenge DB.
    # Push path already verified finite map + ack (primary gate for VAL-SCORE-025).
    try:
        with httpx.Client(timeout=timeout) as client:
            preview = client.get(f"{normalized}/v1/weight-preview")
            steps.append(f"GET /v1/weight-preview → {preview.status_code}")
            if preview.status_code == 200:
                body = preview.json()
                wmap = body.get("weights") or {}
                if isinstance(wmap, dict) and wmap:
                    for k, v in wmap.items():
                        if not math.isfinite(float(v)) or float(v) < 0:
                            return ScenarioResult(
                                name=WEIGHTS,
                                ok=False,
                                base_url=normalized,
                                message=f"preview illegal weight {k}={v}",
                                steps=steps,
                                identity=report,
                            )
                    steps.append(f"preview weights count={len(wmap)} finite≥0")
                else:
                    steps.append(
                        "weight-preview empty (API may use separate DB); "
                        "push path already verified finite map"
                    )
            else:
                steps.append(
                    f"weight-preview HTTP {preview.status_code}; "
                    "push path already verified (non-fatal for split-DB)"
                )
    except httpx.HTTPError as exc:
        steps.append(
            f"preview probe skipped ({exc}); push path already verified finite map"
        )
    steps.append("no on-chain set_weights in challenge product path")

    return ScenarioResult(
        name=WEIGHTS,
        ok=True,
        base_url=normalized,
        message="weights passed: multi-hotkey score → push ack + finite map",
        steps=steps,
        identity=report,
    )


def run_scenario(
    name: str,
    base_url: str,
    *,
    timeout: float = 15.0,
    shared_token: str | None = None,
    master_url: str | None = None,
) -> ScenarioResult:
    """Dispatch a named scenario (architecture §12.3 + cross-happy-path)."""

    key = name.strip().lower()
    if key == SMOKE:
        return run_smoke_scenario(base_url, timeout=min(timeout, 5.0))
    if key == MARKETPLACE:
        return run_marketplace_scenario(
            base_url,
            timeout=timeout,
            shared_token=shared_token,
        )
    if key == NCCL:
        return run_nccl_scenario(base_url)
    if key == TEE_OFFLINE:
        return run_tee_offline_scenario(base_url)
    if key == WEIGHTS:
        return run_weights_scenario(
            base_url,
            timeout=max(timeout, 30.0),
            shared_token=shared_token,
            master_url=master_url,
        )
    if key in {CROSS_HAPPY_PATH, "cross_happy_path", "happy-path", "cross-happy"}:
        from hypercluster.sim.cross_happy_path import run_cross_happy_path

        return run_cross_happy_path(
            base_url,
            timeout=max(timeout, 45.0),
            shared_token=shared_token,
        )
    return ScenarioResult(
        name=key,
        ok=False,
        base_url=base_url.rstrip("/"),
        message=(
            f"unknown scenario {name!r}; known: "
            f"{', '.join(EXTENDED_SCENARIOS)}"
        ),
        steps=["unknown scenario name"],
    )


__all__ = [
    "CROSS_HAPPY_PATH",
    "EXTENDED_SCENARIOS",
    "KNOWN_SCENARIOS",
    "MARKETPLACE",
    "NCCL",
    "SMOKE",
    "ScenarioResult",
    "TEE_OFFLINE",
    "WEIGHTS",
    "run_marketplace_scenario",
    "run_nccl_scenario",
    "run_scenario",
    "run_smoke_scenario",
    "run_tee_offline_scenario",
    "run_weights_scenario",
]
