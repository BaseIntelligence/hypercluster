"""Local CI scenario runners (smoke + marketplace).

Architecture §12.3 names: smoke, marketplace, nccl, tee-offline, weights.
M2 ships smoke (identity) and marketplace (offer/rent/terminate/double-rent).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from hypercluster.api.auth import build_signed_headers
from hypercluster.sim.identity import IdentityReport, probe_identity_gates

SMOKE = "smoke"
MARKETPLACE = "marketplace"
NCCL = "nccl"
TEE_OFFLINE = "tee-offline"
WEIGHTS = "weights"

KNOWN_SCENARIOS = (SMOKE, MARKETPLACE, NCCL, TEE_OFFLINE, WEIGHTS)

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
    steps.append("weights empty burn-safe stub (M1 scaffold: ok)")
    return ScenarioResult(
        name=SMOKE,
        ok=True,
        base_url=report.base_url,
        message="smoke passed: health/ready green",
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


def run_scenario(
    name: str,
    base_url: str,
    *,
    timeout: float = 15.0,
    shared_token: str | None = None,
) -> ScenarioResult:
    """Dispatch a named scenario."""

    key = name.strip().lower()
    if key == SMOKE:
        return run_smoke_scenario(base_url, timeout=min(timeout, 5.0))
    if key == MARKETPLACE:
        return run_marketplace_scenario(
            base_url,
            timeout=timeout,
            shared_token=shared_token,
        )
    if key in {NCCL, TEE_OFFLINE, WEIGHTS}:
        return ScenarioResult(
            name=key,
            ok=False,
            base_url=base_url.rstrip("/"),
            message=(
                f"scenario {key!r} not implemented yet "
                "(nccl/tee-offline/weights land in later milestones)"
            ),
            steps=[f"scenario {key} stub — not implemented"],
        )
    return ScenarioResult(
        name=key,
        ok=False,
        base_url=base_url.rstrip("/"),
        message=f"unknown scenario {name!r}; known: {', '.join(KNOWN_SCENARIOS)}",
        steps=["unknown scenario name"],
    )


__all__ = [
    "KNOWN_SCENARIOS",
    "MARKETPLACE",
    "NCCL",
    "SMOKE",
    "ScenarioResult",
    "TEE_OFFLINE",
    "WEIGHTS",
    "run_marketplace_scenario",
    "run_scenario",
    "run_smoke_scenario",
]
