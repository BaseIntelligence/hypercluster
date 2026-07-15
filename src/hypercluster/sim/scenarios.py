"""Local CI scenario stubs for M1 scaffold (smoke only fully gated).

Architecture §12.3 names: smoke, marketplace, nccl, tee-offline, weights.
M1 implements smoke with identity gates; others return not-implemented fail.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from hypercluster.sim.identity import IdentityReport, probe_identity_gates

SMOKE = "smoke"
MARKETPLACE = "marketplace"
NCCL = "nccl"
TEE_OFFLINE = "tee-offline"
WEIGHTS = "weights"

KNOWN_SCENARIOS = (SMOKE, MARKETPLACE, NCCL, TEE_OFFLINE, WEIGHTS)


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


def run_scenario(
    name: str,
    base_url: str,
    *,
    timeout: float = 5.0,
) -> ScenarioResult:
    """Dispatch a named scenario (M1: smoke fully; others not yet implemented)."""

    key = name.strip().lower()
    if key == SMOKE:
        return run_smoke_scenario(base_url, timeout=timeout)
    if key in {MARKETPLACE, NCCL, TEE_OFFLINE, WEIGHTS}:
        return ScenarioResult(
            name=key,
            ok=False,
            base_url=base_url.rstrip("/"),
            message=f"scenario {key!r} not implemented in M1 scaffold (identity-gated later)",
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
    "run_scenario",
    "run_smoke_scenario",
]
