"""Reusable scenario-suite orchestration for local sim e2e (M7+ cross flows).

Architecture §12.3 scenario order:
  smoke → marketplace → nccl → tee-offline → weights

Cross-area e2e features (VAL-CROSS-*) can import :func:`run_scenario_suite` and
:func:`run_named_scenarios` instead of re-wiring each scenario.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from hypercluster.sim.scenarios import (
    KNOWN_SCENARIOS,
    MARKETPLACE,
    NCCL,
    SMOKE,
    TEE_OFFLINE,
    WEIGHTS,
    ScenarioResult,
    run_scenario,
)

# Architecture §12.3 order — keep stable for cross-suite reproducibility.
DEFAULT_SCENARIO_ORDER: tuple[str, ...] = (
    SMOKE,
    MARKETPLACE,
    NCCL,
    TEE_OFFLINE,
    WEIGHTS,
)

assert tuple(DEFAULT_SCENARIO_ORDER) == tuple(KNOWN_SCENARIOS)


@dataclass(slots=True)
class SuiteResult:
    """Aggregate outcome for a multi-scenario suite run."""

    ok: bool
    base_url: str
    message: str
    results: list[ScenarioResult] = field(default_factory=list)
    order: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"suite result={status}",
            f"base_url={self.base_url}",
            f"message={self.message}",
            f"order={','.join(self.order)}",
        ]
        for r in self.results:
            tag = "ok" if r.ok else "FAIL"
            lines.append(f"  [{tag}] {r.name}: {r.message}")
        return lines


def normalize_scenario_names(
    names: Sequence[str] | Iterable[str] | None = None,
) -> list[str]:
    """Normalize and validate scenario names (lower-case, known-only)."""

    if names is None:
        return list(DEFAULT_SCENARIO_ORDER)
    out: list[str] = []
    known = {n.lower() for n in KNOWN_SCENARIOS}
    for raw in names:
        key = str(raw).strip().lower()
        if not key:
            continue
        if key not in known:
            raise ValueError(
                f"unknown scenario {raw!r}; known: {', '.join(KNOWN_SCENARIOS)}"
            )
        if key not in out:
            out.append(key)
    if not out:
        return list(DEFAULT_SCENARIO_ORDER)
    return out


def run_named_scenarios(
    names: Sequence[str] | Iterable[str] | None,
    base_url: str,
    *,
    timeout: float = 30.0,
    shared_token: str | None = None,
    master_url: str | None = None,
    stop_on_fail: bool = True,
) -> SuiteResult:
    """Run the given (or default) scenarios in order.

    Parameters
    ----------
    stop_on_fail:
        When True (default), stop after the first failing scenario so later
        cross-feature suites can fail closed early.
    """

    order = normalize_scenario_names(names)
    results: list[ScenarioResult] = []
    normalized = base_url.rstrip("/")

    for name in order:
        # Per-scenario timeout preferences (weights is heavier).
        sc_timeout = max(timeout, 30.0) if name == WEIGHTS else timeout
        if name in {NCCL, TEE_OFFLINE}:
            # Offline-core scenarios do not require live identity, but still
            # accept base_url for CLI parity.
            result = run_scenario(
                name,
                normalized,
                timeout=sc_timeout,
                shared_token=shared_token,
                master_url=master_url,
            )
        else:
            result = run_scenario(
                name,
                normalized,
                timeout=sc_timeout,
                shared_token=shared_token,
                master_url=master_url,
            )
        results.append(result)
        if not result.ok and stop_on_fail:
            failed = [r.name for r in results if not r.ok]
            return SuiteResult(
                ok=False,
                base_url=normalized,
                message=f"suite failed at {result.name}: {result.message}",
                results=results,
                order=order,
            )

    failed = [r.name for r in results if not r.ok]
    ok = not failed
    return SuiteResult(
        ok=ok,
        base_url=normalized,
        message=(
            f"suite passed: {len(results)} scenarios green"
            if ok
            else f"suite failed: {', '.join(failed)}"
        ),
        results=results,
        order=order,
    )


def run_scenario_suite(
    base_url: str,
    *,
    timeout: float = 30.0,
    shared_token: str | None = None,
    master_url: str | None = None,
    stop_on_fail: bool = True,
) -> SuiteResult:
    """Architecture §12.3 full suite in canonical order."""

    return run_named_scenarios(
        DEFAULT_SCENARIO_ORDER,
        base_url,
        timeout=timeout,
        shared_token=shared_token,
        master_url=master_url,
        stop_on_fail=stop_on_fail,
    )


# Smaller reusable bundles for cross e2e features.
HAPPY_PATH_SCENARIOS: tuple[str, ...] = (SMOKE, MARKETPLACE, WEIGHTS)
FABRIC_SCENARIOS: tuple[str, ...] = (SMOKE, NCCL, TEE_OFFLINE)


def run_happy_path_bundle(
    base_url: str,
    **kwargs: Any,
) -> SuiteResult:
    """Smoke + marketplace + weights — usable as a cross-e2e preflight."""

    return run_named_scenarios(HAPPY_PATH_SCENARIOS, base_url, **kwargs)


def run_fabric_bundle(
    base_url: str,
    **kwargs: Any,
) -> SuiteResult:
    """Smoke + nccl + tee-offline fabric/TEE bundle."""

    return run_named_scenarios(FABRIC_SCENARIOS, base_url, **kwargs)


def run_cross_happy_path_bundle(
    base_url: str,
    *,
    timeout: float = 45.0,
    shared_token: str | None = None,
    **kwargs: Any,
) -> ScenarioResult:
    """VAL-CROSS-001..003/008/009/013 full chain under pure local sim.

    Prefer this over composing suite pieces when the assertion needs threaded
    marketplace→job→score IDs and egress cleanliness in one process.
    """

    from hypercluster.sim.cross_happy_path import run_cross_happy_path

    _ = kwargs  # recipients for future suite options (master_url unused here)
    return run_cross_happy_path(
        base_url,
        timeout=timeout,
        shared_token=shared_token,
    )


def run_cross_multinode_fabric_tee_bundle(
    base_url: str,
    *,
    timeout: float = 60.0,
    shared_token: str | None = None,
    include_fail_inject: bool = False,
    include_tee_bonus: bool = True,
    **kwargs: Any,
) -> ScenarioResult:
    """VAL-CROSS-004/005/006/021 multi-node fabric + TEE offline bonus.

    Fail inject requires the API process to start with HYPER_SIM_ETH_FALLBACK=1.
    Default CLI bundle runs success + TEE twin on a clean fabric path.
    """

    from hypercluster.sim.cross_multinode_fabric_tee import (
        run_cross_multinode_fabric_tee,
    )

    _ = kwargs
    return run_cross_multinode_fabric_tee(
        base_url,
        timeout=timeout,
        shared_token=shared_token,
        include_fail_inject=include_fail_inject,
        include_tee_bonus=include_tee_bonus,
    )


def run_cross_market_resilience_auth_bundle(
    base_url: str,
    *,
    timeout: float = 45.0,
    shared_token: str | None = None,
    **kwargs: Any,
) -> ScenarioResult:
    """VAL-CROSS-010/011/024 double-rent recover, idle protect, nonce refuse."""

    from hypercluster.sim.cross_market_resilience_auth import (
        run_cross_market_resilience_auth,
    )

    _ = kwargs
    return run_cross_market_resilience_auth(
        base_url,
        timeout=timeout,
        shared_token=shared_token,
    )


def run_cross_worker_durability_paths_bundle(
    base_url: str,
    *,
    timeout: float = 60.0,
    shared_token: str | None = None,
    restart_fn: Any | None = None,
    include_restart: bool = True,
    **kwargs: Any,
) -> ScenarioResult:
    """VAL-CROSS-014/015/016/017/025/026/028 combined worker durability paths."""

    from hypercluster.sim.cross_worker_durability_paths import (
        run_cross_worker_durability_paths,
    )

    _ = kwargs
    return run_cross_worker_durability_paths(
        base_url,
        timeout=timeout,
        shared_token=shared_token,
        restart_fn=restart_fn,
        include_restart=include_restart,
    )


__all__ = [
    "DEFAULT_SCENARIO_ORDER",
    "FABRIC_SCENARIOS",
    "HAPPY_PATH_SCENARIOS",
    "SuiteResult",
    "normalize_scenario_names",
    "run_cross_happy_path_bundle",
    "run_cross_market_resilience_auth_bundle",
    "run_cross_multinode_fabric_tee_bundle",
    "run_cross_worker_durability_paths_bundle",
    "run_fabric_bundle",
    "run_happy_path_bundle",
    "run_named_scenarios",
    "run_scenario_suite",
]
