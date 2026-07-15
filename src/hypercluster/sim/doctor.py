"""Sim doctor: CI readiness of sim backends + identity gates (VAL-CLI-014).

Checks (fail closed with actionable error messages):
- optional identity gates (health + ready) when a base URL is provided
- deterministic inventory seed produces multi-node plan-ready topology
- fabric launcher module / sim_launch importable and contract shape
- TEE offline fixtures present (golden compose + positive TDX)
- mock master module importable (optional live reachability)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hypercluster.sim.identity import IdentityReport, probe_identity_gates
from hypercluster.sim.inventory import plan_readiness
from hypercluster.sim.inventory import (
    seed_sim_inventory as _default_seed_sim_inventory,
)

# Module-level alias so tests can monkeypatch ``doctor.seed_sim_inventory``.
seed_sim_inventory = _default_seed_sim_inventory


@dataclass(slots=True)
class BackendChecks:
    """Result of offline (or no-API) backend readiness checks."""

    ok: bool
    backend_checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DoctorReport:
    """Aggregate sim doctor result."""

    ok: bool
    base_url: str | None
    identity: IdentityReport | None
    backend_checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        status = "PASS" if self.ok else "FAIL"
        lines = [f"sim doctor {status}"]
        if self.base_url is not None:
            lines.append(f"base_url={self.base_url}")
        else:
            lines.append("base_url=<offline-backends-only>")
        lines.extend(f"backend: {c}" for c in self.backend_checks)
        if self.identity is not None:
            lines.extend(self.identity.summary_lines())
        for err in self.errors:
            lines.append(f"error={err}")
        return lines


def _tee_fixture_root() -> Path:
    """Locate tests/fixtures/tee from repo checkout (same policy as scenarios)."""

    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "tests" / "fixtures" / "tee",
        here.parents[4] / "tests" / "fixtures" / "tee",
        Path.cwd() / "tests" / "fixtures" / "tee",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    import os

    env = (os.environ.get("HYPER_TEE_FIXTURE_DIR") or "").strip()
    if env:
        path = Path(env)
        if path.is_dir():
            return path
    # Prefer a sentinel missing path that check_sim_backends can report.
    return candidates[0]


def _check_inventory(backends: BackendChecks) -> None:
    """Inventory backend: deterministic seed + multi-node plan readiness."""

    try:
        inv = seed_sim_inventory(seed=0, node_count=4, gpus_per_node=2)
        if len(inv.nodes) < 2:
            backends.errors.append(
                "inventory backend actionable: expected ≥2 sim nodes after seed; "
                f"got {len(inv.nodes)} (check hypercluster.sim.inventory.seed_sim_inventory)"
            )
            return
        if not inv.graph_digest.startswith("sha256:"):
            backends.errors.append(
                "inventory backend actionable: graph_digest missing sha256: prefix"
            )
            return
        readiness = plan_readiness(
            inv,
            world_size=4,
            nnodes=2,
            nproc_per_node=2,
        )
        if not readiness.ok:
            backends.errors.append(
                "inventory backend actionable: multi-node plan not ready after seed: "
                f"{readiness.reason}"
            )
            return
        inv2 = seed_sim_inventory(seed=0, node_count=4, gpus_per_node=2)
        if inv.graph_digest != inv2.graph_digest:
            backends.errors.append(
                "inventory backend actionable: seed_sim_inventory is non-deterministic "
                f"for identical seed (digest {inv.graph_digest} != {inv2.graph_digest})"
            )
            return
        backends.backend_checks.append(
            f"inventory: seed deterministic ({len(inv.nodes)} nodes, "
            f"digest={inv.graph_digest[:19]}…, plan_ready=true)"
        )
    except Exception as exc:  # noqa: BLE001 — doctor must never crash on check
        backends.errors.append(
            f"inventory backend actionable: seed failed: {exc!r}"
        )


def _check_launcher(backends: BackendChecks) -> None:
    """Launcher backend: sim_launch import + request/result contract available."""

    try:
        from hypercluster.fabric.launcher import (
            LAUNCHER_VERSION,
            LaunchRequest,
            LaunchResult,
            sim_launch,
        )

        if not callable(sim_launch):
            backends.errors.append(
                "launcher backend actionable: sim_launch is not callable "
                "(import hypercluster.fabric.launcher)"
            )
            return
        _ = (LaunchRequest, LaunchResult)
        backends.backend_checks.append(
            f"launcher: sim_launch importable ({LAUNCHER_VERSION})"
        )
    except Exception as exc:  # noqa: BLE001
        backends.errors.append(
            "launcher backend actionable: failed to import fabric launcher "
            f"(sim_launch): {exc!r}"
        )


def _check_tee_fixtures(backends: BackendChecks) -> None:
    """TEE offline fixture pack required for CI (golden compose + positive TDX)."""

    root = _tee_fixture_root()
    required = {
        "golden_compose.yml": root / "golden_compose.yml",
        "golden_compose.sha256": root / "golden_compose.sha256",
        "positive_tdx_v1.json": root / "positive_tdx_v1.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        backends.errors.append(
            "tee fixtures backend actionable: missing fixture file(s) "
            f"{missing} under {root} "
            "(expected tests/fixtures/tee/{golden_compose.yml,"
            "golden_compose.sha256,positive_tdx_v1.json} on default checkout; "
            "set HYPER_TEE_FIXTURE_DIR to override)"
        )
        return

    try:
        from hypercluster.attest.compose_hash import (
            hash_compose_file,
            load_golden_hash_file,
        )

        digest = hash_compose_file(required["golden_compose.yml"])
        golden = load_golden_hash_file(required["golden_compose.sha256"])
        if digest != golden:
            backends.errors.append(
                "tee fixtures backend actionable: golden compose hash mismatch "
                f"(computed={digest} golden={golden}); fixtures may be corrupted"
            )
            return
        from hypercluster.attest.offline_fixtures import load_quote_fixture

        env = load_quote_fixture(required["positive_tdx_v1.json"])
        if not getattr(env, "compose_hash", None):
            backends.errors.append(
                "tee fixtures backend actionable: positive_tdx_v1.json missing compose_hash"
            )
            return
        backends.backend_checks.append(
            f"tee fixtures: golden_compose + positive_tdx ready under {root}"
        )
    except Exception as exc:  # noqa: BLE001
        backends.errors.append(
            f"tee fixtures backend actionable: fixture validation failed: {exc!r}"
        )


def _check_mock_master_module(backends: BackendChecks) -> None:
    """Mock master is optional at runtime; module must import for local CI wiring."""

    try:
        from hypercluster.sim import mock_master as mm

        if not hasattr(mm, "app"):
            backends.errors.append(
                "mock_master backend actionable: hypercluster.sim.mock_master "
                "missing FastAPI app attribute"
            )
            return
        backends.backend_checks.append(
            "mock_master: module importable (optional live :3201)"
        )
    except Exception as exc:  # noqa: BLE001
        backends.errors.append(
            f"mock_master backend actionable: import failed: {exc!r}"
        )


def check_sim_backends() -> BackendChecks:
    """Run offline simulator backend checks (no live identity URL required)."""

    backends = BackendChecks(ok=True)
    _check_inventory(backends)
    _check_launcher(backends)
    _check_tee_fixtures(backends)
    _check_mock_master_module(backends)
    backends.ok = not backends.errors
    return backends


def run_doctor(
    base_url: str | None = None,
    *,
    timeout: float = 5.0,
    require_identity: bool = True,
    identity_probe: Callable[..., IdentityReport] = probe_identity_gates,
) -> DoctorReport:
    """Run doctor: identity gates (optional) + real backend fixture checks.

    VAL-CLI-014: readiness means inventory, launcher, tee fixtures, and
    mock-master module present. When *require_identity* is True and a base URL
    is provided, health+ready must also be green (VAL-SCAF-036).
    """

    backends = check_sim_backends()
    errors: list[str] = list(backends.errors)
    identity: IdentityReport | None = None
    normalized: str | None = None
    backend_checks = list(backends.backend_checks)

    if base_url is not None and str(base_url).strip():
        normalized = str(base_url).rstrip("/")
        if require_identity:
            identity = identity_probe(normalized, timeout=timeout)
            if not identity.ok:
                errors.extend(identity.errors)
            else:
                backend_checks.insert(0, "identity_gates: health+ready green")
        else:
            backend_checks.insert(0, "identity_gates: skipped (require_identity=false)")
    elif require_identity:
        errors.append(
            "identity backend actionable: base_url required when require_identity=true "
            "(pass --url or use require_identity=False for offline backend doctor)"
        )
    else:
        backend_checks.insert(0, "identity_gates: skipped (no base_url; offline backends only)")

    ok = not errors
    return DoctorReport(
        ok=ok,
        base_url=normalized,
        identity=identity,
        backend_checks=backend_checks,
        errors=errors,
    )


__all__ = [
    "BackendChecks",
    "DoctorReport",
    "check_sim_backends",
    "run_doctor",
    "seed_sim_inventory",
]
