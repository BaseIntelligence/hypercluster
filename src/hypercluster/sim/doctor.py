"""Sim doctor: CI readiness of sim backends + identity gates (M1 stub)."""

from __future__ import annotations

from dataclasses import dataclass, field

from hypercluster.sim.identity import IdentityReport, probe_identity_gates


@dataclass(slots=True)
class DoctorReport:
    """Aggregate sim doctor result."""

    ok: bool
    base_url: str
    identity: IdentityReport
    backend_checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        status = "PASS" if self.ok else "FAIL"
        lines = [f"sim doctor {status}", f"base_url={self.base_url}"]
        lines.extend(f"backend: {c}" for c in self.backend_checks)
        lines.extend(self.identity.summary_lines())
        for err in self.errors:
            lines.append(f"error={err}")
        return lines


def run_doctor(
    base_url: str,
    *,
    timeout: float = 5.0,
) -> DoctorReport:
    """Run doctor: identity gates first, then lightweight backend stubs.

    Smoke/sim doctor uses identity gates (VAL-SCAF-036): health+ready must be
    green or doctor fails.
    """

    identity = probe_identity_gates(base_url, timeout=timeout)
    backends = [
        "identity_gates: health+ready",
        "local_sim_backend: stub_ok (M1)",
        "inventory_fixture: stub_ok (M1)",
        "tee_offline_fixtures: stub_ok (M1 path present later)",
    ]
    errors: list[str] = []
    if not identity.ok:
        errors.extend(identity.errors)
    ok = identity.ok and not errors
    return DoctorReport(
        ok=ok,
        base_url=identity.base_url,
        identity=identity,
        backend_checks=backends,
        errors=errors,
    )


__all__ = ["DoctorReport", "run_doctor"]
