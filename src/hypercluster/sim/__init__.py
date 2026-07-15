"""Local simulator harness: identity gates, doctor, smoke scenarios, port band."""

from __future__ import annotations

from hypercluster.sim.doctor import DoctorReport, run_doctor
from hypercluster.sim.identity import IdentityReport, probe_identity_gates
from hypercluster.sim.ports import (
    DEFAULT_BAREMETAL_PORT,
    MAX_MISSION_PORT,
    MIN_MISSION_PORT,
    assert_mission_port,
    mission_port_band,
)
from hypercluster.sim.scenarios import ScenarioResult, run_scenario, run_smoke_scenario

__all__ = [
    "DEFAULT_BAREMETAL_PORT",
    "DoctorReport",
    "IdentityReport",
    "MAX_MISSION_PORT",
    "MIN_MISSION_PORT",
    "ScenarioResult",
    "assert_mission_port",
    "mission_port_band",
    "probe_identity_gates",
    "run_doctor",
    "run_scenario",
    "run_smoke_scenario",
]
