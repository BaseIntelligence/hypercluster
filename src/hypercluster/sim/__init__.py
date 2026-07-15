"""Local simulator harness: identity gates, doctor, smoke scenarios, port band."""

from __future__ import annotations

from hypercluster.sim.doctor import DoctorReport, run_doctor
from hypercluster.sim.identity import IdentityReport, probe_identity_gates
from hypercluster.sim.inventory import (
    PlanReadiness,
    SimInventory,
    SimNode,
    default_sim_inventory,
    plan_readiness,
    seed_sim_inventory,
)
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
    "PlanReadiness",
    "ScenarioResult",
    "SimInventory",
    "SimNode",
    "assert_mission_port",
    "default_sim_inventory",
    "mission_port_band",
    "plan_readiness",
    "probe_identity_gates",
    "run_doctor",
    "run_scenario",
    "run_smoke_scenario",
    "seed_sim_inventory",
]
