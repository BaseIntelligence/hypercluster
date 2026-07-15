"""Local simulator harness: identity gates, doctor, smoke scenarios, port band."""

from __future__ import annotations

from hypercluster.sim.doctor import BackendChecks, DoctorReport, check_sim_backends, run_doctor
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
from hypercluster.sim.seed import (
    DEFAULT_SIM_SEED,
    SIM_SEED_ENV,
    inventory_shape_digest,
    resolve_sim_seed,
)

__all__ = [
    "DEFAULT_BAREMETAL_PORT",
    "DEFAULT_SIM_SEED",
    "BackendChecks",
    "DoctorReport",
    "IdentityReport",
    "MAX_MISSION_PORT",
    "MIN_MISSION_PORT",
    "PlanReadiness",
    "SIM_SEED_ENV",
    "ScenarioResult",
    "SimInventory",
    "SimNode",
    "assert_mission_port",
    "check_sim_backends",
    "default_sim_inventory",
    "inventory_shape_digest",
    "mission_port_band",
    "plan_readiness",
    "probe_identity_gates",
    "resolve_sim_seed",
    "run_doctor",
    "run_scenario",
    "run_smoke_scenario",
    "seed_sim_inventory",
]
