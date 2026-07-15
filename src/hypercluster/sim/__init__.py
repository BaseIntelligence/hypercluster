"""Local simulator harness: identity gates, doctor, smoke scenarios, port band."""

from __future__ import annotations

from hypercluster.sim.cross_happy_path import (
    EgressTrace,
    capture_httpx_egress,
    run_cross_happy_path,
)
from hypercluster.sim.cross_market_resilience_auth import (
    run_cross_double_rent_recover,
    run_cross_idle_rental_protection,
    run_cross_market_resilience_auth,
    run_cross_nonce_replay_refuse,
)
from hypercluster.sim.cross_multinode_fabric_tee import (
    run_cross_multinode_fabric_fail,
    run_cross_multinode_fabric_tee,
    run_cross_multinode_success,
    run_cross_tee_offline_bonus,
)
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
from hypercluster.sim.orchestration import (
    DEFAULT_SCENARIO_ORDER,
    SuiteResult,
    run_cross_happy_path_bundle,
    run_cross_market_resilience_auth_bundle,
    run_cross_multinode_fabric_tee_bundle,
    run_named_scenarios,
    run_scenario_suite,
)
from hypercluster.sim.ports import (
    DEFAULT_BAREMETAL_PORT,
    MAX_MISSION_PORT,
    MIN_MISSION_PORT,
    assert_mission_port,
    mission_port_band,
)
from hypercluster.sim.scenarios import (
    CROSS_HAPPY_PATH,
    CROSS_MARKET_RESILIENCE,
    CROSS_MULTINODE,
    KNOWN_SCENARIOS,
    ScenarioResult,
    run_marketplace_scenario,
    run_nccl_scenario,
    run_scenario,
    run_smoke_scenario,
    run_tee_offline_scenario,
    run_weights_scenario,
)
from hypercluster.sim.seed import (
    DEFAULT_SIM_SEED,
    SIM_SEED_ENV,
    inventory_shape_digest,
    resolve_sim_seed,
)

__all__ = [
    "CROSS_HAPPY_PATH",
    "CROSS_MARKET_RESILIENCE",
    "CROSS_MULTINODE",
    "DEFAULT_BAREMETAL_PORT",
    "DEFAULT_SCENARIO_ORDER",
    "DEFAULT_SIM_SEED",
    "KNOWN_SCENARIOS",
    "BackendChecks",
    "DoctorReport",
    "EgressTrace",
    "IdentityReport",
    "MAX_MISSION_PORT",
    "MIN_MISSION_PORT",
    "PlanReadiness",
    "SIM_SEED_ENV",
    "ScenarioResult",
    "SimInventory",
    "SimNode",
    "SuiteResult",
    "assert_mission_port",
    "capture_httpx_egress",
    "check_sim_backends",
    "default_sim_inventory",
    "inventory_shape_digest",
    "mission_port_band",
    "plan_readiness",
    "probe_identity_gates",
    "resolve_sim_seed",
    "run_cross_double_rent_recover",
    "run_cross_happy_path",
    "run_cross_happy_path_bundle",
    "run_cross_idle_rental_protection",
    "run_cross_market_resilience_auth",
    "run_cross_market_resilience_auth_bundle",
    "run_cross_multinode_fabric_fail",
    "run_cross_multinode_fabric_tee",
    "run_cross_multinode_fabric_tee_bundle",
    "run_cross_multinode_success",
    "run_cross_nonce_replay_refuse",
    "run_cross_tee_offline_bonus",
    "run_doctor",
    "run_marketplace_scenario",
    "run_named_scenarios",
    "run_nccl_scenario",
    "run_scenario",
    "run_scenario_suite",
    "run_smoke_scenario",
    "run_tee_offline_scenario",
    "run_weights_scenario",
    "seed_sim_inventory",
]
