"""VAL-GPU-012..017: GpuHostEvidence + ordered fatal/advisory checks over FakeSsh.

No real network, no Verda, no set_weights, scoring formula untouched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from hypercluster.probe.model_table import (
    known_families,
    lookup_vram_window,
    models_match,
    normalize_gpu_model,
)
from hypercluster.probe.pipeline import (
    ADVISORY_CHECK_IDS,
    CHECK_ORDER,
    FATAL_CHECK_IDS,
    GpuProbeConfig,
    GpuProbeContext,
    GpuProbeService,
    occupied_uuid_index,
    run_gpu_probe,
)
from hypercluster.probe.transport import (
    COMMAND_ALLOWLIST,
    FakeOutcome,
    FakeSshTransport,
    TransportError,
    build_pass_script,
)
from hypercluster.probe.types import ClaimedInventory, GpuHostEvidence

V100_UUID = "GPU-11111111-1111-1111-1111-111111111111"
V100_UUID_B = "GPU-22222222-2222-2222-2222-222222222222"
A100_UUID = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _claim(
    model: str = "1V100.6V",
    count: int = 1,
) -> ClaimedInventory:
    return ClaimedInventory(gpu_model=model, gpu_count=count)


def _pass_ctx(**overrides: Any) -> GpuProbeContext:
    base: dict[str, Any] = {
        "node_id": "node-a",
        "provider_hotkey": "hk-provider-a",
        "ssh_endpoint": "10.0.0.1:22",
        "claimed": _claim(),
        "key_fingerprint": "sha256:keyfp-test",
        "occupied_uuids": set(),
        "prior_verified_uuids": None,
    }
    base.update(overrides)
    return GpuProbeContext(**base)


def _run(
    script: dict[str, FakeOutcome] | None = None,
    *,
    ctx: GpuProbeContext | None = None,
    config: GpuProbeConfig | None = None,
) -> tuple[GpuHostEvidence, FakeSshTransport]:
    transport = FakeSshTransport(scripted=script or build_pass_script())
    evidence = run_gpu_probe(transport, ctx or _pass_ctx(), config=config)
    return evidence, transport


# ---------------------------------------------------------------------------
# Schema / allowlist smoke
# ---------------------------------------------------------------------------


def test_check_order_starts_with_ssh_and_smi() -> None:
    assert CHECK_ORDER[0] == "ssh_connect"
    assert CHECK_ORDER[1] == "nvidia_smi_list"
    assert "gpu_model_match" in CHECK_ORDER
    assert "fingerprint_stable" in CHECK_ORDER
    assert "docker_runtime" in CHECK_ORDER
    assert {"power_limit_ratio", "idle_util"} <= ADVISORY_CHECK_IDS
    assert "ssh_connect" in FATAL_CHECK_IDS
    assert "nvidia_smi_list" in FATAL_CHECK_IDS


def test_pass_all_fixture_status_passed() -> None:
    evidence, transport = _run()
    assert evidence.status == "passed"
    assert evidence.failure_code is None
    assert evidence.transport == "fake"
    assert evidence.measured.gpu_count == 1
    assert V100_UUID in evidence.measured.uuid_set()
    assert evidence.digests.evidence_sha256
    assert evidence.digests.inventory_sha256
    assert evidence.digests.microbench_digest
    # Residual fatal checks executed after connect.
    ids = [c.id for c in evidence.checks]
    assert ids[0] == "ssh_connect"
    assert "claim_consistency" in ids
    assert "ssh_connect" in transport.commands_run
    # No private-key shape in public doc
    public = evidence.to_public()
    blob = json.dumps(public)
    assert "BEGIN" not in blob
    assert "PRIVATE KEY" not in blob
    assert "key_pem" not in blob


def test_service_wrapper_matches_run_gpu_probe() -> None:
    transport = FakeSshTransport(scripted=build_pass_script())
    svc = GpuProbeService(transport, config=GpuProbeConfig())
    evidence = svc.run(_pass_ctx())
    assert evidence.status == "passed"


# ---------------------------------------------------------------------------
# VAL-GPU-012: fatal ssh fail aborts residual checks
# ---------------------------------------------------------------------------


def test_val_gpu_012_ssh_fail_aborts_residual_checks() -> None:
    """When ssh_connect fails fatally, residual checks must not run as pass."""

    script = build_pass_script()
    script["ssh_connect"] = FakeOutcome(
        exit_code=255,
        fail_connect=True,
        error="connection refused",
        stderr="ssh: connect to host failed",
    )
    evidence, transport = _run(script)
    assert evidence.status in {"failed", "error"}
    assert evidence.failure_code == "ssh_connect"
    by_id = evidence.checks_by_id()
    assert "ssh_connect" in by_id
    assert by_id["ssh_connect"].passed is False
    assert by_id["ssh_connect"].fatal is True

    # Residual required checks must be absent (not invented as pass).
    residual_fatals = [
        "nvidia_smi_list",
        "gpu_count",
        "gpu_model_match",
        "gpu_uuid_valid",
        "gpu_uuid_unique",
        "vram_window",
        "driver_present",
        "cuda_microbench",
        "docker_runtime",
        "claim_consistency",
    ]
    for cid in residual_fatals:
        assert cid not in by_id, f"residual check {cid} must not run after ssh fail"
        assert cid not in transport.commands_run

    # None of residual should be greened.
    assert all(c.id == "ssh_connect" or not c.passed for c in evidence.checks)


def test_val_gpu_012_ssh_timeout_classifies_error() -> None:
    script = build_pass_script()
    script["ssh_connect"] = FakeOutcome(timed_out=True, exit_code=1, error="timeout")
    evidence, _ = _run(script)
    assert evidence.status in {"failed", "error"}
    assert evidence.checks_by_id()["ssh_connect"].passed is False
    assert "nvidia_smi_list" not in evidence.checks_by_id()


def test_val_gpu_012_nvidia_smi_fail_aborts_residual() -> None:
    script = build_pass_script()
    script["nvidia_smi_list"] = FakeOutcome(
        exit_code=127,
        stderr="nvidia-smi: command not found",
        error="not found",
    )
    evidence, transport = _run(script)
    assert evidence.status == "failed"
    assert evidence.failure_code == "nvidia_smi_list"
    by_id = evidence.checks_by_id()
    assert by_id["ssh_connect"].passed is True
    assert by_id["nvidia_smi_list"].passed is False
    for cid in ("gpu_model_match", "cuda_microbench", "claim_consistency"):
        assert cid not in by_id
    assert "cuda_microbench" not in transport.commands_run


# ---------------------------------------------------------------------------
# VAL-GPU-013: advisory-only failures keep status=passed
# ---------------------------------------------------------------------------


def test_val_gpu_013_advisory_only_keeps_passed() -> None:
    """power_limit_ratio / idle_util fail alone must not fail the run."""

    gpus = [
        {
            "name": "Tesla V100-SXM2-16GB",
            "uuid": V100_UUID,
            "memory_total_mb": 16160,
            "driver_version": "535.104.05",
            "power_limit_w": 100.0,  # << default → ratio ~0.33
            "power_default_w": 300.0,
            "util_gpu": 99.0,  # high util on "idle" advertise
            "util_mem": 80.0,
            "clocks_sm_mhz": 1000.0,
        }
    ]
    evidence, _ = _run(build_pass_script(gpus=gpus))
    assert evidence.status == "passed"
    by_id = evidence.checks_by_id()
    assert by_id["power_limit_ratio"].passed is False
    assert by_id["power_limit_ratio"].fatal is False
    assert by_id["idle_util"].passed is False
    assert by_id["idle_util"].fatal is False
    assert evidence.advisories
    assert {a.id for a in evidence.advisories} >= {"power_limit_ratio", "idle_util"}
    assert evidence.failure_code is None
    # Fatal checks still green.
    for cid in ("ssh_connect", "nvidia_smi_list", "gpu_model_match", "claim_consistency"):
        assert by_id[cid].passed is True


# ---------------------------------------------------------------------------
# VAL-GPU-014: model normalize table maps catalog alias → family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claimed", "measured"),
    [
        ("1V100.6V", "Tesla V100-SXM2-16GB"),
        ("Tesla V100", "Tesla V100-SXM2-32GB"),
        ("v100", "Tesla V100-SXM2-16GB"),
        ("1A100.80G", "NVIDIA A100-SXM4-80GB"),
        ("A100", "NVIDIA A100-SXM4-40GB"),
        ("H100", "NVIDIA H100 80GB HBM3"),
        ("T4", "Tesla T4"),
    ],
)
def test_val_gpu_014_alias_maps_to_family(claimed: str, measured: str) -> None:
    assert normalize_gpu_model(claimed) == normalize_gpu_model(measured)
    assert models_match(claimed, measured)


def test_val_gpu_014_wrong_family_rejected() -> None:
    assert not models_match("1V100.6V", "NVIDIA A100-SXM4-40GB")
    assert not models_match("A100", "Tesla V100-SXM2-16GB")
    assert normalize_gpu_model("1V100.6V") == "v100"
    assert normalize_gpu_model("NVIDIA A100-SXM4-80GB") == "a100"
    assert "v100" in known_families()
    win = lookup_vram_window("1V100.6V")
    assert win is not None
    assert win.contains(16160)
    assert not win.contains(2048)


def test_val_gpu_014_pipeline_accepts_catalog_alias() -> None:
    evidence, _ = _run(ctx=_pass_ctx(claimed=_claim("1V100.6V", 1)))
    assert evidence.status == "passed"
    assert evidence.checks_by_id()["gpu_model_match"].passed is True
    details = evidence.checks_by_id()["gpu_model_match"].details
    assert details["claimed_family"] == "v100"


def test_val_gpu_014_pipeline_rejects_wrong_model() -> None:
    script = build_pass_script(
        gpus=[
            {
                "name": "NVIDIA A100-SXM4-40GB",
                "uuid": A100_UUID,
                "memory_total_mb": 40960,
                "driver_version": "535.104.05",
                "power_limit_w": 400.0,
                "power_default_w": 400.0,
                "util_gpu": 0.0,
                "util_mem": 0.0,
                "clocks_sm_mhz": 0.0,
            }
        ]
    )
    # Node claims V100, host reports A100 → fatal model mismatch.
    evidence, transport = _run(script, ctx=_pass_ctx(claimed=_claim("1V100.6V", 1)))
    assert evidence.status == "failed"
    assert evidence.failure_code == "gpu_model_match"
    assert evidence.checks_by_id()["gpu_model_match"].passed is False
    # Later fatals aborted.
    assert "cuda_microbench" not in evidence.checks_by_id()
    assert "cuda_microbench" not in transport.commands_run


# ---------------------------------------------------------------------------
# VAL-GPU-015: docker runtime required fails without nvidia runtime
# ---------------------------------------------------------------------------


def test_val_gpu_015_docker_required_fails_without_nvidia_runtime() -> None:
    script = build_pass_script(
        docker={"present": True, "runtimes": ["runc"], "gpu_in_container": False}
    )
    # force docker_info script to omit nvidia runtime
    script["docker_info"] = FakeOutcome(
        exit_code=0,
        stdout=json.dumps({"Runtimes": {"runc": {}}, "Name": "fake"}),
    )
    evidence, _ = _run(
        script,
        config=GpuProbeConfig(require_docker_runtime=True),
    )
    assert evidence.status == "failed"
    assert evidence.failure_code == "docker_runtime"
    dc = evidence.checks_by_id()["docker_runtime"]
    assert dc.passed is False
    assert dc.fatal is True


def test_val_gpu_015_docker_missing_process_fails_when_required() -> None:
    script = build_pass_script()
    script["docker_info"] = FakeOutcome(
        exit_code=127,
        stderr="docker: not found",
        error="not found",
    )
    evidence, _ = _run(script, config=GpuProbeConfig(require_docker_runtime=True))
    assert evidence.status == "failed"
    assert evidence.checks_by_id()["docker_runtime"].passed is False


def test_val_gpu_015_docker_advisory_when_not_required() -> None:
    script = build_pass_script()
    script["docker_info"] = FakeOutcome(exit_code=127, stderr="no docker")
    evidence, _ = _run(script, config=GpuProbeConfig(require_docker_runtime=False))
    assert evidence.status == "passed"
    dc = evidence.checks_by_id()["docker_runtime"]
    assert dc.passed is False
    assert dc.fatal is False
    assert any(a.id == "docker_runtime" for a in evidence.advisories)


# ---------------------------------------------------------------------------
# VAL-GPU-016: duplicate GPU UUID across healthy nodes fatal for second
# ---------------------------------------------------------------------------


def test_val_gpu_016_duplicate_uuid_across_nodes_fatal() -> None:
    """Node B reporting node A's UUID fails gpu_uuid_unique."""

    # Node A owns V100_UUID.
    occupied = {V100_UUID}
    evidence, _ = _run(
        ctx=_pass_ctx(node_id="node-b", occupied_uuids=occupied),
    )
    assert evidence.status == "failed"
    assert evidence.failure_code == "gpu_uuid_unique"
    check = evidence.checks_by_id()["gpu_uuid_unique"]
    assert check.passed is False
    assert check.fatal is True
    assert V100_UUID in check.details.get("collisions", [])


def test_val_gpu_016_unique_uuid_passes() -> None:
    occupied = {V100_UUID_B}  # different UUID — fine
    evidence, _ = _run(ctx=_pass_ctx(occupied_uuids=occupied))
    assert evidence.status == "passed"
    assert evidence.checks_by_id()["gpu_uuid_unique"].passed is True


def test_occupied_uuid_index_excludes_self() -> None:
    claims = [
        ("node-a", [V100_UUID]),
        ("node-b", [V100_UUID_B]),
    ]
    occupied = occupied_uuid_index(claims, exclude_node_id="node-a")
    assert V100_UUID not in occupied
    assert V100_UUID_B in occupied


# ---------------------------------------------------------------------------
# VAL-GPU-017: fingerprint change after verified forces re-admit fail
# ---------------------------------------------------------------------------


def test_val_gpu_017_fingerprint_churn_after_verified_fails() -> None:
    prior = {V100_UUID}
    script = build_pass_script(
        gpus=[
            {
                "name": "Tesla V100-SXM2-16GB",
                "uuid": V100_UUID_B,  # different set
                "memory_total_mb": 16160,
                "driver_version": "535.104.05",
                "power_limit_w": 300.0,
                "power_default_w": 300.0,
                "util_gpu": 0.0,
                "util_mem": 0.0,
                "clocks_sm_mhz": 0.0,
            }
        ]
    )
    evidence, _ = _run(
        script,
        ctx=_pass_ctx(prior_verified_uuids=prior),
    )
    assert evidence.status == "failed"
    assert evidence.failure_code == "fingerprint_stable"
    check = evidence.checks_by_id()["fingerprint_stable"]
    assert check.passed is False
    assert check.fatal is True
    assert check.details.get("re_admit_required") is True


def test_val_gpu_017_stable_fingerprint_passes() -> None:
    prior = {V100_UUID}
    evidence, _ = _run(ctx=_pass_ctx(prior_verified_uuids=prior))
    assert evidence.status == "passed"
    assert evidence.checks_by_id()["fingerprint_stable"].passed is True


def test_val_gpu_017_no_prior_skips_fingerprint_gate() -> None:
    evidence, _ = _run(ctx=_pass_ctx(prior_verified_uuids=None))
    assert evidence.status == "passed"
    check = evidence.checks_by_id()["fingerprint_stable"]
    assert check.passed is True
    assert check.details.get("skipped") is True


# ---------------------------------------------------------------------------
# Extra honesty gates (support VAL series fixtures later)
# ---------------------------------------------------------------------------


def test_vram_lie_fails_vram_window() -> None:
    script = build_pass_script(
        gpus=[
            {
                "name": "Tesla V100-SXM2-16GB",
                "uuid": V100_UUID,
                "memory_total_mb": 2048,  # absurd for V100
                "driver_version": "535.104.05",
                "power_limit_w": 300.0,
                "power_default_w": 300.0,
                "util_gpu": 0.0,
                "util_mem": 0.0,
                "clocks_sm_mhz": 0.0,
            }
        ]
    )
    evidence, _ = _run(script)
    assert evidence.status == "failed"
    assert evidence.failure_code == "vram_window"


def test_microbench_fail_fatal() -> None:
    script = build_pass_script()
    script["cuda_microbench"] = FakeOutcome(
        exit_code=1,
        stdout=json.dumps({"ok": False, "digest": "sha256:bad"}),
        stderr="cuda error",
    )
    evidence, _ = _run(script)
    assert evidence.status == "failed"
    assert evidence.failure_code == "cuda_microbench"


def test_allowlist_rejects_unknown_command_id() -> None:
    t = FakeSshTransport(scripted=build_pass_script())
    t.connect()
    with pytest.raises(TransportError) as ei:
        t.run("rm_rf_slash")
    assert ei.value.code == "unknown_command_id"
    assert "rm_rf_slash" not in COMMAND_ALLOWLIST


def test_evidence_schema_forbids_extra_private_keys_leakage_keys() -> None:
    evidence, _ = _run()
    public = evidence.to_public()
    # Ensure only fingerprint-style key material.
    assert public.get("key_fingerprint", "").startswith("sha256:")
    raw = public.get("raw_redacted") or {}
    assert "private_key" not in raw
    assert "pem" not in raw
