"""VAL-GPU-020..028: FakeSsh fixture bank CI matrix + production refuse silent fake.

No real GPUs, no live SSH, no Verda, never set_weights. Formula unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hypercluster.probe.fixtures import (
    KNOWN_FIXTURE_NAMES,
    PEER_CLONE_UUID,
    V100_UUID,
    fixture_to_public_dict,
    get_fixture,
    list_fixtures,
    load_fixture_json,
    load_named_fixture,
    package_fixture_dir,
)
from hypercluster.probe.inventory_merge import (
    GPU_PROBE_STATUS_VERIFIED,
    apply_probe_to_node_fields,
    merge_probe_into_inventory,
)
from hypercluster.probe.pipeline import GpuProbeConfig, GpuProbeContext, run_gpu_probe
from hypercluster.probe.resolve import (
    FAKE_SSH_NOT_ALLOWED,
    SSH_TRANSPORT_UNAVAILABLE,
    TransportConfigError,
    resolve_ssh_transport,
)
from hypercluster.probe.transport import FakeSshTransport
from hypercluster.settings import HyperSettings, clear_settings_cache

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "gpu_probe"


def _ctx_from_fixture(fx: Any, **overrides: Any) -> GpuProbeContext:
    base: dict[str, Any] = {
        "node_id": "node-fake-a",
        "provider_hotkey": "hk-provider-fake",
        "ssh_endpoint": "10.0.0.9:22",
        "claimed": fx.claimed,
        "key_fingerprint": "sha256:fake-key-fingerprint",
        "occupied_uuids": set(fx.occupied_uuids),
        "prior_verified_uuids": (
            None if fx.prior_verified_uuids is None else set(fx.prior_verified_uuids)
        ),
    }
    base.update(overrides)
    return GpuProbeContext(**base)


def _run_fixture(name: str, **ctx_overrides: Any):
    fx = get_fixture(name)
    transport = FakeSshTransport(scripted=fx.scripted)
    config = GpuProbeConfig(require_docker_runtime=fx.require_docker_runtime)
    evidence = run_gpu_probe(
        transport,
        _ctx_from_fixture(fx, **ctx_overrides),
        config=config,
    )
    return evidence, transport, fx


# ---------------------------------------------------------------------------
# Fixture bank presence
# ---------------------------------------------------------------------------


def test_fixture_bank_names_complete() -> None:
    expected = {
        "pass_all",
        "no_gpu",
        "wrong_model",
        "uuid_clone",
        "vram_lie",
        "bench_fail",
        "docker_missing",
        "ssh_timeout",
        "fingerprint_churn",
    }
    assert set(list_fixtures()) == expected
    assert KNOWN_FIXTURE_NAMES == expected
    for name in expected:
        fx = get_fixture(name)
        assert fx.name == name
        assert fx.scripted  # non-empty script


def test_json_fixtures_exist_and_roundtrip() -> None:
    """JSON bank under tests/fixtures/gpu_probe mirrors builders (AGENTS.md)."""

    mapping = {
        "v100_pass_all.json": "pass_all",
        "no_gpu.json": "no_gpu",
        "wrong_model.json": "wrong_model",
        "uuid_clone.json": "uuid_clone",
        "vram_lie.json": "vram_lie",
        "bench_fail.json": "bench_fail",
        "docker_missing.json": "docker_missing",
        "ssh_timeout.json": "ssh_timeout",
        "fingerprint_churn.json": "fingerprint_churn",
    }
    assert FIXTURE_DIR.is_dir(), f"missing fixture dir {FIXTURE_DIR}"
    for filename, canonical in mapping.items():
        path = FIXTURE_DIR / filename
        assert path.is_file(), f"missing fixture file {path}"
        loaded = load_fixture_json(path)
        built = get_fixture(canonical)
        assert loaded.expected_failure_code == built.expected_failure_code
        assert set(loaded.scripted.keys()) >= {"ssh_connect"}


# ---------------------------------------------------------------------------
# VAL-GPU-020: FakeSsh no_gpu fails nvidia_smi_list
# ---------------------------------------------------------------------------


def test_val_gpu_020_no_gpu_fails_nvidia_smi_list() -> None:
    evidence, transport, fx = _run_fixture("no_gpu")
    assert fx.expected_failure_code == "nvidia_smi_list"
    assert evidence.status == "failed"
    assert evidence.failure_code == "nvidia_smi_list"
    by_id = evidence.checks_by_id()
    assert "nvidia_smi_list" in by_id
    assert by_id["nvidia_smi_list"].passed is False
    assert by_id["nvidia_smi_list"].fatal is True
    assert by_id["ssh_connect"].passed is True
    # residual after nvidia_smi_list not greened
    assert "gpu_model_match" not in by_id
    assert "cuda_microbench" not in transport.commands_run


# ---------------------------------------------------------------------------
# VAL-GPU-021: FakeSsh wrong model fails gpu_model_match
# ---------------------------------------------------------------------------


def test_val_gpu_021_wrong_model_fails_gpu_model_match() -> None:
    evidence, transport, _ = _run_fixture("wrong_model")
    assert evidence.status == "failed"
    assert evidence.failure_code == "gpu_model_match"
    check = evidence.checks_by_id()["gpu_model_match"]
    assert check.passed is False
    assert check.fatal is True
    assert "cuda_microbench" not in evidence.checks_by_id()
    assert "cuda_microbench" not in transport.commands_run


# ---------------------------------------------------------------------------
# VAL-GPU-022: FakeSsh clone UUID fails gpu_uuid_unique
# ---------------------------------------------------------------------------


def test_val_gpu_022_uuid_clone_fails_uuid_unique() -> None:
    evidence, _, fx = _run_fixture("uuid_clone")
    assert PEER_CLONE_UUID in fx.occupied_uuids
    assert evidence.status == "failed"
    assert evidence.failure_code == "gpu_uuid_unique"
    check = evidence.checks_by_id()["gpu_uuid_unique"]
    assert check.passed is False
    assert check.fatal is True
    assert PEER_CLONE_UUID in check.details.get("collisions", [])


# ---------------------------------------------------------------------------
# VAL-GPU-023: FakeSsh vram lie fails vram_window
# ---------------------------------------------------------------------------


def test_val_gpu_023_vram_lie_fails_vram_window() -> None:
    evidence, _, _ = _run_fixture("vram_lie")
    assert evidence.status == "failed"
    assert evidence.failure_code == "vram_window"
    assert evidence.checks_by_id()["vram_window"].passed is False


# ---------------------------------------------------------------------------
# VAL-GPU-024: FakeSsh bench fail fails cuda_microbench
# ---------------------------------------------------------------------------


def test_val_gpu_024_bench_fail_fails_cuda_microbench() -> None:
    evidence, _, _ = _run_fixture("bench_fail")
    assert evidence.status == "failed"
    assert evidence.failure_code == "cuda_microbench"
    assert evidence.checks_by_id()["cuda_microbench"].passed is False


# ---------------------------------------------------------------------------
# VAL-GPU-025: FakeSsh docker missing with require fails
# ---------------------------------------------------------------------------


def test_val_gpu_025_docker_missing_with_require_fails() -> None:
    evidence, _, fx = _run_fixture("docker_missing")
    assert fx.require_docker_runtime is True
    assert evidence.status == "failed"
    assert evidence.failure_code == "docker_runtime"
    check = evidence.checks_by_id()["docker_runtime"]
    assert check.passed is False
    assert check.fatal is True


def test_val_gpu_025_docker_missing_advisory_when_not_required() -> None:
    fx = get_fixture("docker_missing")
    transport = FakeSshTransport(scripted=fx.scripted)
    evidence = run_gpu_probe(
        transport,
        _ctx_from_fixture(fx),
        config=GpuProbeConfig(require_docker_runtime=False),
    )
    assert evidence.status == "passed"
    assert evidence.checks_by_id()["docker_runtime"].fatal is False


# ---------------------------------------------------------------------------
# VAL-GPU-026: FakeSsh ssh timeout yields error/failed connect
# ---------------------------------------------------------------------------


def test_val_gpu_026_ssh_timeout_error_or_failed_connect() -> None:
    evidence, transport, _ = _run_fixture("ssh_timeout")
    assert evidence.status in {"error", "failed"}
    assert evidence.failure_code == "ssh_connect"
    check = evidence.checks_by_id()["ssh_connect"]
    assert check.passed is False
    assert check.fatal is True
    # residual not invent-passed
    assert "nvidia_smi_list" not in evidence.checks_by_id()
    assert "nvidia_smi_list" not in transport.commands_run


# ---------------------------------------------------------------------------
# VAL-GPU-027: FakeSsh pass-all merges inventory_json UUID list
# ---------------------------------------------------------------------------


def test_val_gpu_027_pass_all_merges_inventory_json_uuid_list() -> None:
    evidence, _, _ = _run_fixture("pass_all")
    assert evidence.status == "passed"
    assert V100_UUID in evidence.measured.uuid_set()

    # Simulate a node inventory blob before probe.
    prior = {
        "has_ib": False,
        "source": "self_report",
        "gpu_model_claim": "1V100.6V",
    }
    merged = merge_probe_into_inventory(prior, evidence)
    assert merged["gpu_probe_status"] == GPU_PROBE_STATUS_VERIFIED
    assert V100_UUID in merged["gpu_uuids"]
    assert merged["measured_gpu_count"] == 1
    assert merged["gpu_probe_evidence_id"] == evidence.id
    assert merged["source"] == "self_report"  # prior keys preserved
    assert "gpu_probe_digests" in merged

    patch = apply_probe_to_node_fields(inventory_json=json.dumps(prior), evidence=evidence)
    assert patch["gpu_probe_status"] == "verified"
    assert V100_UUID in patch["gpu_uuids"]
    reparsed = json.loads(patch["inventory_json"])
    assert V100_UUID in reparsed["gpu_uuids"]
    assert reparsed["gpu_probe_status"] == "verified"


def test_val_gpu_027_failed_probe_stamps_status_without_inventing_uuids() -> None:
    evidence, _, _ = _run_fixture("no_gpu")
    merged = merge_probe_into_inventory({}, evidence)
    assert merged["gpu_probe_status"] == "failed"
    assert merged.get("gpu_uuids", []) == [] or "gpu_uuids" not in merged or not merged["gpu_uuids"]


# ---------------------------------------------------------------------------
# VAL-GPU-028: Production settings refuse silent fake transport
# ---------------------------------------------------------------------------


def test_val_gpu_028_production_defaults_refuse_silent_fake() -> None:
    """Default HyperSettings: real transport, allow_fake_ssh=False."""

    clear_settings_cache()
    prod = HyperSettings()  # defaults, no env override needed for fields
    assert prod.ssh_transport == "real"
    assert prod.allow_fake_ssh is False

    with pytest.raises(TransportConfigError) as ei:
        resolve_ssh_transport(prod)
    assert ei.value.code == SSH_TRANSPORT_UNAVAILABLE
    assert ei.value.status_code == 503
    assert "silent" in ei.value.message.lower() or "refuse" in ei.value.message.lower()


def test_val_gpu_028_fake_without_allow_flag_rejected() -> None:
    settings = HyperSettings(ssh_transport="fake", allow_fake_ssh=False)
    with pytest.raises(TransportConfigError) as ei:
        resolve_ssh_transport(settings, fixture_name="pass_all")
    assert ei.value.code == FAKE_SSH_NOT_ALLOWED
    assert ei.value.status_code == 503


def test_val_gpu_028_explicit_fake_allow_works() -> None:
    settings = HyperSettings(ssh_transport="fake", allow_fake_ssh=True)
    transport = resolve_ssh_transport(settings, fixture_name="pass_all")
    assert isinstance(transport, FakeSshTransport)
    assert transport.name == "fake"
    # full probe under allowed fake passes
    fx = get_fixture("pass_all")
    evidence = run_gpu_probe(
        transport,
        _ctx_from_fixture(fx),
        config=GpuProbeConfig(require_docker_runtime=True),
    )
    assert evidence.status == "passed"
    assert evidence.transport == "fake"


def test_val_gpu_028_real_with_injected_executor_ok() -> None:
    """When a real executor is provided, settings real path uses it (no fake)."""

    class _StubReal:
        name = "real"

        def connect(self):  # pragma: no cover - not used for resolve assert
            raise NotImplementedError

        def run(self, command_id: str, *, timeout_s: float | None = None):
            raise NotImplementedError

        def close(self) -> None:
            return None

    stub = _StubReal()
    settings = HyperSettings(ssh_transport="real", allow_fake_ssh=False)
    resolved = resolve_ssh_transport(settings, real_transport=stub)  # type: ignore[arg-type]
    assert resolved is stub
    assert resolved.name == "real"


def test_val_gpu_028_env_cannot_silent_default_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    """If someone sets TRANSPORT=fake without ALLOW, still refuse."""

    clear_settings_cache()
    monkeypatch.setenv("HYPER_SSH_TRANSPORT", "fake")
    monkeypatch.delenv("HYPER_ALLOW_FAKE_SSH", raising=False)
    settings = HyperSettings()
    assert settings.ssh_transport == "fake"
    assert settings.allow_fake_ssh is False
    with pytest.raises(TransportConfigError) as ei:
        resolve_ssh_transport(settings)
    assert ei.value.code == FAKE_SSH_NOT_ALLOWED
    clear_settings_cache()


# ---------------------------------------------------------------------------
# Extra: fingerprint_churn bank + alias names + matrix parametrize
# ---------------------------------------------------------------------------


def test_fingerprint_churn_fixture() -> None:
    evidence, _, _ = _run_fixture("fingerprint_churn")
    assert evidence.status == "failed"
    assert evidence.failure_code == "fingerprint_stable"


@pytest.mark.parametrize(
    ("name", "expect_status", "expect_code"),
    [
        ("pass_all", "passed", None),
        ("no_gpu", "failed", "nvidia_smi_list"),
        ("wrong_model", "failed", "gpu_model_match"),
        ("uuid_clone", "failed", "gpu_uuid_unique"),
        ("vram_lie", "failed", "vram_window"),
        ("bench_fail", "failed", "cuda_microbench"),
        ("docker_missing", "failed", "docker_runtime"),
        ("ssh_timeout", "error", "ssh_connect"),
        ("fingerprint_churn", "failed", "fingerprint_stable"),
    ],
)
def test_full_fixture_matrix(
    name: str,
    expect_status: str,
    expect_code: str | None,
) -> None:
    evidence, _, fx = _run_fixture(name)
    assert evidence.status == expect_status, (
        f"{name}: status={evidence.status} failure={evidence.failure_code}"
    )
    assert evidence.failure_code == expect_code
    assert fx.expected_status == expect_status
    assert fx.expected_failure_code == expect_code
    # No private-key leakage in public evidence.
    public = json.dumps(evidence.to_public())
    assert "BEGIN" not in public
    assert "PRIVATE KEY" not in public


def test_alias_v100_pass_all_and_hyphen_names() -> None:
    for alias in ("v100_pass_all", "pass-all", "pass_all"):
        fx = load_named_fixture(alias)
        assert fx.name == "pass_all"


def test_package_fixture_dir_points_at_tests() -> None:
    assert package_fixture_dir().name == "gpu_probe"


def test_fixture_public_dict_serializable() -> None:
    fx = get_fixture("pass_all")
    doc = fixture_to_public_dict(fx)
    raw = json.dumps(doc)
    assert "script" in doc
    assert "ssh_connect" in doc["script"]
    assert "PRIVATE" not in raw
