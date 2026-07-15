"""FakeSsh fixture bank for CI GPU-probe gates (M9 design §6).

Named fixtures script every fatal gate without real GPUs or live SSH.
JSON copies under ``tests/fixtures/gpu_probe/`` mirror these builders for
human inspection and for ``HYPER_FAKE_SSH_SCRIPT`` path loads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hypercluster.probe.transport import FakeOutcome, build_pass_script
from hypercluster.probe.types import ClaimedInventory

# Stable UUIDs used across the bank (design matrix + uniqueness tests).
V100_UUID = "GPU-11111111-1111-1111-1111-111111111111"
V100_UUID_B = "GPU-22222222-2222-2222-2222-222222222222"
A100_UUID = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
PEER_CLONE_UUID = "GPU-cccccccc-cccc-cccc-cccc-cccccccccccc"

# Fixture name aliases accepted by loaders (AGENTS.md + design).
FIXTURE_ALIASES: dict[str, str] = {
    "pass-all": "pass_all",
    "pass_all": "pass_all",
    "v100_pass_all": "pass_all",
    "no_gpu": "no_gpu",
    "wrong_model": "wrong_model",
    "uuid_clone": "uuid_clone",
    "vram_lie": "vram_lie",
    "bench_fail": "bench_fail",
    "docker_missing": "docker_missing",
    "ssh_timeout": "ssh_timeout",
    "fingerprint_churn": "fingerprint_churn",
}

KNOWN_FIXTURE_NAMES: frozenset[str] = frozenset(
    {
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
)


@dataclass(slots=True)
class FakeSshFixture:
    """One nameable CI scenario for the ordered GPU probe pipeline."""

    name: str
    scripted: dict[str, FakeOutcome]
    claimed: ClaimedInventory = field(
        default_factory=lambda: ClaimedInventory(gpu_model="1V100.6V", gpu_count=1)
    )
    # Peer-node UUIDs already claimed (uuid_clone uniqueness gate).
    occupied_uuids: set[str] = field(default_factory=set)
    # Prior verified UUID set on *this* node (fingerprint_stable gate).
    prior_verified_uuids: set[str] | None = None
    require_docker_runtime: bool = True
    # Free-form notes for docs / JSON dump.
    description: str = ""
    expected_failure_code: str | None = None
    expected_status: str = "passed"


def _v100_gpu(
    *,
    uuid: str = V100_UUID,
    memory_total_mb: int = 16160,
    name: str = "Tesla V100-SXM2-16GB",
) -> dict[str, Any]:
    return {
        "name": name,
        "uuid": uuid,
        "memory_total_mb": memory_total_mb,
        "driver_version": "535.104.05",
        "power_limit_w": 300.0,
        "power_default_w": 300.0,
        "util_gpu": 0.0,
        "util_mem": 0.0,
        "clocks_sm_mhz": 0.0,
    }


def _a100_gpu(*, uuid: str = A100_UUID) -> dict[str, Any]:
    return {
        "name": "NVIDIA A100-SXM4-40GB",
        "uuid": uuid,
        "memory_total_mb": 40960,
        "driver_version": "535.104.05",
        "power_limit_w": 400.0,
        "power_default_w": 400.0,
        "util_gpu": 0.0,
        "util_mem": 0.0,
        "clocks_sm_mhz": 0.0,
    }


def fixture_pass_all() -> FakeSshFixture:
    return FakeSshFixture(
        name="pass_all",
        description="One Tesla V100, stable UUID, microbench ok, docker nvidia",
        scripted=build_pass_script(gpus=[_v100_gpu()]),
        claimed=ClaimedInventory(gpu_model="1V100.6V", gpu_count=1),
        expected_failure_code=None,
        expected_status="passed",
    )


def fixture_no_gpu() -> FakeSshFixture:
    script = build_pass_script(gpus=[_v100_gpu()])
    script["nvidia_smi_list"] = FakeOutcome(
        exit_code=127,
        stderr="nvidia-smi: command not found",
        error="not found",
        stdout="",
    )
    script["nvidia_smi_query"] = FakeOutcome(
        exit_code=127,
        stderr="nvidia-smi: command not found",
        error="not found",
    )
    return FakeSshFixture(
        name="no_gpu",
        description="nvidia-smi unavailable / empty → fatal nvidia_smi_list",
        scripted=script,
        expected_failure_code="nvidia_smi_list",
        expected_status="failed",
    )


def fixture_wrong_model() -> FakeSshFixture:
    return FakeSshFixture(
        name="wrong_model",
        description="Host reports A100 while claim is V100 → gpu_model_match fail",
        scripted=build_pass_script(gpus=[_a100_gpu()]),
        claimed=ClaimedInventory(gpu_model="1V100.6V", gpu_count=1),
        expected_failure_code="gpu_model_match",
        expected_status="failed",
    )


def fixture_uuid_clone() -> FakeSshFixture:
    # Host reports a UUID already held by a peer node.
    return FakeSshFixture(
        name="uuid_clone",
        description="UUID collides with seeded peer → gpu_uuid_unique fail",
        scripted=build_pass_script(gpus=[_v100_gpu(uuid=PEER_CLONE_UUID)]),
        claimed=ClaimedInventory(gpu_model="1V100.6V", gpu_count=1),
        occupied_uuids={PEER_CLONE_UUID},
        expected_failure_code="gpu_uuid_unique",
        expected_status="failed",
    )


def fixture_vram_lie() -> FakeSshFixture:
    return FakeSshFixture(
        name="vram_lie",
        description="2GB total for V100 class → vram_window fail",
        scripted=build_pass_script(gpus=[_v100_gpu(memory_total_mb=2048)]),
        claimed=ClaimedInventory(gpu_model="1V100.6V", gpu_count=1),
        expected_failure_code="vram_window",
        expected_status="failed",
    )


def fixture_bench_fail() -> FakeSshFixture:
    script = build_pass_script(gpus=[_v100_gpu()])
    script["cuda_microbench"] = FakeOutcome(
        exit_code=1,
        stdout=json.dumps(
            {
                "ok": False,
                "digest": "sha256:badbeef00000000000000000000000000000000000000000",
                "gflops": 0.0,
                "elapsed_ms": 1,
            }
        )
        + "\n",
        stderr="CUDA error: no device",
        error="bench_failed",
    )
    return FakeSshFixture(
        name="bench_fail",
        description="microbench non-zero exit / bad result → cuda_microbench fail",
        scripted=script,
        expected_failure_code="cuda_microbench",
        expected_status="failed",
    )


def fixture_docker_missing() -> FakeSshFixture:
    script = build_pass_script(
        gpus=[_v100_gpu()],
        docker={"present": False, "runtimes": [], "gpu_in_container": False},
    )
    script["docker_info"] = FakeOutcome(
        exit_code=127,
        stderr="docker: not found",
        error="not found",
        stdout="",
    )
    script["docker_gpu_smi"] = FakeOutcome(
        exit_code=127,
        stderr="docker: not found",
        error="not found",
    )
    return FakeSshFixture(
        name="docker_missing",
        description="docker absent with require_docker_runtime → docker_runtime fail",
        scripted=script,
        require_docker_runtime=True,
        expected_failure_code="docker_runtime",
        expected_status="failed",
    )


def fixture_ssh_timeout() -> FakeSshFixture:
    script = build_pass_script(gpus=[_v100_gpu()])
    script["ssh_connect"] = FakeOutcome(
        exit_code=1,
        timed_out=True,
        error="timeout",
        stderr="ssh: connect to host timed out",
        duration_ms=180_000,
    )
    return FakeSshFixture(
        name="ssh_timeout",
        description="connect hang past budget → ssh_connect error/failed",
        scripted=script,
        expected_failure_code="ssh_connect",
        expected_status="error",
    )


def fixture_fingerprint_churn() -> FakeSshFixture:
    # Prior good UUID set was V100_UUID; host now reports a different set.
    return FakeSshFixture(
        name="fingerprint_churn",
        description="second run new UUID set → fingerprint_stable fail",
        scripted=build_pass_script(gpus=[_v100_gpu(uuid=V100_UUID_B)]),
        claimed=ClaimedInventory(gpu_model="1V100.6V", gpu_count=1),
        prior_verified_uuids={V100_UUID},
        expected_failure_code="fingerprint_stable",
        expected_status="failed",
    )


_BUILDERS: dict[str, Any] = {
    "pass_all": fixture_pass_all,
    "no_gpu": fixture_no_gpu,
    "wrong_model": fixture_wrong_model,
    "uuid_clone": fixture_uuid_clone,
    "vram_lie": fixture_vram_lie,
    "bench_fail": fixture_bench_fail,
    "docker_missing": fixture_docker_missing,
    "ssh_timeout": fixture_ssh_timeout,
    "fingerprint_churn": fixture_fingerprint_churn,
}


def normalize_fixture_name(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "_")
    if key not in FIXTURE_ALIASES:
        raise KeyError(f"unknown FakeSsh fixture {name!r}")
    return FIXTURE_ALIASES[key]


def get_fixture(name: str) -> FakeSshFixture:
    """Return a named fixture from the in-process CI bank."""

    canonical = normalize_fixture_name(name)
    return _BUILDERS[canonical]()


def list_fixtures() -> list[str]:
    return sorted(KNOWN_FIXTURE_NAMES)


def outcome_to_dict(outcome: FakeOutcome) -> dict[str, Any]:
    return {
        "exit_code": outcome.exit_code,
        "stdout": outcome.stdout,
        "stderr": outcome.stderr,
        "duration_ms": outcome.duration_ms,
        "timed_out": outcome.timed_out,
        "error": outcome.error,
        "fail_connect": outcome.fail_connect,
    }


def outcome_from_dict(data: dict[str, Any]) -> FakeOutcome:
    return FakeOutcome(
        exit_code=int(data.get("exit_code", 0)),
        stdout=str(data.get("stdout") or ""),
        stderr=str(data.get("stderr") or ""),
        duration_ms=int(data.get("duration_ms") or 5),
        timed_out=bool(data.get("timed_out", False)),
        error=data.get("error"),
        fail_connect=bool(data.get("fail_connect", False)),
    )


def fixture_to_public_dict(fx: FakeSshFixture) -> dict[str, Any]:
    """JSON-serializable document matching tests/fixtures/gpu_probe/*.json."""

    return {
        "fixture_id": fx.name,
        "description": fx.description,
        "claimed": fx.claimed.model_dump(mode="json"),
        "occupied_uuids": sorted(fx.occupied_uuids),
        "prior_verified_uuids": (
            None if fx.prior_verified_uuids is None else sorted(fx.prior_verified_uuids)
        ),
        "require_docker_runtime": fx.require_docker_runtime,
        "expected_status": fx.expected_status,
        "expected_failure_code": fx.expected_failure_code,
        "script": {cid: outcome_to_dict(out) for cid, out in fx.scripted.items()},
    }


def fixture_from_dict(data: dict[str, Any]) -> FakeSshFixture:
    claimed_raw = data.get("claimed") or {"gpu_model": "1V100.6V", "gpu_count": 1}
    claimed = ClaimedInventory(
        gpu_model=str(claimed_raw.get("gpu_model") or "unknown"),
        gpu_count=int(claimed_raw.get("gpu_count") or 0),
    )
    script_raw = data.get("script") or {}
    scripted = {
        str(cid): outcome_from_dict(out if isinstance(out, dict) else {})
        for cid, out in script_raw.items()
    }
    occupied = {str(u) for u in (data.get("occupied_uuids") or []) if u}
    prior_raw = data.get("prior_verified_uuids")
    prior: set[str] | None
    if prior_raw is None:
        prior = None
    else:
        prior = {str(u) for u in prior_raw if u}
    return FakeSshFixture(
        name=str(data.get("fixture_id") or data.get("name") or "custom"),
        description=str(data.get("description") or ""),
        scripted=scripted,
        claimed=claimed,
        occupied_uuids=occupied,
        prior_verified_uuids=prior,
        require_docker_runtime=bool(data.get("require_docker_runtime", True)),
        expected_failure_code=data.get("expected_failure_code"),
        expected_status=str(data.get("expected_status") or "failed"),
    )


def load_fixture_json(path: str | Path) -> FakeSshFixture:
    """Load a FakeSsh fixture from a JSON path (HYPER_FAKE_SSH_SCRIPT)."""

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"fixture file {p} must be a JSON object")
    return fixture_from_dict(data)


def package_fixture_dir() -> Path:
    """Directory of vendored JSON fixtures (repo tests path when present)."""

    # Prefer the repository tests fixtures when running from a worktree.
    here = Path(__file__).resolve()
    # src/hypercluster/probe/fixtures.py → repo root is parents[3]
    repo_root = here.parents[3]
    candidate = repo_root / "tests" / "fixtures" / "gpu_probe"
    return candidate


def load_named_fixture(
    name: str,
    *,
    prefer_json: bool = False,
    fixture_dir: Path | None = None,
) -> FakeSshFixture:
    """Resolve a named bank fixture; optionally load JSON when present."""

    canonical = normalize_fixture_name(name)
    if prefer_json:
        base = fixture_dir or package_fixture_dir()
        # Accept both design names and aliases for files.
        candidates: list[Path | None] = [base / f"{canonical}.json"]
        if canonical == "pass_all":
            candidates.append(base / "v100_pass_all.json")
            candidates.append(base / f"v100_{canonical}.json")
        for path in candidates:
            if path is not None and path.is_file():
                return load_fixture_json(path)
    return get_fixture(canonical)


__all__ = [
    "A100_UUID",
    "FIXTURE_ALIASES",
    "KNOWN_FIXTURE_NAMES",
    "PEER_CLONE_UUID",
    "V100_UUID",
    "V100_UUID_B",
    "FakeSshFixture",
    "fixture_from_dict",
    "fixture_to_public_dict",
    "get_fixture",
    "list_fixtures",
    "load_fixture_json",
    "load_named_fixture",
    "normalize_fixture_name",
    "outcome_from_dict",
    "outcome_to_dict",
    "package_fixture_dir",
]
