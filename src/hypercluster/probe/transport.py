"""SSH transport protocol + FakeSsh (no real network).

Real allowlist executor lives in ``hypercluster.probe.ssh_exec`` /
``allowlist`` (``m9-ssh-allowlist-executor``). This module defines the
transport protocol the ordered check pipeline depends on, plus a
deterministic FakeSsh used by unit tests and the fixture bank.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# Fixed allowlist of command_ids the probe may request. Real executor maps
# each id → fixed argv; free-form remote shell is never accepted.
COMMAND_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ssh_connect",
        "nvidia_smi_list",
        "nvidia_smi_query",
        "cuda_microbench",
        "docker_info",
        "docker_gpu_smi",
        "echo_ping",
    }
)

CommandId = Literal[
    "ssh_connect",
    "nvidia_smi_list",
    "nvidia_smi_query",
    "cuda_microbench",
    "docker_info",
    "docker_gpu_smi",
    "echo_ping",
]


class TransportError(Exception):
    """Transport-level failure (connect timeout, unknown command, etc.)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class SshCommandResult:
    """Result of one allowlisted remote step."""

    command_id: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None


@runtime_checkable
class SshTransport(Protocol):
    """Pluggable SSH / FakeSsh transport (design §3 + §6)."""

    name: str  # "real" | "fake"

    def connect(self) -> SshCommandResult:
        """Establish session (auth). Fatal when not ok."""
        ...

    def run(self, command_id: str, *, timeout_s: float | None = None) -> SshCommandResult:
        """Run allowlisted command id; raise TransportError on unknown id."""
        ...

    def close(self) -> None:
        """Close session; idempotent."""
        ...


@dataclass
class FakeOutcome:
    """Scripted outcome for one FakeSsh command_id."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 5
    timed_out: bool = False
    error: str | None = None
    # When true, connect() fails before any command runs.
    fail_connect: bool = False


def _default_v100_inventory() -> dict[str, Any]:
    """Measured inventory fields embedded in FakeSsh pass fixtures."""

    return {
        "gpus": [
            {
                "name": "Tesla V100-SXM2-16GB",
                "uuid": "GPU-11111111-1111-1111-1111-111111111111",
                "memory_total_mb": 16160,
                "driver_version": "535.104.05",
                "power_limit_w": 300.0,
                "power_default_w": 300.0,
                "util_gpu": 0.0,
                "util_mem": 0.0,
                "clocks_sm_mhz": 0.0,
            }
        ],
        "cuda_runtime_hint": "12.2",
        "docker": {
            "present": True,
            "runtimes": ["runc", "nvidia"],
            "gpu_in_container": True,
        },
    }


def build_pass_script(
    *,
    gpus: list[dict[str, Any]] | None = None,
    docker: dict[str, Any] | None = None,
    received: dict[str, FakeOutcome] | None = None,
) -> dict[str, FakeOutcome]:
    """Build a complete FakeSsh pass-all script for a synthetic host."""

    inv = _default_v100_inventory()
    if gpus is not None:
        inv["gpus"] = gpus
    if docker is not None:
        inv["docker"] = docker

    smi_l_lines = []
    query_lines = []
    for idx, gpu in enumerate(inv["gpus"]):
        name = gpu["name"]
        uuid = gpu["uuid"]
        smi_l_lines.append(f"GPU {idx}: {name} (UUID: {uuid})")
        mem = gpu.get("memory_total_mb", 16160)
        driver = gpu.get("driver_version", "535.104.05")
        pl = gpu.get("power_limit_w", 300.0)
        pd = gpu.get("power_default_w", 300.0)
        ug = gpu.get("util_gpu", 0.0)
        um = gpu.get("util_mem", 0.0)
        clocks = gpu.get("clocks_sm_mhz", 0.0)
        query_lines.append(f"{name}, {uuid}, {mem}, {driver}, {pl}, {pd}, {ug}, {um}, {clocks}")

    docker_info = {
        "Runtimes": {rt: {} for rt in inv["docker"].get("runtimes", ["runc", "nvidia"])},
        "Name": "fake-host",
    }

    script: dict[str, FakeOutcome] = {
        "ssh_connect": FakeOutcome(exit_code=0, stdout="ok"),
        "echo_ping": FakeOutcome(exit_code=0, stdout="hyper-gpu-probe-ping"),
        "nvidia_smi_list": FakeOutcome(
            exit_code=0,
            stdout="\n".join(smi_l_lines) + "\n",
        ),
        "nvidia_smi_query": FakeOutcome(
            exit_code=0,
            stdout="\n".join(query_lines) + "\n",
        ),
        "cuda_microbench": FakeOutcome(
            exit_code=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "digest": "sha256:deadbeefcafebabe00000000000000000000000000000001",
                    "gflops": 1200.0,
                    "elapsed_ms": 42,
                }
            )
            + "\n",
        ),
        "docker_info": FakeOutcome(
            exit_code=0 if inv["docker"].get("present", True) else 127,
            stdout=json.dumps(docker_info) if inv["docker"].get("present", True) else "",
            stderr="" if inv["docker"].get("present", True) else "docker: not found",
        ),
        "docker_gpu_smi": FakeOutcome(
            exit_code=0 if inv["docker"].get("gpu_in_container", True) else 1,
            stdout="\n".join(smi_l_lines) + "\n"
            if inv["docker"].get("gpu_in_container", True)
            else "",
        ),
    }
    if received:
        script.update(received)
    return script


@dataclass
class FakeSshTransport:
    """Deterministic FakeSsh transport (CI only; never silent-prod).

    * ``scripted`` maps ``command_id`` → :class:`FakeOutcome`.
    * Missing scripted ids default to exit 0 empty stdout (callers should
      still build explicit fixtures for fail cases).
    * Unknown command_ids (outside COMMAND_ALLOWLIST) raise TransportError.
    """

    scripted: dict[str, FakeOutcome] = field(default_factory=dict)
    name: str = "fake"
    _connected: bool = field(default=False, init=False, repr=False)
    _commands_run: list[str] = field(default_factory=list, init=False, repr=False)

    def connect(self) -> SshCommandResult:
        outcome = self.scripted.get("ssh_connect", FakeOutcome())
        if outcome.fail_connect or outcome.timed_out or outcome.error:
            self._connected = False
            return SshCommandResult(
                command_id="ssh_connect",
                exit_code=outcome.exit_code if outcome.exit_code != 0 else 1,
                stdout=outcome.stdout,
                stderr=outcome.stderr or outcome.error or "connect failed",
                duration_ms=outcome.duration_ms,
                timed_out=outcome.timed_out,
                error=outcome.error or ("timeout" if outcome.timed_out else "connect_failed"),
            )
        if outcome.exit_code != 0:
            self._connected = False
            return SshCommandResult(
                command_id="ssh_connect",
                exit_code=outcome.exit_code,
                stdout=outcome.stdout,
                stderr=outcome.stderr or "ssh connect failed",
                duration_ms=outcome.duration_ms,
                timed_out=False,
                error="connect_failed",
            )
        self._connected = True
        self._commands_run.append("ssh_connect")
        return SshCommandResult(
            command_id="ssh_connect",
            exit_code=0,
            stdout=outcome.stdout or "ok",
            stderr=outcome.stderr,
            duration_ms=outcome.duration_ms,
        )

    def run(self, command_id: str, *, timeout_s: float | None = None) -> SshCommandResult:
        del timeout_s  # FakeSsh ignores wall-clock; outcomes carry timed_out.
        if command_id not in COMMAND_ALLOWLIST:
            raise TransportError(
                "unknown_command_id",
                f"command_id {command_id!r} is not on the allowlist",
            )
        if command_id == "ssh_connect":
            return self.connect()
        if not self._connected:
            return SshCommandResult(
                command_id=command_id,
                exit_code=255,
                stderr="not connected",
                error="not_connected",
            )
        outcome = self.scripted.get(command_id, FakeOutcome())
        self._commands_run.append(command_id)
        return SshCommandResult(
            command_id=command_id,
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            duration_ms=outcome.duration_ms,
            timed_out=outcome.timed_out,
            error=outcome.error,
        )

    def close(self) -> None:
        self._connected = False

    @property
    def commands_run(self) -> list[str]:
        return list(self._commands_run)


__all__ = [
    "COMMAND_ALLOWLIST",
    "CommandId",
    "FakeOutcome",
    "FakeSshTransport",
    "SshCommandResult",
    "SshTransport",
    "TransportError",
    "build_pass_script",
]
