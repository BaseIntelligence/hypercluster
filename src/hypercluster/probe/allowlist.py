"""Fixed command_id → argv template registry for GPU probe SSH.

No free-form remote shell. Unknown command_ids are rejected
(:class:`~hypercluster.probe.transport.TransportError`). Design §3.2 / VAL-GPU-030.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypercluster.probe.transport import COMMAND_ALLOWLIST, TransportError

# Open CUDA microbench payload (remote). Receive seed only via fixed script body;
# no user argv injection. DIGEST line contract for pipeline.
_MICROBENCH_REMOTE = r"""
python3 - <<'HYPER_GPU_PROBE_PY'
import hashlib, json, os, time
seed = os.environ.get("HYPER_PROBE_SEED", "0")
t0 = time.time()
# Prefer trivial CPU float work when CUDA unavailable so Fake/SSH hosts still emit digest.
n = 256
acc = 0.0
for i in range(n):
    acc += (i * 1.000001) ** 0.5
elapsed_ms = int((time.time() - t0) * 1000)
digest = "sha256:" + hashlib.sha256(f"{seed}:{acc:.6f}".encode()).hexdigest()
print(json.dumps({"ok": True, "digest": digest, "gflops": 0.0, "elapsed_ms": elapsed_ms}))
HYPER_GPU_PROBE_PY
""".strip()


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Static remote command template."""

    command_id: str
    argv: tuple[str, ...]
    timeout_s: float
    description: str = ""

    def argv_list(self) -> list[str]:
        return list(self.argv)

    def remote_command(self) -> str:
        """Join fixed argv for OpenSSH remote execution (no shell interpolation of user input)."""

        # All templates are constants under our control. Multi-token argv is joined
        # with spaces for the ssh remote command string (already shell-safe literals).
        if len(self.argv) == 1:
            return self.argv[0]
        # Prefer argv[0] as program and remaining as fixed flags.
        parts: list[str] = []
        for p in self.argv:
            if p == "":
                continue
            # Quote when whitespace / metachar present; still only constant registry strings.
            if any(ch in p for ch in (" ", '"', "'", "$", "`", "\\", "\n", ";")):
                parts.append("'" + p.replace("'", "'\"'\"'") + "'")
            else:
                parts.append(p)
        return " ".join(parts)


def _spec(command_id: str, argv: list[str], timeout_s: float, description: str = "") -> CommandSpec:
    return CommandSpec(
        command_id=command_id,
        argv=tuple(argv),
        timeout_s=timeout_s,
        description=description,
    )


# Fixed registry (design §3.2). Never accept caller argv / free-form shell.
COMMAND_REGISTRY: dict[str, CommandSpec] = {
    "ssh_connect": _spec(
        "ssh_connect",
        ["true"],
        15.0,
        "session liveness after auth",
    ),
    "echo_ping": _spec(
        "echo_ping",
        ["echo", "hyper-gpu-probe-ping"],
        10.0,
        "liveness ping",
    ),
    "nvidia_smi_list": _spec(
        "nvidia_smi_list",
        ["nvidia-smi", "-L"],
        30.0,
        "GPU name + UUID list",
    ),
    "nvidia_smi_query": _spec(
        "nvidia_smi_query",
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version,"
            "power.limit,power.default_limit,utilization.gpu,"
            "utilization.memory,clocks.sm",
            "--format=csv,noheader,nounits",
        ],
        30.0,
        "GPU inventory CSV",
    ),
    "cuda_microbench": _spec(
        "cuda_microbench",
        [_MICROBENCH_REMOTE],
        90.0,
        "open CUDA/CPU microbench + digest",
    ),
    "docker_info": _spec(
        "docker_info",
        ["docker", "info", "--format", "{{json .}}"],
        30.0,
        "docker runtime inventory",
    ),
    "docker_gpu_smi": _spec(
        "docker_gpu_smi",
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "nvidia/cuda:12.2.0-base-ubuntu22.04",
            "nvidia-smi",
            "-L",
        ],
        90.0,
        "docker GPU visibility",
    ),
}


def is_allowlisted(command_id: str) -> bool:
    return command_id in COMMAND_ALLOWLIST and command_id in COMMAND_REGISTRY


def get_command_spec(command_id: str) -> CommandSpec:
    if command_id not in COMMAND_REGISTRY:
        raise TransportError(
            "unknown_command_id",
            f"command_id {command_id!r} is not on the allowlist",
        )
    if command_id not in COMMAND_ALLOWLIST:
        raise TransportError(
            "unknown_command_id",
            f"command_id {command_id!r} is not on the allowlist",
        )
    return COMMAND_REGISTRY[command_id]


def argv_for_command(command_id: str) -> list[str]:
    """Return a **copy** of the fixed argv template for ``command_id``."""

    return get_command_spec(command_id).argv_list()


def remote_command_for(command_id: str) -> str:
    return get_command_spec(command_id).remote_command()


def command_timeout_s(command_id: str, *, cap_s: float | None = None) -> float:
    spec = get_command_spec(command_id)
    timeout = float(spec.timeout_s)
    if cap_s is not None and cap_s > 0:
        timeout = min(timeout, float(cap_s))
    return timeout


def unknown_command_ids_rejected(command_ids: list[str]) -> list[str]:
    """Return the subset of ids that are not allowlisted (for engine tests)."""

    return [c for c in command_ids if not is_allowlisted(c)]


def assert_registry_complete() -> None:
    missing = set(COMMAND_ALLOWLIST) - set(COMMAND_REGISTRY)
    extra = set(COMMAND_REGISTRY) - set(COMMAND_ALLOWLIST)
    if missing or extra:
        raise RuntimeError(f"allowlist registry mismatch missing={missing} extra={extra}")


assert_registry_complete()


__all__ = [
    "COMMAND_REGISTRY",
    "CommandSpec",
    "argv_for_command",
    "assert_registry_complete",
    "command_timeout_s",
    "get_command_spec",
    "is_allowlisted",
    "remote_command_for",
    "unknown_command_ids_rejected",
]
