"""Real allowlist SshExecutor (sync OpenSSH CLI; optional integration).

- Maps ``command_id`` → fixed argv only (VAL-GPU-030)
- Per-command + wall-budget timeouts; concurrent probe mutex per node
- Private key PEM never appears in evidence JSON / logs (VAL-GPU-031)
- Default CI still uses FakeSsh; Real is opt-in under ``HYPER_SSH_TRANSPORT=real``

Does **not** shell out free-form remote commands from API bodies.
Does **not** import Verda. Never ``set_weights``.
"""

from __future__ import annotations

import shlex
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hypercluster.probe.allowlist import (
    argv_for_command,
    command_timeout_s,
    is_allowlisted,
    remote_command_for,
)
from hypercluster.probe.keys import (
    KeyMaterialError,
    KeyRef,
    compute_key_fingerprint,
    key_ref_from_settings,
    public_key_meta_for_evidence,
    resolve_key_ref,
)
from hypercluster.probe.redact import OUTPUT_CAP_DEFAULT, sanitize_output
from hypercluster.probe.transport import SshCommandResult, TransportError
from hypercluster.settings import HyperSettings

# Runner protocol: returns (exit_code, stdout, stderr, timed_out)
RunnerFn = Callable[..., tuple[int, str, str, bool]]


@dataclass(slots=True)
class RealSshTarget:
    host: str
    port: int = 22
    username: str = "root"
    key_path: str | None = None
    key_fingerprint: str = ""
    key_ref: KeyRef | None = None
    # Connect options
    strict_host_key_checking: bool = False


class NodeProbeLock:
    """Per-node mutex so concurrent probes on the same node serialize (design §9)."""

    def __init__(self) -> None:
        self._guards: dict[str, threading.Lock] = {}
        self._meta = threading.Lock()

    def _lock_for(self, node_id: str) -> threading.Lock:
        with self._meta:
            if node_id not in self._guards:
                self._guards[node_id] = threading.Lock()
            return self._guards[node_id]

    @contextmanager
    def acquire(self, node_id: str, *, timeout_s: float | None = None) -> Iterator[None]:
        lock = self._lock_for(node_id)
        if timeout_s is None:
            lock.acquire()
            try:
                yield
            finally:
                lock.release()
            return
        got = lock.acquire(timeout=float(timeout_s))
        if not got:
            raise TimeoutError(f"probe mutex busy for node {node_id!r}")
        try:
            yield
        finally:
            lock.release()


# Process-wide lock manager (tests may construct their own).
DEFAULT_NODE_PROBE_LOCK = NodeProbeLock()


def default_openssh_runner(
    *,
    target: RealSshTarget,
    remote_cmd: str,
    timeout_s: float,
) -> tuple[int, str, str, bool]:
    """Execute a fixed remote command via OpenSSH ``ssh`` binary.

    ``remote_cmd`` must come only from the allowlist registry.
    """

    if remote_cmd == "__connect__":
        # Auth-only: run ``true`` under the same identity.
        remote_cmd = "true"

    identity: list[str] = []
    if target.key_path:
        identity = ["-i", target.key_path]

    strict = "accept-new" if not target.strict_host_key_checking else "yes"
    cmd = [
        "ssh",
        "-p",
        str(target.port),
        *identity,
        "-o",
        f"StrictHostKeyChecking={strict}",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        f"ConnectTimeout={max(1, int(timeout_s))}",
        f"{target.username}@{target.host}",
        remote_cmd,
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
        return proc.returncode, proc.stdout or "", proc.stderr or "", False
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout
            if isinstance(exc.stdout, str)
            else (exc.stdout or b"").decode("utf-8", errors="replace")
        )
        stderr = (
            exc.stderr
            if isinstance(exc.stderr, str)
            else (exc.stderr or b"").decode("utf-8", errors="replace")
        )
        return 124, stdout or "", stderr or "ssh timeout", True
    except OSError as exc:
        return 255, "", f"ssh exec error: {exc}", False


@dataclass
class RealSshExecutor:
    """Allowlist-only real SSH transport implementing :class:`SshTransport`."""

    target: RealSshTarget
    name: str = "real"
    runner: RunnerFn | None = None
    connected: bool = False
    connect_timeout_s: float = 15.0
    cmd_timeout_cap_s: float = 90.0
    wall_budget_s: float = 180.0
    wall_spent_s: float = 0.0
    output_cap_bytes: int = OUTPUT_CAP_DEFAULT
    _started_at: float = field(default_factory=time.monotonic, init=False, repr=False)
    _commands_run: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.target.key_fingerprint and self.target.key_path:
            try:
                pem = Path(self.target.key_path).read_bytes()
                self.target.key_fingerprint = compute_key_fingerprint(pem)
            except OSError:
                pass

    @property
    def key_fingerprint(self) -> str:
        return self.target.key_fingerprint or ""

    def evidence_transport_meta(self) -> dict[str, Any]:
        """Public metadata for evidence (fingerprint + key_ref only)."""

        ref = self.target.key_ref
        if ref is None and self.target.key_path:
            ref = KeyRef(kind="file", name=self.target.key_path)
        if ref is None:
            return {
                "key_fingerprint": self.key_fingerprint,
                "key_ref": None,
                "transport": "real",
                "ssh_endpoint": f"{self.target.host}:{self.target.port}",
            }
        meta = public_key_meta_for_evidence(ref, fingerprint=self.key_fingerprint)
        meta["transport"] = "real"
        meta["ssh_endpoint"] = f"{self.target.host}:{self.target.port}"
        return meta

    def _remaining_wall(self) -> float:
        spent = self.wall_spent_s + max(0.0, time.monotonic() - self._started_at)
        return float(self.wall_budget_s) - spent

    def _invoke(
        self,
        *,
        remote_cmd: str,
        timeout_s: float,
        command_id: str,
    ) -> SshCommandResult:
        remaining = self._remaining_wall()
        if remaining <= 0:
            return SshCommandResult(
                command_id=command_id,
                exit_code=124,
                stderr="wall budget exceeded",
                timed_out=True,
                error="wall_budget_exceeded",
            )
        effective = min(float(timeout_s), remaining, float(self.cmd_timeout_cap_s))
        if effective <= 0:
            return SshCommandResult(
                command_id=command_id,
                exit_code=124,
                stderr="wall budget exceeded",
                timed_out=True,
                error="wall_budget_exceeded",
            )

        fn = self.runner
        t0 = time.perf_counter()
        if fn is None:
            exit_code, stdout, stderr, timed_out = default_openssh_runner(
                target=self.target,
                remote_cmd=remote_cmd,
                timeout_s=effective,
            )
        else:
            # Call with keywords that tests may accept via **kwargs
            exit_code, stdout, stderr, timed_out = fn(
                remote_cmd=remote_cmd,
                timeout_s=effective,
                target=self.target,
                command_id=command_id,
                remote_argv=list(shlex.split(remote_cmd)) if remote_cmd != "__connect__" else [],
                argv=list(shlex.split(remote_cmd)) if remote_cmd != "__connect__" else [],
            )
        duration_ms = int((time.perf_counter() - t0) * 1000)
        self.wall_spent_s += time.perf_counter() - t0
        stdout_s = sanitize_output(stdout, max_bytes=self.output_cap_bytes)
        stderr_s = sanitize_output(stderr, max_bytes=self.output_cap_bytes)
        err: str | None = None
        if timed_out:
            err = "timeout"
        elif exit_code != 0:
            err = "command_failed"
        return SshCommandResult(
            command_id=command_id,
            exit_code=int(exit_code),
            stdout=stdout_s,
            stderr=stderr_s,
            duration_ms=duration_ms,
            timed_out=bool(timed_out),
            error=err,
        )

    def connect(self) -> SshCommandResult:
        res = self._invoke(
            remote_cmd="__connect__",
            timeout_s=self.connect_timeout_s,
            command_id="ssh_connect",
        )
        if res.ok:
            self.connected = True
            self._commands_run.append("ssh_connect")
        else:
            self.connected = False
        return res

    def run(self, command_id: str, *, timeout_s: float | None = None) -> SshCommandResult:
        if not is_allowlisted(command_id):
            raise TransportError(
                "unknown_command_id",
                f"command_id {command_id!r} is not on the allowlist",
            )
        # Touch argv builder so unknown-id and registry mutation tests stay coupled.
        _ = argv_for_command(command_id)

        if command_id == "ssh_connect":
            return self.connect()

        if not self.connected:
            return SshCommandResult(
                command_id=command_id,
                exit_code=255,
                stderr="not connected",
                error="not_connected",
            )

        default_t = command_timeout_s(command_id, cap_s=self.cmd_timeout_cap_s)
        effective = default_t if timeout_s is None else min(float(timeout_s), default_t)
        remote = remote_command_for(command_id)
        res = self._invoke(remote_cmd=remote, timeout_s=effective, command_id=command_id)
        self._commands_run.append(command_id)
        return res

    def close(self) -> None:
        self.connected = False

    @property
    def commands_run(self) -> list[str]:
        return list(self._commands_run)


def parse_ssh_endpoint(endpoint: str) -> tuple[str, int]:
    """Parse ``host``, ``host:port``, or ``[ipv6]:port`` into (host, port)."""

    text = (endpoint or "").strip()
    if not text:
        raise ValueError("empty ssh endpoint")
    if text.startswith("["):
        # [ipv6]:port
        end = text.find("]")
        if end == -1:
            raise ValueError(f"invalid ssh endpoint: {endpoint!r}")
        host = text[1:end]
        rest = text[end + 1 :]
        if rest.startswith(":"):
            return host, int(rest[1:] or "22")
        return host, 22
    if text.count(":") == 1:
        host, _, port_s = text.partition(":")
        return host, int(port_s or "22")
    return text, 22


def build_real_ssh_transport(
    settings: HyperSettings,
    *,
    host: str,
    port: int = 22,
    username: str = "root",
    key_ref: KeyRef | None = None,
    runner: RunnerFn | None = None,
) -> RealSshExecutor:
    """Construct a RealSshExecutor from settings + target host.

    Raises :class:`KeyMaterialError` when key cannot be resolved.
    """

    ref = key_ref or key_ref_from_settings(
        ssh_key_path=getattr(settings, "ssh_key_path", None),
        ssh_key_env=getattr(settings, "ssh_key_env", None),
    )
    if ref is None:
        raise KeyMaterialError(
            "key_not_found",
            "real SSH requires HYPER_SSH_KEY_PATH or key_ref",
        )
    resolved = resolve_key_ref(ref)
    key_path = str(resolved.path) if resolved.path is not None else None
    # When PEM-only env, write is avoided; runner may be mocked in tests.
    # Live OpenSSH path still needs a file — require path for default runner.
    if key_path is None and runner is None:
        raise KeyMaterialError(
            "key_not_found",
            "real SSH default runner requires a file-backed key (HYPER_SSH_KEY_PATH)",
        )

    target = RealSshTarget(
        host=host,
        port=port,
        username=username,
        key_path=key_path,
        key_fingerprint=resolved.fingerprint,
        key_ref=ref,
    )
    return RealSshExecutor(
        target=target,
        runner=runner,
        connect_timeout_s=float(getattr(settings, "ssh_connect_timeout_s", 15.0) or 15.0),
        cmd_timeout_cap_s=float(getattr(settings, "ssh_cmd_timeout_s", 90.0) or 90.0),
        wall_budget_s=float(getattr(settings, "gpu_probe_timeout_s", 180) or 180),
        output_cap_bytes=int(
            getattr(settings, "ssh_output_cap_bytes", OUTPUT_CAP_DEFAULT) or OUTPUT_CAP_DEFAULT
        ),
    )


__all__ = [
    "DEFAULT_NODE_PROBE_LOCK",
    "NodeProbeLock",
    "RealSshExecutor",
    "RealSshTarget",
    "RunnerFn",
    "build_real_ssh_transport",
    "default_openssh_runner",
    "parse_ssh_endpoint",
]
