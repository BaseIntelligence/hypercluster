"""Docker image inspect, healthy run, volume integrity, bind fail, cleanup.

Covers VAL-SCAF-018, 019, 020, 023, 037, 039, 040.
Docker daemon tests are skipped when the daemon is unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_TAG = "hypercluster:m1-scaffold"
# Mission host band (3200–3299). Container-internal remains 8000.
HOST_PORT = 3250
CONTAINER_NAME = "hypercluster-m1-lifecycle"
VOLUME_NAME = "hypercluster-m1-data"
TOKEN = "docker-lifecycle-test-token"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            check=False,
            capture_output=True,
            timeout=20,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


docker_required = pytest.mark.skipif(
    not _docker_available(),
    reason="docker daemon not available",
)


def _run(
    args: list[str],
    *,
    check: bool = True,
    timeout: float = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _cleanup_container(name: str = CONTAINER_NAME) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _cleanup_volume(name: str = VOLUME_NAME) -> None:
    subprocess.run(
        ["docker", "volume", "rm", "-f", name],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _wait_http_ok(url: str, *, timeout_s: float = 60.0) -> dict[str, object]:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                body = response.read().decode("utf-8")
                if response.status == 200:
                    return json.loads(body)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_err = exc
            time.sleep(0.5)
    raise AssertionError(f"URL {url} not healthy within {timeout_s}s: {last_err!r}")


def _wait_container_healthy(name: str, *, timeout_s: float = 90.0) -> str:
    deadline = time.time() + timeout_s
    last = "unknown"
    while time.time() < deadline:
        proc = _run(
            [
                "docker",
                "inspect",
                "--format",
                "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}",
                name,
            ],
            check=False,
        )
        last = (proc.stdout or "").strip()
        if last == "healthy":
            return last
        if last in {"exited", "dead"}:
            logs = _run(["docker", "logs", name], check=False, timeout=30)
            raise AssertionError(
                f"container {name} exited before healthy (state={last}):\n"
                f"{logs.stdout}\n{logs.stderr}"
            )
        time.sleep(1.0)
    logs = _run(["docker", "logs", name], check=False, timeout=30)
    raise AssertionError(
        f"container {name} not healthy within {timeout_s}s (last={last}):\n"
        f"{logs.stdout}\n{logs.stderr}"
    )


@pytest.fixture(scope="module")
def built_image() -> str:
    """Build the challenge image once for this module (cheap-ish layered rebuilds)."""

    if not _docker_available():
        pytest.skip("docker daemon not available")

    # Stage Base wheel into build context so we do not require internet for base.
    vendor = REPO_ROOT / "docker" / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    wheel_src = Path("/projects/platform-network/platform/dist/base-3.1.2-py3-none-any.whl")
    wheel_dst = vendor / "base-3.1.2-py3-none-any.whl"
    if wheel_src.is_file() and (
        not wheel_dst.is_file() or wheel_dst.stat().st_size != wheel_src.stat().st_size
    ):
        shutil.copy2(wheel_src, wheel_dst)

    # Ensure no leftovers pollute free ports.
    _cleanup_container()
    build = _run(
        ["docker", "build", "-t", IMAGE_TAG, str(REPO_ROOT)],
        check=False,
        timeout=600,
    )
    if build.returncode != 0:
        raise AssertionError(
            f"docker build failed:\nSTDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}"
        )
    return IMAGE_TAG


@docker_required
def test_image_healthcheck_and_runtime_config(built_image: str) -> None:
    """VAL-SCAF-018 / 020: inspect HEALTHCHECK, EXPOSE, CMD, non-root user."""

    proc = _run(
        [
            "docker",
            "image",
            "inspect",
            built_image,
            "--format",
            "{{json .Config}}",
        ]
    )
    config = json.loads(proc.stdout)
    health = config.get("Healthcheck") or {}
    test_cmd = " ".join(health.get("Test") or [])
    assert "/health" in test_cmd
    assert "127.0.0.1" in test_cmd
    assert "status" in test_cmd
    assert "ok" in test_cmd

    exposed = config.get("ExposedPorts") or {}
    assert "8000/tcp" in exposed

    user = (config.get("User") or "").strip()
    assert user and user not in {"", "0", "root"}, f"expected non-root user, got {user!r}"

    cmd = config.get("Cmd") or []
    cmd_joined = " ".join(cmd)
    assert "uvicorn" in cmd_joined
    assert "hypercluster.app:app" in cmd_joined
    assert "8000" in cmd_joined


@docker_required
def test_docker_run_becomes_healthy_and_curl_identity(built_image: str) -> None:
    """VAL-SCAF-019: container becomes healthy; host curl /health is ok hypercluster."""

    _cleanup_container()
    _cleanup_volume()
    try:
        _run(
            [
                "docker",
                "volume",
                "create",
                VOLUME_NAME,
            ]
        )
        run = _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                CONTAINER_NAME,
                "-p",
                f"127.0.0.1:{HOST_PORT}:8000",
                "-v",
                f"{VOLUME_NAME}:/data",
                "-e",
                f"CHALLENGE_SHARED_TOKEN={TOKEN}",
                "-e",
                "CHALLENGE_SHARED_TOKEN_FILE=",
                "-e",
                "CHALLENGE_DATABASE_URL=sqlite+aiosqlite:////data/challenge.sqlite3",
                built_image,
            ]
        )
        assert run.returncode == 0, run.stderr
        status = _wait_container_healthy(CONTAINER_NAME, timeout_s=90)
        assert status == "healthy"

        payload = _wait_http_ok(f"http://127.0.0.1:{HOST_PORT}/health", timeout_s=30)
        assert payload.get("status") == "ok"
        assert payload.get("slug") == "hypercluster"
        assert payload.get("role") == "challenge"
        assert payload.get("ready") is True

        # DB file lands under /data inside the container.
        ls = _run(
            ["docker", "exec", CONTAINER_NAME, "ls", "-la", "/data"],
            check=False,
        )
        assert "challenge.sqlite3" in (ls.stdout or "")
    finally:
        _cleanup_container()
        # keep volume for following integrity test whether it ran first or second


@docker_required
def test_data_volume_integrity_across_container_recreate(built_image: str) -> None:
    """VAL-SCAF-023: named /data volume retains sqlite across stop/rm/new run."""

    _cleanup_container()
    # Recreate a clean volume for an isolated integrity probe.
    _cleanup_volume()
    _run(["docker", "volume", "create", VOLUME_NAME])
    marker = "hypercluster-integrity-marker-42"
    try:
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                CONTAINER_NAME,
                "-p",
                f"127.0.0.1:{HOST_PORT}:8000",
                "-v",
                f"{VOLUME_NAME}:/data",
                "-e",
                f"CHALLENGE_SHARED_TOKEN={TOKEN}",
                "-e",
                "CHALLENGE_SHARED_TOKEN_FILE=",
                built_image,
            ]
        )
        _wait_container_healthy(CONTAINER_NAME, timeout_s=90)
        _wait_http_ok(f"http://127.0.0.1:{HOST_PORT}/ready", timeout_s=30)

        # Write a marker alongside the challenge DB; DB file itself must also survive.
        _run(
            [
                "docker",
                "exec",
                CONTAINER_NAME,
                "sh",
                "-c",
                f"echo {marker} > /data/integrity-marker.txt && ls /data",
            ]
        )
        before = _run(["docker", "exec", CONTAINER_NAME, "ls", "/data"])
        assert "challenge.sqlite3" in before.stdout
        assert "integrity-marker.txt" in before.stdout

        _run(["docker", "stop", CONTAINER_NAME], timeout=60)
        _run(["docker", "rm", CONTAINER_NAME], timeout=30)

        # New container, same volume/image.
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                CONTAINER_NAME,
                "-p",
                f"127.0.0.1:{HOST_PORT}:8000",
                "-v",
                f"{VOLUME_NAME}:/data",
                "-e",
                f"CHALLENGE_SHARED_TOKEN={TOKEN}",
                "-e",
                "CHALLENGE_SHARED_TOKEN_FILE=",
                built_image,
            ]
        )
        _wait_container_healthy(CONTAINER_NAME, timeout_s=90)
        _wait_http_ok(f"http://127.0.0.1:{HOST_PORT}/ready", timeout_s=30)

        after = _run(["docker", "exec", CONTAINER_NAME, "cat", "/data/integrity-marker.txt"])
        assert marker in after.stdout
        ls_after = _run(["docker", "exec", CONTAINER_NAME, "ls", "/data"])
        assert "challenge.sqlite3" in ls_after.stdout
    finally:
        _cleanup_container()
        _cleanup_volume()


@docker_required
def test_docker_cleanup_leaves_no_orphans(built_image: str) -> None:
    """VAL-SCAF-040: after lifecycle teardown, validation container/port are gone."""

    _cleanup_container()
    try:
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                CONTAINER_NAME,
                "-p",
                f"127.0.0.1:{HOST_PORT}:8000",
                "-e",
                f"CHALLENGE_SHARED_TOKEN={TOKEN}",
                "-e",
                "CHALLENGE_SHARED_TOKEN_FILE=",
                built_image,
            ]
        )
        _wait_http_ok(f"http://127.0.0.1:{HOST_PORT}/health", timeout_s=90)
    finally:
        _cleanup_container()
        _cleanup_volume()

    ps = _run(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
        check=False,
    )
    names = [line.strip() for line in (ps.stdout or "").splitlines() if line.strip()]
    assert CONTAINER_NAME not in names

    # Host port free again.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        connect_result = sock.connect_ex(("127.0.0.1", HOST_PORT))
    assert connect_result != 0, f"port {HOST_PORT} still accepting connections after cleanup"


def test_port_bind_conflict_fails_visibly() -> None:
    """VAL-SCAF-037: second bind on occupied port fails non-zero; first stays healthy."""

    bind_port = 3251
    env = os.environ.copy()
    env["CHALLENGE_DATABASE_URL"] = f"sqlite+aiosqlite:////tmp/hypercluster-bind-{os.getpid()}.sqlite3"
    env["CHALLENGE_SHARED_TOKEN"] = "bind-conflict-token"
    env["CHALLENGE_SHARED_TOKEN_FILE"] = ""
    env["CHALLENGE_PORT"] = str(bind_port)

    first: subprocess.Popen[str] | None = None
    second: subprocess.Popen[str] | None = None
    try:
        first = subprocess.Popen(
            [
                "uv",
                "run",
                "uvicorn",
                "hypercluster.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(bind_port),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _wait_http_ok(f"http://127.0.0.1:{bind_port}/health", timeout_s=30)

        second = subprocess.Popen(
            [
                "uv",
                "run",
                "uvicorn",
                "hypercluster.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(bind_port),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            second_code = second.wait(timeout=15)
        except subprocess.TimeoutExpired:
            second.kill()
            second.wait(timeout=5)
            raise AssertionError("second bind hung instead of failing visibly") from None

        assert second_code != 0, "second bind must exit non-zero on occupied port"

        # First process still healthy.
        payload = _wait_http_ok(f"http://127.0.0.1:{bind_port}/health", timeout_s=5)
        assert payload.get("status") == "ok"
        assert first.poll() is None, "first process should still be running"
    finally:
        for proc in (second, first):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


def test_proxy_path_identity_optional_skip() -> None:
    """VAL-SCAF-039: proxy path is N/A when master public proxy is not configured."""

    # No Base master proxy harness is supplied in bare-metal M1 scaffold.
    # Skip with an explicit reason so validators can mark N/A.
    pytest.skip(
        "VAL-SCAF-039 N/A: Base master public proxy not configured in this harness; "
        "direct challenge /health identity is covered by VAL-SCAF-001/019"
    )
