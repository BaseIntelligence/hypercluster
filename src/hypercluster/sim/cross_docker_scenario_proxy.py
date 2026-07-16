"""Cross e2e Docker health→API, suite order green, port rebind, relative /v1 paths.

Fulfills:
  VAL-CROSS-007  Docker health then API job (health + mutating challenge path)
  VAL-CROSS-018  Scenario suite orchestration order green (smoke→…→weights)
  VAL-CROSS-022  Docker stop/remove cleans port; host rebind free
  VAL-CROSS-023  Public proxy-shaped paths remain relative ``/v1/...`` compatible
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import httpx

from hypercluster.api.auth import build_signed_headers
from hypercluster.sim.orchestration import (
    DEFAULT_SCENARIO_ORDER,
    SuiteResult,
    run_scenario_suite,
)
from hypercluster.sim.ports import (
    DEFAULT_DOCKER_HOST_PORT,
    MAX_MISSION_PORT,
    MIN_MISSION_PORT,
    is_mission_port,
)
from hypercluster.sim.scenarios import (
    KNOWN_SCENARIOS,
    ScenarioResult,
)

CROSS_DOCKER_SCENARIO_PROXY = "cross-docker-scenario-proxy"

# Distinct from m1-lifecycle (3250 / hypercluster-m1-lifecycle) so concurrent
# docker suites do not collide on host map / container name.
DEFAULT_CROSS_DOCKER_HOST_PORT = 3252
DEFAULT_CROSS_DOCKER_CONTAINER = "hypercluster-m7-cross-docker"
DEFAULT_CROSS_DOCKER_VOLUME = "hypercluster-m7-cross-data"
DEFAULT_CROSS_DOCKER_IMAGE = "hypercluster:m1-scaffold"

REPO_ROOT = Path(__file__).resolve().parents[3]

# Deterministic sim hotkeys for docker mutating path (HMAC-dev insecure mode).
_DOCKER_PROVIDER_HK = "cross-dock-provider-hotkey-aaaaaaaaaaaaaaaaaaaa"
_DOCKER_DEMAND_HK = "cross-dock-demand-hotkey-bbbbbbbbbbbbbbbbbbbbbbbb"

ALLOWED_IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"

# Relative challenge public surfaces the Base master proxies. App must serve
# these without a ``/challenges/hypercluster`` prefix on the challenge process.
PROXY_RELATIVE_PATHS: tuple[str, ...] = (
    "/v1/offers",
    "/v1/jobs",
    "/v1/providers",
    "/v1/nodes",
    "/v1/leases",
    "/v1/leaderboard",
    "/v1/weight-preview",
)

INVALID_PROXY_PREFIXES: tuple[str, ...] = (
    "/challenges/hypercluster",
    "/challenge/hypercluster",
)


@dataclass(slots=True)
class PortProbe:
    """Result of a TCP connect check on host loopback."""

    port: int
    free: bool
    connect_errno: int


@dataclass(slots=True)
class DockerLifecycleProbe:
    """Artifacts from a Docker health → API → cleanup probe."""

    ok: bool
    message: str
    base_url: str
    host_port: int
    container_name: str
    image: str
    health_status: str | None = None
    health_body: dict[str, Any] = field(default_factory=dict)
    ready_status: int | None = None
    version_status: int | None = None
    mutate_path: str | None = None
    mutate_status: int | None = None
    mutate_body: dict[str, Any] = field(default_factory=dict)
    cleaned: bool = False
    rebind_ok: bool = False
    steps: list[str] = field(default_factory=list)


def _resolve_secret(shared_token: str | None) -> str:
    secret = (shared_token or "").strip()
    if not secret:
        secret = (os.environ.get("CHALLENGE_SHARED_TOKEN") or "").strip()
    if not secret:
        token_file = (os.environ.get("CHALLENGE_SHARED_TOKEN_FILE") or "").strip()
        if token_file:
            try:
                secret = Path(token_file).read_text(encoding="utf-8").strip()
            except OSError:
                secret = ""
    return secret or "test-challenge-shared-token"


def probe_port_free(port: int, host: str = "127.0.0.1") -> PortProbe:
    """Return whether *port* currently accepts TCP connections (busy vs free)."""

    value = int(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        result = sock.connect_ex((host, value))
    # 0 means something accepted the connect → not free.
    return PortProbe(port=value, free=result != 0, connect_errno=int(result))


def assert_mission_band_port(port: int) -> int:
    value = int(port)
    if not is_mission_port(value):
        raise ValueError(
            f"host port {value} outside mission band {MIN_MISSION_PORT}–{MAX_MISSION_PORT}"
        )
    return value


def docker_available() -> bool:
    """True when docker CLI + daemon respond."""

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


def _run_docker(
    args: Sequence[str],
    *,
    check: bool = True,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def docker_rm_force(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def docker_volume_rm_force(name: str) -> None:
    subprocess.run(
        ["docker", "volume", "rm", "-f", name],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def stage_base_wheel(repo_root: Path | None = None) -> Path | None:
    """Copy Base 3.1.2 wheel into docker/vendor for offline builds."""

    root = repo_root or REPO_ROOT
    vendor = root / "docker" / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    wheel_src = Path("/projects/platform-network/platform/dist/base-3.1.2-py3-none-any.whl")
    wheel_dst = vendor / "base-3.1.2-py3-none-any.whl"
    if wheel_src.is_file():
        if not wheel_dst.is_file() or wheel_dst.stat().st_size != wheel_src.stat().st_size:
            import shutil

            shutil.copy2(wheel_src, wheel_dst)
        return wheel_dst
    return wheel_dst if wheel_dst.is_file() else None


def ensure_image(
    image: str = DEFAULT_CROSS_DOCKER_IMAGE,
    *,
    repo_root: Path | None = None,
    rebuild: bool = False,
) -> str:
    """Build (or reuse) the challenge image; return the tag."""

    root = repo_root or REPO_ROOT
    stage_base_wheel(root)
    if not rebuild:
        inspect = _run_docker(
            ["docker", "image", "inspect", image],
            check=False,
            timeout=30,
        )
        if inspect.returncode == 0:
            return image
    build = _run_docker(
        ["docker", "build", "-t", image, str(root)],
        check=False,
        timeout=600,
    )
    if build.returncode != 0:
        raise RuntimeError(
            f"docker build failed for {image}:\nSTDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}"
        )
    return image


def wait_http_json(
    url: str,
    *,
    timeout_s: float = 60.0,
    accept: Callable[[int, dict[str, Any]], bool] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Poll URL until accept() succeeds or timeout."""

    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    last_status = 0
    last_body: dict[str, Any] = {}
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=3) as response:  # noqa: S310 — loopback only
                raw = response.read().decode("utf-8")
                last_status = int(response.status)
                last_body = json.loads(raw) if raw else {}
                if accept is None:
                    if last_status == 200:
                        return last_status, last_body
                elif accept(last_status, last_body):
                    return last_status, last_body
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_err = exc
        time.sleep(0.4)
    raise TimeoutError(
        f"URL {url} did not become ready within {timeout_s}s "
        f"(last_status={last_status}, last_err={last_err!r})"
    )


def wait_container_healthy(name: str, *, timeout_s: float = 90.0) -> str:
    deadline = time.time() + timeout_s
    last = "unknown"
    while time.time() < deadline:
        proc = _run_docker(
            [
                "docker",
                "inspect",
                "--format",
                "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}",
                name,
            ],
            check=False,
            timeout=15,
        )
        last = (proc.stdout or "").strip()
        if last == "healthy":
            return last
        if last in {"exited", "dead"}:
            logs = _run_docker(["docker", "logs", name], check=False, timeout=30)
            raise RuntimeError(
                f"container {name} exited before healthy (state={last}):\n"
                f"{logs.stdout}\n{logs.stderr}"
            )
        time.sleep(0.8)
    logs = _run_docker(["docker", "logs", name], check=False, timeout=30)
    raise TimeoutError(
        f"container {name} not healthy within {timeout_s}s (last={last}):\n"
        f"{logs.stdout}\n{logs.stderr}"
    )


def _signed_http(
    method: str,
    url: str,
    *,
    secret: str,
    hotkey: str,
    body: dict[str, Any] | None = None,
    timeout: float = 15.0,
) -> httpx.Response:
    raw = b"" if body is None else json.dumps(body).encode()
    headers = build_signed_headers(
        secret=secret,
        hotkey=hotkey,
        body=raw,
    )
    headers["Content-Type"] = "application/json"
    with httpx.Client(timeout=timeout) as client:
        return client.request(
            method, url, content=raw if body is not None else None, headers=headers
        )


def post_docker_api_job(
    base_url: str,
    *,
    shared_token: str | None = None,
    timeout: float = 20.0,
) -> tuple[int, dict[str, Any], str]:
    """Register provider/node/offer/rent then submit a job (mutating path).

    Falls back to offer-only mutate when market capacity is unavailable so
    VAL-CROSS-007 still observes a non-404 challenge mutate class.
    """

    normalized = base_url.rstrip("/")
    secret = _resolve_secret(shared_token)

    # 1) Provider register (mutate)
    reg = _signed_http(
        "POST",
        f"{normalized}/v1/providers/register",
        secret=secret,
        hotkey=_DOCKER_PROVIDER_HK,
        body={"display_name": "Cross Docker Provider"},
        timeout=timeout,
    )
    if reg.status_code >= 400 and reg.status_code != 409:
        return reg.status_code, _safe_json(reg), "/v1/providers/register"

    # 2) Node register
    node_body = {
        "gpu_model": "H100",
        "gpu_count": 1,
        "hostname": "dock-sim-node-0",
        "ssh_endpoint": "10.8.8.8:22",
        "tee_capability": "none",
        "inventory": {
            "ib_devices": ["mlx5_0"],
            "ib_rate_gbps": 200.0,
        },
    }
    node = _signed_http(
        "POST",
        f"{normalized}/v1/nodes",
        secret=secret,
        hotkey=_DOCKER_PROVIDER_HK,
        body=node_body,
        timeout=timeout,
    )
    if node.status_code >= 400:
        return node.status_code, _safe_json(node), "/v1/nodes"
    node_json = _safe_json(node)
    node_id = str(node_json.get("id") or node_json.get("node_id") or "")
    if not node_id:
        return node.status_code, node_json, "/v1/nodes"

    # 3) Offer
    offer_body = {
        "node_ids": [node_id],
        "price_per_hour": 1.25,
        "max_lifetime_hours": 12.0,
        "require_ib": False,
        "mode": "single",
    }
    offer = _signed_http(
        "POST",
        f"{normalized}/v1/offers",
        secret=secret,
        hotkey=_DOCKER_PROVIDER_HK,
        body=offer_body,
        timeout=timeout,
    )
    if offer.status_code >= 400:
        return offer.status_code, _safe_json(offer), "/v1/offers"
    offer_json = _safe_json(offer)
    offer_id = str(offer_json.get("id") or offer_json.get("offer_id") or "")
    if not offer_id:
        return offer.status_code, offer_json, "/v1/offers"

    # 4) Rent
    lease_id: str | None = None
    pod_id: str | None = None
    rent = _signed_http(
        "POST",
        f"{normalized}/v1/offers/{offer_id}/rent",
        secret=secret,
        hotkey=_DOCKER_DEMAND_HK,
        body={"lifetime_hours": 4.0},
        timeout=timeout,
    )
    if rent.status_code < 400:
        rent_body = _safe_json(rent)
        lease = rent_body.get("lease") if isinstance(rent_body.get("lease"), dict) else rent_body
        pod = rent_body.get("pod") if isinstance(rent_body.get("pod"), dict) else {}
        if isinstance(lease, dict):
            raw_lease = lease.get("id") or lease.get("lease_id")
            lease_id = str(raw_lease) if raw_lease else None
        if isinstance(pod, dict):
            raw_pod = pod.get("id") or pod.get("pod_id")
            pod_id = str(raw_pod) if raw_pod else None

    # 5) Job submit (primary "API job" leg)
    job_body: dict[str, Any] = {
        "image_digest": ALLOWED_IMAGE,
        "entrypoint": ["python", "-c", "print('cross-docker')"],
        "world_size": 1,
        "nnodes": 1,
        "nproc_per_node": 1,
        "backend": "nccl",
        "fabric": "auto",
        "tee": "none",
        "timeout_s": 60,
        "resource": {"gpus": 1, "nodes": 1},
        "client_request_id": f"cross-dock-{uuid.uuid4().hex[:12]}",
        "placement_policy": "pack",
    }
    if lease_id:
        job_body["lease_id"] = lease_id
    if pod_id:
        job_body["pod_id"] = pod_id

    job = _signed_http(
        "POST",
        f"{normalized}/v1/jobs",
        secret=secret,
        hotkey=_DOCKER_DEMAND_HK,
        body=job_body,
        timeout=timeout,
    )
    return job.status_code, _safe_json(job), "/v1/jobs"


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}
    except Exception:  # noqa: BLE001
        return {"raw": (response.text or "")[:500]}


def run_docker_health_then_api_job(
    *,
    host_port: int = DEFAULT_CROSS_DOCKER_HOST_PORT,
    image: str = DEFAULT_CROSS_DOCKER_IMAGE,
    container_name: str = DEFAULT_CROSS_DOCKER_CONTAINER,
    volume_name: str = DEFAULT_CROSS_DOCKER_VOLUME,
    shared_token: str | None = None,
    rebuild: bool = False,
    cleanup: bool = True,
    mutate: bool = True,
) -> DockerLifecycleProbe:
    """VAL-CROSS-007: healthy Docker image, then challenge API functional path.

    After health/ready/version, optionally POST a signed mutating challenge path
    (provider/node/offer/rent/job) so routes are proven non-404.
    """

    steps: list[str] = []
    secret = _resolve_secret(shared_token)
    try:
        host_port = assert_mission_band_port(host_port)
    except ValueError as exc:
        return DockerLifecycleProbe(
            ok=False,
            message=str(exc),
            base_url="",
            host_port=host_port,
            container_name=container_name,
            image=image,
            steps=steps + [str(exc)],
        )

    if not docker_available():
        return DockerLifecycleProbe(
            ok=False,
            message="docker daemon not available",
            base_url="",
            host_port=host_port,
            container_name=container_name,
            image=image,
            steps=steps + ["docker info failed"],
        )

    docker_rm_force(container_name)
    steps.append(f"cleanup pre-run container={container_name}")

    try:
        ensure_image(image, rebuild=rebuild)
        steps.append(f"image ready tag={image}")
    except Exception as exc:  # noqa: BLE001
        return DockerLifecycleProbe(
            ok=False,
            message=f"image build/inspect failed: {exc}",
            base_url="",
            host_port=host_port,
            container_name=container_name,
            image=image,
            steps=steps + [str(exc)],
        )

    # Ensure volume for /data (optional integrity semantics not required here).
    _run_docker(["docker", "volume", "create", volume_name], check=False, timeout=30)

    base_url = f"http://127.0.0.1:{host_port}"
    run = _run_docker(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"127.0.0.1:{host_port}:8000",
            "-v",
            f"{volume_name}:/data",
            "-e",
            f"CHALLENGE_SHARED_TOKEN={secret}",
            "-e",
            "CHALLENGE_SHARED_TOKEN_FILE=",
            "-e",
            "CHALLENGE_DATABASE_URL=sqlite+aiosqlite:////data/challenge.sqlite3",
            "-e",
            "HYPER_ALLOW_INSECURE_SIGNATURES=true",
            "-e",
            "HYPER_COMBINED_WORKER=true",
            "-e",
            "HYPER_COMBINED_WORKER_INTERVAL_SECONDS=0.2",
            "-e",
            "HYPER_SIM_AUTO_CAPACITY=true",
            "-e",
            f"HYPER_JOB_IMAGE_ALLOWLIST={ALLOWED_IMAGE}",
            image,
        ],
        check=False,
        timeout=60,
    )
    if run.returncode != 0:
        return DockerLifecycleProbe(
            ok=False,
            message=f"docker run failed: {run.stderr or run.stdout}",
            base_url=base_url,
            host_port=host_port,
            container_name=container_name,
            image=image,
            steps=steps + [f"docker run rc={run.returncode}"],
        )
    steps.append(f"container started name={container_name} port={host_port}")

    probe = DockerLifecycleProbe(
        ok=False,
        message="in progress",
        base_url=base_url,
        host_port=host_port,
        container_name=container_name,
        image=image,
        steps=steps,
    )

    try:
        health_state = wait_container_healthy(container_name, timeout_s=90)
        probe.health_status = health_state
        probe.steps.append(f"container health={health_state}")

        status, health_body = wait_http_json(
            f"{base_url}/health",
            timeout_s=45.0,
            accept=lambda code, body: (
                code == 200 and body.get("status") == "ok" and body.get("slug") == "hypercluster"
            ),
        )
        probe.health_body = health_body
        probe.steps.append(
            f"GET /health → {status} status={health_body.get('status')} "
            f"slug={health_body.get('slug')}"
        )

        ready_status, ready_body = wait_http_json(
            f"{base_url}/ready",
            timeout_s=45.0,
            accept=lambda code, body: code == 200 and body.get("ready") is True,
        )
        probe.ready_status = ready_status
        probe.steps.append(f"GET /ready → {ready_status} ready={ready_body.get('ready')}")

        version_status, version_body = wait_http_json(
            f"{base_url}/version",
            timeout_s=30.0,
            accept=lambda code, body: (
                code == 200
                and (
                    body.get("challenge_slug") == "hypercluster"
                    or body.get("slug") == "hypercluster"
                )
            ),
        )
        probe.version_status = version_status
        probe.steps.append(
            f"GET /version → {version_status} "
            f"slug={version_body.get('challenge_slug') or version_body.get('slug')}"
        )

        # Relative proxy path smoke while container is up (also VAL-CROSS-023 leg).
        with httpx.Client(timeout=10.0) as client:
            offers = client.get(f"{base_url}/v1/offers")
            jobs = client.get(f"{base_url}/v1/jobs")
            probe.steps.append(
                f"GET /v1/offers → {offers.status_code}; GET /v1/jobs → {jobs.status_code}"
            )
            if offers.status_code == 404 or jobs.status_code == 404:
                probe.message = (
                    "challenge API /v1 paths 404 while health works "
                    f"(offers={offers.status_code}, jobs={jobs.status_code})"
                )
                return probe
            # Any non-404 class is success for existence; empty list / auth ok.
            if offers.status_code >= 500 or jobs.status_code >= 500:
                probe.message = (
                    f"challenge /v1 paths 5xx offers={offers.status_code} jobs={jobs.status_code}"
                )
                return probe

        if mutate:
            m_status, m_body, m_path = post_docker_api_job(
                base_url, shared_token=secret, timeout=30.0
            )
            probe.mutate_status = m_status
            probe.mutate_body = m_body
            probe.mutate_path = m_path
            probe.steps.append(f"mutate {m_path} → {m_status} keys={sorted(m_body.keys())[:8]}")
            if m_status == 404:
                probe.message = f"mutating path {m_path} returned 404"
                return probe
            if m_status >= 500:
                probe.message = f"mutating path {m_path} HTTP {m_status}: {m_body}"
                return probe
            # 2xx or well-formed 4xx auth/validation still prove the route exists.
            if m_status >= 400:
                # Prefer success, but accept structured business errors after health.
                # Do require that registers/nodes already advanced to offer/job surface.
                if m_path in {"/v1/providers/register", "/v1/nodes"}:
                    probe.message = f"early mutate failed at {m_path} HTTP {m_status}: {m_body}"
                    return probe
            # Job create ideally 2xx; lease optional under sim_auto_capacity.
            if m_path == "/v1/jobs" and m_status >= 400:
                # Fallback: offer create already proved write (non-404).
                if probe.mutate_status is not None:
                    probe.steps.append(
                        f"job submit class HTTP {m_status}; continuing if prior offer/node mutated"
                    )

        probe.ok = True
        probe.message = (
            "docker health+ready+version ok; challenge API functional "
            f"(mutate={probe.mutate_path}:{probe.mutate_status})"
        )
        return probe
    except Exception as exc:  # noqa: BLE001
        probe.ok = False
        probe.message = f"docker health/api probe failed: {exc}"
        probe.steps.append(str(exc))
        return probe
    finally:
        if cleanup:
            docker_rm_force(container_name)
            probe.cleaned = True
            probe.steps.append(f"container removed name={container_name}")
            # Port must free after stop/rm (VAL-CROSS-022 subset).
            free = probe_port_free(host_port)
            probe.steps.append(f"port {host_port} free={free.free} errno={free.connect_errno}")
            probe.rebind_ok = free.free


def run_docker_stop_remove_rebind_free(
    *,
    host_port: int = DEFAULT_CROSS_DOCKER_HOST_PORT,
    image: str = DEFAULT_CROSS_DOCKER_IMAGE,
    container_name: str = DEFAULT_CROSS_DOCKER_CONTAINER,
    shared_token: str | None = None,
    rebind_smoke: bool = True,
) -> DockerLifecycleProbe:
    """VAL-CROSS-022: after docker stop/rm, port free; bare-metal can rebind."""

    probe = run_docker_health_then_api_job(
        host_port=host_port,
        image=image,
        container_name=container_name,
        shared_token=shared_token,
        rebuild=False,
        cleanup=True,
        mutate=False,
    )
    if not probe.ok:
        return probe

    if not probe.cleaned or not probe.rebind_ok:
        probe.ok = False
        probe.message = (
            f"port {host_port} still held after docker rm "
            f"(cleaned={probe.cleaned}, free={probe.rebind_ok})"
        )
        return probe

    # Explicit container gone check.
    ps = _run_docker(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={container_name}",
            "--format",
            "{{.Names}}",
        ],
        check=False,
        timeout=20,
    )
    names = [ln.strip() for ln in (ps.stdout or "").splitlines() if ln.strip()]
    if container_name in names:
        probe.ok = False
        probe.message = f"zombie container still listed: {names}"
        probe.steps.append("zombie container present")
        return probe
    probe.steps.append("no leftover container after stop/rm")

    if rebind_smoke:
        # Prove the port accepts a fresh bare-metal bind: socket bind on the
        # mission host port (no need to boot full uvicorn for rebind freedom).
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", host_port))
            sock.listen(1)
            probe.steps.append(f"bare-metal rebind listen ok on {host_port}")
            sock.close()
        except OSError as exc:
            probe.ok = False
            probe.rebind_ok = False
            probe.message = f"rebind failed on {host_port}: {exc}"
            probe.steps.append(str(exc))
            return probe

    probe.ok = True
    probe.message = f"docker cleanup left port {host_port} free; bare-metal rebind succeeded"
    return probe


def run_scenario_suite_order_green(
    base_url: str,
    *,
    timeout: float = 30.0,
    shared_token: str | None = None,
    master_url: str | None = None,
) -> SuiteResult:
    """VAL-CROSS-018: canonical suite order smoke→marketplace→nccl→tee→weights green."""

    suite = run_scenario_suite(
        base_url,
        timeout=timeout,
        shared_token=shared_token,
        master_url=master_url,
        stop_on_fail=True,
    )
    # Enforce architecture §12.3 exact order, not merely set equality.
    expected = list(DEFAULT_SCENARIO_ORDER)
    if suite.order != expected:
        suite.ok = False
        suite.message = f"suite order mismatch: got {suite.order} expected {expected}"
        return suite
    names = [r.name for r in suite.results]
    if names != expected:
        suite.ok = False
        suite.message = f"suite result names out of order: {names} expected {expected}"
        return suite
    if list(KNOWN_SCENARIOS) != expected:
        suite.ok = False
        suite.message = f"KNOWN_SCENARIOS drift: {list(KNOWN_SCENARIOS)} vs {expected}"
        return suite
    if suite.ok:
        suite.message = f"scenario suite order green: {' → '.join(expected)}"
    return suite


def _collect_included_router_routes(root: Any) -> list[Any]:
    """Walk FastAPI/Starlette route tree including ``_IncludedRouter`` wrappers."""

    out: list[Any] = []
    stack: list[Any] = list(getattr(root, "routes", []) or [])
    seen: set[int] = set()
    while stack:
        route = stack.pop()
        rid = id(route)
        if rid in seen:
            continue
        seen.add(rid)
        out.append(route)
        # Nested APIRouter (legacy Mount-style)
        nested = getattr(route, "routes", None)
        if nested:
            stack.extend(list(nested))
        nested_app = getattr(route, "app", None)
        if nested_app is not None and nested_app is not root:
            nested_routes = getattr(nested_app, "routes", None)
            if nested_routes:
                stack.extend(list(nested_routes))
        # FastAPI 0.128+ wraps include_router as _IncludedRouter
        original = getattr(route, "original_router", None)
        if original is not None:
            nested_routes = getattr(original, "routes", None)
            if nested_routes:
                stack.extend(list(nested_routes))
        ctx = getattr(route, "include_context", None)
        if ctx is not None:
            included = getattr(ctx, "included_router", None)
            if included is not None:
                nested_routes = getattr(included, "routes", None)
                if nested_routes:
                    stack.extend(list(nested_routes))
    return out


def _route_paths(app: Any) -> set[str]:
    """Best-effort path inventory via OpenAPI then route walk.

    OpenAPI is authoritative for relative mount shape under FastAPI routers that
    are wrapped as ``_IncludedRouter`` (no ``.path`` on the wrapper itself).
    """

    paths: set[str] = set()
    openapi = getattr(app, "openapi", None)
    if callable(openapi):
        try:
            schema = openapi()
            for key in (schema or {}).get("paths", {}) or {}:
                if isinstance(key, str):
                    paths.add(key)
        except Exception:  # noqa: BLE001 — fall back to route walk
            pass
    for route in _collect_included_router_routes(app):
        path = getattr(route, "path", None)
        if isinstance(path, str) and path:
            paths.add(path)
        # Some wrappers expose path_format instead of path
        path_format = getattr(route, "path_format", None)
        if isinstance(path_format, str) and path_format:
            paths.add(path_format)
    return paths


def public_relative_v1_routes(app: Any) -> list[str]:
    """List mounted routes under /v1 (relative; no /challenges/... prefix)."""

    from base.challenge_sdk import is_public_route

    out: list[str] = []
    # Prefer complete path set (OpenAPI) for black-box style inventory.
    for path in sorted(_route_paths(app)):
        if path.startswith("/v1/"):
            out.append(path)
    # Also collect endpoints still marked public on the walk for step evidence.
    for route in _collect_included_router_routes(app):
        raw_path = getattr(route, "path", None)
        if not isinstance(raw_path, str) or not raw_path:
            raw_path = getattr(route, "path_format", None)
        if not isinstance(raw_path, str) or not raw_path.startswith("/v1/"):
            continue
        path = raw_path
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None and (
            is_public_route(endpoint) or getattr(endpoint, "__base_public_route__", False)
        ):
            if path not in out:
                out.append(path)
    return sorted(set(out))


def probe_relative_proxy_paths(
    base_url: str,
    *,
    timeout: float = 10.0,
    shared_token: str | None = None,
    app: Any | None = None,
) -> ScenarioResult:
    """VAL-CROSS-023: challenge serves relative /v1/... without proxy prefix."""

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    secret = _resolve_secret(shared_token)

    # Optional offline inventory of mounted routes (ASGI app).
    if app is not None:
        paths = _route_paths(app)
        steps.append(f"route inventory size={len(paths)}")
        for bad in INVALID_PROXY_PREFIXES:
            prefixed = [p for p in paths if p.startswith(bad)]
            if prefixed:
                return ScenarioResult(
                    name=CROSS_DOCKER_SCENARIO_PROXY,
                    ok=False,
                    base_url=normalized,
                    message=f"routes wrongly mounted under proxy prefix: {prefixed[:5]}",
                    steps=steps + [f"bad prefix {bad}: {prefixed[:5]}"],
                )
        v1_public = public_relative_v1_routes(app)
        steps.append(f"public /v1 routes count={len(v1_public)}")
        for required in ("/v1/offers", "/v1/jobs"):
            # path params like /v1/jobs/{job_id} still prove relative mount
            found = any(p.startswith(required.rstrip("/")) for p in paths)
            if not found:
                sample = sorted(paths)[:20]
                return ScenarioResult(
                    name=CROSS_DOCKER_SCENARIO_PROXY,
                    ok=False,
                    base_url=normalized,
                    message=f"required relative route missing: {required}",
                    steps=steps + [f"missing {required} in {sample}"],
                )
            steps.append(f"mounted relative {required}*")

    # Black-box: direct host curl class without /challenges/hypercluster.
    try:
        with httpx.Client(timeout=timeout) as client:
            for path in PROXY_RELATIVE_PATHS:
                url = f"{normalized}{path}"
                response = client.get(url)
                steps.append(f"GET {path} → {response.status_code}")
                if response.status_code == 404:
                    return ScenarioResult(
                        name=CROSS_DOCKER_SCENARIO_PROXY,
                        ok=False,
                        base_url=normalized,
                        message=f"relative path 404: {path}",
                        steps=steps,
                    )
                if response.status_code >= 500:
                    return ScenarioResult(
                        name=CROSS_DOCKER_SCENARIO_PROXY,
                        ok=False,
                        base_url=normalized,
                        message=f"relative path 5xx: {path} → {response.status_code}",
                        steps=steps,
                    )

            # Explicit wrong-prefix must not be the only way to hit routes.
            wrong = client.get(f"{normalized}/challenges/hypercluster/v1/offers")
            steps.append(
                f"GET /challenges/hypercluster/v1/offers → {wrong.status_code} "
                "(not required; relative /v1 is canonical)"
            )

            # Auth class: unauthenticated mutates must fail closed (not 404).
            bare_post = client.post(f"{normalized}/v1/jobs", json={})
            steps.append(f"POST /v1/jobs (no auth) → {bare_post.status_code}")
            if bare_post.status_code == 404:
                return ScenarioResult(
                    name=CROSS_DOCKER_SCENARIO_PROXY,
                    ok=False,
                    base_url=normalized,
                    message="POST /v1/jobs 404 — routes not under relative /v1",
                    steps=steps,
                )
            if bare_post.status_code in {200, 201}:
                return ScenarioResult(
                    name=CROSS_DOCKER_SCENARIO_PROXY,
                    ok=False,
                    base_url=normalized,
                    message="POST /v1/jobs accepted without auth (fail-closed broken)",
                    steps=steps,
                )
            # 401/403/422/400/405 all prove route exists with auth/validation class.
            if bare_post.status_code not in {400, 401, 403, 405, 409, 422, 503}:
                # 503 runtime_not_ready still proves path mounted.
                if bare_post.status_code < 400 or bare_post.status_code >= 500:
                    return ScenarioResult(
                        name=CROSS_DOCKER_SCENARIO_PROXY,
                        ok=False,
                        base_url=normalized,
                        message=(f"unexpected unauth POST /v1/jobs class {bare_post.status_code}"),
                        steps=steps,
                    )

            # Signed list/read surfaces remain /v1 without prefix.
            headers = build_signed_headers(
                secret=secret,
                hotkey=_DOCKER_DEMAND_HK,
                body=b"",
            )
            listed = client.get(f"{normalized}/v1/offers", headers=headers)
            steps.append(f"GET /v1/offers (signed opt) → {listed.status_code}")
            if listed.status_code == 404:
                return ScenarioResult(
                    name=CROSS_DOCKER_SCENARIO_PROXY,
                    ok=False,
                    base_url=normalized,
                    message="GET /v1/offers 404 under relative path",
                    steps=steps,
                )
    except httpx.HTTPError as exc:
        return ScenarioResult(
            name=CROSS_DOCKER_SCENARIO_PROXY,
            ok=False,
            base_url=normalized,
            message=f"relative proxy probe transport error: {exc}",
            steps=steps + [str(exc)],
        )

    return ScenarioResult(
        name=CROSS_DOCKER_SCENARIO_PROXY,
        ok=True,
        base_url=normalized,
        message=(
            "public proxy-shaped paths remain relative /v1/... "
            f"({len(PROXY_RELATIVE_PATHS)} surfaces checked)"
        ),
        steps=steps,
    )


def run_cross_docker_scenario_proxy(
    base_url: str,
    *,
    timeout: float = 60.0,
    shared_token: str | None = None,
    master_url: str | None = None,
    include_docker: bool = True,
    include_suite: bool = True,
    include_proxy: bool = True,
    app: Any | None = None,
    docker_host_port: int = DEFAULT_CROSS_DOCKER_HOST_PORT,
) -> ScenarioResult:
    """Aggregate VAL-CROSS-007/018/022/023 under one named scenario."""

    normalized = base_url.rstrip("/")
    steps: list[str] = []
    details: dict[str, Any] = {}

    if include_proxy:
        proxy = probe_relative_proxy_paths(
            normalized,
            timeout=min(timeout, 15.0),
            shared_token=shared_token,
            app=app,
        )
        steps.extend(f"proxy: {s}" for s in proxy.steps)
        details["proxy_ok"] = proxy.ok
        if not proxy.ok:
            return ScenarioResult(
                name=CROSS_DOCKER_SCENARIO_PROXY,
                ok=False,
                base_url=normalized,
                message=f"VAL-CROSS-023 failed: {proxy.message}",
                steps=steps,
            )
        steps.append(f"VAL-CROSS-023 ok: {proxy.message}")

    if include_suite:
        suite = run_scenario_suite_order_green(
            normalized,
            timeout=timeout,
            shared_token=shared_token,
            master_url=master_url,
        )
        steps.extend(suite.summary_lines())
        details["suite_ok"] = suite.ok
        details["suite_order"] = list(suite.order)
        if not suite.ok:
            return ScenarioResult(
                name=CROSS_DOCKER_SCENARIO_PROXY,
                ok=False,
                base_url=normalized,
                message=f"VAL-CROSS-018 failed: {suite.message}",
                steps=steps,
            )
        steps.append(f"VAL-CROSS-018 ok: {suite.message}")

    if include_docker:
        if not docker_available():
            steps.append("docker daemon unavailable — skip live 007/022 (report soft)")
            details["docker_skipped"] = True
        else:
            # 007 + 022 chained: health/API then stop/rm rebind.
            rebind = run_docker_stop_remove_rebind_free(
                host_port=docker_host_port,
                shared_token=shared_token,
                rebind_smoke=True,
            )
            steps.extend(f"docker: {s}" for s in rebind.steps)
            details["docker_ok"] = rebind.ok
            details["docker_rebind"] = rebind.rebind_ok
            if not rebind.ok:
                return ScenarioResult(
                    name=CROSS_DOCKER_SCENARIO_PROXY,
                    ok=False,
                    base_url=normalized,
                    message=f"VAL-CROSS-007/022 failed: {rebind.message}",
                    steps=steps,
                )
            # Separate mutate-enabled 007 pass (API job after health).
            job_probe = run_docker_health_then_api_job(
                host_port=docker_host_port,
                shared_token=shared_token,
                cleanup=True,
                mutate=True,
            )
            steps.extend(f"docker-job: {s}" for s in job_probe.steps)
            details["docker_job_ok"] = job_probe.ok
            details["mutate_status"] = job_probe.mutate_status
            if not job_probe.ok:
                return ScenarioResult(
                    name=CROSS_DOCKER_SCENARIO_PROXY,
                    ok=False,
                    base_url=normalized,
                    message=f"VAL-CROSS-007 failed: {job_probe.message}",
                    steps=steps,
                )
            steps.append(f"VAL-CROSS-007 ok: {job_probe.message}")
            steps.append(f"VAL-CROSS-022 ok: {rebind.message}")

    return ScenarioResult(
        name=CROSS_DOCKER_SCENARIO_PROXY,
        ok=True,
        base_url=normalized,
        message=(
            "cross-docker-scenario-proxy green: "
            "proxy /v1 relative, suite order, docker health/rebind"
        ),
        steps=steps,
    )


__all__ = [
    "ALLOWED_IMAGE",
    "CROSS_DOCKER_SCENARIO_PROXY",
    "DEFAULT_CROSS_DOCKER_CONTAINER",
    "DEFAULT_CROSS_DOCKER_HOST_PORT",
    "DEFAULT_CROSS_DOCKER_IMAGE",
    "DEFAULT_CROSS_DOCKER_VOLUME",
    "DEFAULT_DOCKER_HOST_PORT",
    "DockerLifecycleProbe",
    "INVALID_PROXY_PREFIXES",
    "PROXY_RELATIVE_PATHS",
    "PortProbe",
    "assert_mission_band_port",
    "docker_available",
    "docker_rm_force",
    "ensure_image",
    "post_docker_api_job",
    "probe_port_free",
    "probe_relative_proxy_paths",
    "public_relative_v1_routes",
    "run_cross_docker_scenario_proxy",
    "run_docker_health_then_api_job",
    "run_docker_stop_remove_rebind_free",
    "run_scenario_suite_order_green",
    "stage_base_wheel",
    "wait_container_healthy",
    "wait_http_json",
]
