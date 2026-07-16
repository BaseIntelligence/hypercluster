#!/usr/bin/env python3
"""M9 host GPU probe ops runner (external QA only; no Verda SDK in product).

Protocol:
  1. WaitSSH (TCP + key-auth) with backoff
  2. Run product GpuProbeService over RealSshExecutor allowlist commands
  3. Emit host_probe.json with status / measured class / digests (no PEM)

This module intentionally:
  - lives under scripts/qa (never imported into product cloud-rental paths from CI gates)
  - reuses product probe parsers/pipeline/allowlist (no commercial cloud SDK)
  - never calls set_weights or commercial cloud rent APIs
Usage::

    uv run python scripts/qa/host_gpu_probe.py \\
      --host 1.2.3.4 --port 22 \\
      --key-file /root/.ssh/tetta.pem \\
      --claimed-model \"1A100.40S.22V\" \\
      --out /tmp/host_probe.json
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from hypercluster.probe.keys import (  # noqa: E402
    KeyRef,
    resolve_key_ref,
)
from hypercluster.probe.model_table import normalize_gpu_model  # noqa: E402
from hypercluster.probe.pipeline import (  # noqa: E402
    GpuProbeConfig,
    GpuProbeContext,
    run_gpu_probe,
)
from hypercluster.probe.ssh_exec import (  # noqa: E402
    RealSshExecutor,
    RealSshTarget,
    parse_ssh_endpoint,
)
from hypercluster.probe.types import ClaimedInventory  # noqa: E402


def _j(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_j, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def wait_tcp(
    host: str,
    port: int,
    *,
    timeout_s: float = 180.0,
    interval_s: float = 2.0,
) -> dict[str, Any]:
    """Block until TCP accept or timeout (WaitSSH step 1)."""

    deadline = time.time() + timeout_s
    attempts = 0
    last_err: str | None = None
    while time.time() < deadline:
        attempts += 1
        try:
            with socket.create_connection((host, port), timeout=min(5.0, interval_s + 1.0)):
                return {
                    "ok": True,
                    "attempts": attempts,
                    "host": host,
                    "port": port,
                    "elapsed_s": timeout_s - max(0.0, deadline - time.time()),
                }
        except OSError as exc:
            last_err = str(exc)
            time.sleep(interval_s)
    return {
        "ok": False,
        "attempts": attempts,
        "host": host,
        "port": port,
        "error": last_err or "tcp_timeout",
    }


def wait_ssh_auth(
    host: str,
    port: int,
    *,
    key_path: str,
    username: str = "root",
    timeout_s: float = 300.0,
    interval_s: float = 3.0,
) -> dict[str, Any]:
    """TCP then key-auth via RealSshExecutor.connect (WaitSSH)."""

    tcp = wait_tcp(host, port, timeout_s=min(timeout_s, 180.0), interval_s=interval_s)
    if not tcp.get("ok"):
        return {"ok": False, "phase": "tcp", **tcp}

    deadline = time.time() + timeout_s
    attempts = 0
    last_msg = ""
    key_path_obj = Path(key_path).expanduser()
    if not key_path_obj.is_file():
        return {"ok": False, "phase": "key", "error": f"key file missing: {key_path}"}

    ref = KeyRef(kind="file", name=str(key_path_obj))
    resolved = resolve_key_ref(ref)
    target = RealSshTarget(
        host=host,
        port=port,
        username=username,
        key_path=str(key_path_obj),
        key_fingerprint=resolved.fingerprint,
        key_ref=ref,
    )
    while time.time() < deadline:
        attempts += 1
        executor = RealSshExecutor(target=target, connect_timeout_s=min(20.0, interval_s + 10.0))
        try:
            res = executor.connect()
            if res.exit_code == 0 and not res.timed_out:
                return {
                    "ok": True,
                    "phase": "auth",
                    "attempts": attempts,
                    "host": host,
                    "port": port,
                    "key_fingerprint": resolved.fingerprint,
                    "username": username,
                    "tcp": tcp,
                }
            last_msg = (res.stderr or res.stdout or f"exit={res.exit_code}")[:300]
        except Exception as exc:  # noqa: BLE001
            last_msg = str(exc)[:300]
        finally:
            executor.close()
        time.sleep(interval_s)
    return {
        "ok": False,
        "phase": "auth",
        "attempts": attempts,
        "host": host,
        "port": port,
        "error": last_msg or "ssh_auth_timeout",
        "tcp": tcp,
    }


def run_host_gpu_probe(
    *,
    host: str,
    port: int = 22,
    key_path: str,
    username: str = "root",
    claimed_model: str,
    claimed_gpu_count: int = 1,
    node_id: str | None = None,
    provider_hotkey: str | None = None,
    mode: str = "full",
    require_docker: bool = False,
    skip_wait: bool = False,
    wait_timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Run full (or quick) GPU probe and return host_probe.json-ready dict.

    Returns a JSON-serializable pack including ``status`` and model class fields
    for VAL-GPU-060 / VAL-GPU-062.
    """

    started = time.time()
    key_path_obj = Path(key_path).expanduser()
    if not key_path_obj.is_file():
        return {
            "status": "error",
            "ok": False,
            "failure_code": "key_not_found",
            "error": f"SSH key file not found: {key_path}",
        }

    ref = KeyRef(kind="file", name=str(key_path_obj))
    resolved = resolve_key_ref(ref)
    wait_info: dict[str, Any] | None = None
    if not skip_wait:
        wait_info = wait_ssh_auth(
            host,
            port,
            key_path=str(key_path_obj),
            username=username,
            timeout_s=wait_timeout_s,
        )
        if not wait_info.get("ok"):
            return {
                "status": "error",
                "ok": False,
                "failure_code": "ssh_connect",
                "wait_ssh": wait_info,
                "claimed": {
                    "gpu_model": claimed_model,
                    "gpu_count": claimed_gpu_count,
                    "family": normalize_gpu_model(claimed_model),
                },
                "key_fingerprint": resolved.fingerprint,
                "ssh_endpoint": f"{host}:{port}",
            }

    target = RealSshTarget(
        host=host,
        port=port,
        username=username,
        key_path=str(key_path_obj),
        key_fingerprint=resolved.fingerprint,
        key_ref=ref,
    )
    transport = RealSshExecutor(
        target=target,
        connect_timeout_s=20.0,
        cmd_timeout_cap_s=90.0,
        wall_budget_s=240.0,
    )
    claimed = ClaimedInventory(gpu_model=claimed_model, gpu_count=int(claimed_gpu_count))
    ctx = GpuProbeContext(
        node_id=node_id,
        provider_hotkey=provider_hotkey,
        ssh_endpoint=f"{host}:{port}",
        claimed=claimed,
        key_fingerprint=resolved.fingerprint,
    )
    from typing import Literal

    mode_norm: Literal["full", "quick"] = "quick" if str(mode).lower() == "quick" else "full"
    config = GpuProbeConfig(
        mode=mode_norm,
        require_docker_runtime=bool(require_docker),
        skip_microbench=mode_norm == "quick",
    )
    try:
        evidence = run_gpu_probe(transport, ctx, config=config)
    finally:
        transport.close()

    public = evidence.to_public()
    # Prefer measured primary GPU name for class compare.
    measured_names = [g.get("name") for g in (public.get("measured") or {}).get("gpus") or []]
    measured_primary = next((n for n in measured_names if n), None)
    measured_family = normalize_gpu_model(measured_primary) if measured_primary else None
    claimed_family = normalize_gpu_model(claimed_model)
    class_match = (
        claimed_family is not None
        and measured_family is not None
        and claimed_family == measured_family
    )
    pack: dict[str, Any] = {
        "status": public.get("status"),
        "ok": public.get("status") == "passed",
        "evidence_id": public.get("id") or public.get("evidence_id"),
        "mode": public.get("mode"),
        "transport": public.get("transport") or "real",
        "failure_code": public.get("failure_code"),
        "ssh_endpoint": f"{host}:{port}",
        "node_id": node_id,
        "provider_hotkey": provider_hotkey,
        "key_fingerprint": resolved.fingerprint,
        "key_ref": {"kind": "file", "name": str(key_path_obj)},
        "claimed": {
            "gpu_model": claimed_model,
            "gpu_count": claimed_gpu_count,
            "family": claimed_family,
        },
        "measured": public.get("measured"),
        "measured_primary_name": measured_primary,
        "measured_family": measured_family,
        "claim_model_class_match": class_match,
        "checks": public.get("checks"),
        "advisories": public.get("advisories"),
        "digests": public.get("digests"),
        "raw_redacted": public.get("raw_redacted"),
        "wait_ssh": wait_info,
        "elapsed_s": time.time() - started,
        "source": "scripts.qa.host_gpu_probe",
        "product_verda_adapter": False,
        "set_weights": False,
    }
    return pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M9 host GPU probe (ops, real SSH)")
    parser.add_argument("--host", default="", help="SSH host IP/DNS")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument(
        "--endpoint",
        default="",
        help="host:port alternative to --host/--port",
    )
    parser.add_argument(
        "--key-file",
        default="",
        help="Path to private key (0600); never logged as PEM",
    )
    parser.add_argument(
        "--key-ref",
        default="",
        help="file:/path form (preferred for product key_ref parity)",
    )
    parser.add_argument("--username", default="root")
    parser.add_argument("--claimed-model", required=True, help="Catalog/claim GPU model")
    parser.add_argument("--claimed-gpu-count", type=int, default=1)
    parser.add_argument("--node-id", default=None)
    parser.add_argument("--provider-hotkey", default=None)
    parser.add_argument("--mode", default="full", choices=["full", "quick"])
    parser.add_argument("--require-docker", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    parser.add_argument(
        "--out",
        default="host_probe.json",
        help="Output host_probe.json path",
    )
    args = parser.parse_args(argv)

    host = args.host
    port = int(args.port)
    if args.endpoint:
        host, port = parse_ssh_endpoint(args.endpoint)
    if not host:
        print("error: --host or --endpoint required", file=sys.stderr)
        return 1

    key_path = args.key_file
    if args.key_ref:
        raw = args.key_ref
        if raw.startswith("file:"):
            key_path = raw[5:]
        elif raw.startswith("env:"):
            import os

            val = os.environ.get(raw[4:], "")
            key_path = val if Path(val).expanduser().is_file() else ""
            if not key_path:
                print(f"error: env key_ref {raw!r} not a file path", file=sys.stderr)
                return 1
        else:
            key_path = raw
    if not key_path:
        print("error: --key-file or --key-ref required", file=sys.stderr)
        return 1

    try:
        pack = run_host_gpu_probe(
            host=host,
            port=port,
            key_path=key_path,
            username=args.username,
            claimed_model=args.claimed_model,
            claimed_gpu_count=args.claimed_gpu_count,
            node_id=args.node_id,
            provider_hotkey=args.provider_hotkey,
            mode=args.mode,
            require_docker=args.require_docker,
            skip_wait=args.skip_wait,
            wait_timeout_s=args.wait_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        pack = {
            "status": "error",
            "ok": False,
            "failure_code": "host_probe_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }

    out_path = Path(args.out)
    write_json(out_path, pack)
    # Human summary without secrets
    summary = {
        "ok": pack.get("ok"),
        "status": pack.get("status"),
        "claim_model_class_match": pack.get("claim_model_class_match"),
        "claimed_family": (pack.get("claimed") or {}).get("family"),
        "measured_family": pack.get("measured_family"),
        "evidence_id": pack.get("evidence_id"),
        "out": str(out_path),
        "failure_code": pack.get("failure_code"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if pack.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
