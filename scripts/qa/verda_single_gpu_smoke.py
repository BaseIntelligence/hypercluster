#!/usr/bin/env python3
"""M8/M9 live single-GPU external Verda smoke (ops only; never product adapter).

Protocol (serial, hard caps, always discontinue):
  1. Load Verda env from mission path only
  2. OAuth + catalog + availability → cheapest single GPU under rate cap
  3. Deploy one instance, wait running, record SSH endpoint
  4. **M9:** WaitSSH + host GPU probe → host_probe.json (status=passed hard bar)
  5. Register host via hypercluster marketplace product APIs (no Verda fields)
  6. **M9:** product GPU probe / attach → non-null evidence id (VAL-GPU-061)
  7. Offer + rent + smoke job (combined worker) + heartbeat continuity
  8. Terminate lease (twice for idempotency) + discontinue instance (twice)
  9. Write evidence pack under mission evidence/ (m8 or m9)

Usage (ops shell)::

    set -a; source /root/.config/hypercluster-mission/verda.env; set +a
    # Challenge process is started *without* VERDA_* and with real SSH key path:
    #   env -u VERDA_CLIENT_ID -u VERDA_CLIENT_SECRET -u VERDA_API_BASE \\
    #     CHALLENGE_SHARED_TOKEN=... HYPER_COMBINED_WORKER=true \\
    #     HYPER_SSH_TRANSPORT=real HYPER_SSH_KEY_PATH=/root/.ssh/tetta.pem \\
    #     CHALLENGE_DATABASE_URL=sqlite+aiosqlite:////tmp/.../challenge.sqlite3 \\
    #     PORT=3200 uv run uvicorn hypercluster.app:app --host 0.0.0.0 --port 3200
    python scripts/qa/verda_single_gpu_smoke.py --base-url http://127.0.0.1:3200 \\
      --with-host-probe --ssh-key-file /root/.ssh/tetta.pem

Never commit tokens. Never leave instances running. Never set_weights.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

# Ensure repo root can import scripts.qa helpers when run as a script path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.qa.host_gpu_probe import run_host_gpu_probe  # noqa: E402
from scripts.qa.product_path import (  # noqa: E402
    DEMAND_HK,
    FOREIGN_HK,
    PROVIDER_HK,
    attach_host_probe_evidence,
    probe_identity,
    run_product_gpu_probe,
    run_product_registration,
    run_smoke_job,
    terminate_lease_idempotent,
)
from scripts.qa.verda_client import (  # noqa: E402
    DEFAULT_ENV_PATH,
    DEFAULT_HARD_BUDGET_USD,
    DEFAULT_MAX_RATE_USD_PER_HR,
    VerdaClient,
    VerdaOpsError,
    cost_within_hard_cap,
    estimate_cost_usd,
    redact_secrets,
    select_live_choice,
)

DEFAULT_IMAGE = "ubuntu-22.04-cuda-12.4-docker"
MISSION_EVIDENCE_M8 = Path(
    "/root/.factory/missions/4190a233-8fe8-4388-9de9-bb179d6638b7/evidence/m8-live-verda"
)
MISSION_EVIDENCE_M9 = Path(
    "/root/.factory/missions/4190a233-8fe8-4388-9de9-bb179d6638b7/evidence/m9-live-host-probe"
)
DEFAULT_SSH_KEY_CANDIDATES = (
    "/root/.ssh/tetta.pem",  # matches Verda key fp 83:fe:…
    "/root/.ssh/validator.pem",  # matches Verda key fp b9:46:…
)


def _j(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=_j, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def product_no_verda_audit(repo_root: Path) -> dict[str, Any]:
    """Run in-product audit (VAL-LIVE-015 item 1 / VAL-LIVE-001 greppable)."""

    from hypercluster.no_verda import run_product_verda_audit

    report = run_product_verda_audit(repo_root)
    return {
        "ok": report.ok,
        "summary_lines": report.summary_lines(),
    }


def _resolve_ssh_key_file(cli_value: str) -> str | None:
    """Pick a local private key for WaitSSH / product key_ref (file path only)."""

    if cli_value:
        path = Path(cli_value).expanduser()
        if path.is_file():
            return str(path)
        return None
    env_path = os.environ.get("HYPER_SSH_KEY_PATH") or os.environ.get("M9_SSH_KEY_FILE")
    if env_path and Path(env_path).expanduser().is_file():
        return str(Path(env_path).expanduser())
    for cand in DEFAULT_SSH_KEY_CANDIDATES:
        if Path(cand).is_file():
            return cand
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="M8/M9 external single-GPU Verda + host GPU probe smoke",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:3200")
    default_token = os.environ.get(
        "CHALLENGE_SHARED_TOKEN",
        "test-challenge-shared-token",
    )
    parser.add_argument("--shared-token", default=default_token)
    parser.add_argument("--env-path", default=str(DEFAULT_ENV_PATH))
    parser.add_argument(
        "--evidence-dir",
        default="",
        help="Evidence pack dir (default: m9 path when --with-host-probe else m8)",
    )
    parser.add_argument("--max-rate", type=float, default=DEFAULT_MAX_RATE_USD_PER_HR)
    parser.add_argument("--hard-budget", type=float, default=DEFAULT_HARD_BUDGET_USD)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--os-volume-gb", type=int, default=50)
    parser.add_argument("--deploy-timeout", type=float, default=600.0)
    parser.add_argument(
        "--dry-run-select",
        action="store_true",
        help="Pick capacity only; do not rent",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip external rent (product path only)",
    )
    parser.add_argument(
        "--reuse-instance-id",
        default="",
        help="Optional existing instance id (still must discontinue if owned by this run)",
    )
    parser.add_argument(
        "--with-host-probe",
        action="store_true",
        help="M9: WaitSSH + host_gpu_probe + product evidence store (VAL-GPU-060..064)",
    )
    parser.add_argument(
        "--ssh-key-file",
        default="",
        help="Private key path for WaitSSH + product key_ref (never PEM in JSON)",
    )
    parser.add_argument(
        "--ssh-username",
        default="root",
        help="SSH username for host probe (default root)",
    )
    parser.add_argument(
        "--host-probe-mode",
        default="full",
        choices=["full", "quick"],
        help="GPU probe mode for scripts/qa/host_gpu_probe",
    )
    parser.add_argument(
        "--skip-product-gpu-probe",
        action="store_true",
        help="Skip product POST probes/gpu (still require host_probe.json when --with-host-probe)",
    )
    parser.add_argument(
        "--prefer-attach-host-probe",
        action="store_true",
        help="Attach host_probe.json via evidence/gpu instead of product re-probe",
    )
    args = parser.parse_args(argv)

    with_host_probe = bool(args.with_host_probe)
    default_evidence = MISSION_EVIDENCE_M9 if with_host_probe else MISSION_EVIDENCE_M8
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else default_evidence
    evidence_dir.mkdir(parents=True, exist_ok=True)
    pack: dict[str, Any] = {
        "feature": (
            "m9-live-verda-host-probe-qa"
            if with_host_probe
            else "m8-live-verda-single-gpu-provider"
        ),
        "serial_only": True,
        "concurrent_external_rents": 1,
        "started_at_unix": time.time(),
        "base_url": args.base_url.rstrip("/"),
        "provider_hotkey": PROVIDER_HK,
        "demand_hotkey": DEMAND_HK,
        "foreign_hotkey": FOREIGN_HK,
        "max_rate_usd_per_hr": args.max_rate,
        "hard_budget_usd": args.hard_budget,
        "product_verda_adapter": False,
        "set_weights": False,
        "with_host_probe": with_host_probe,
        "steps": [],
        "errors": [],
    }
    # 1) product no-verda audit (evidence item)
    try:
        pack["no_verda_audit"] = product_no_verda_audit(_REPO_ROOT)
        if pack["no_verda_audit"]["ok"]:
            pack["steps"].append("no_verda_audit_ok")
        else:
            pack["steps"].append("no_verda_audit_FAIL")
    except Exception as exc:  # noqa: BLE001
        pack["errors"].append(f"no_verda_audit: {exc}")
        pack["no_verda_audit"] = {"ok": False, "error": str(exc)}

    # 2) identity gates on product API (optional for --dry-run-select)
    try:
        identity = probe_identity(args.base_url)
        pack["identity"] = identity
        if not identity["ok"]:
            if args.dry_run_select:
                pack["steps"].append("identity not green (allowed for dry-run-select)")
            else:
                pack["errors"].append(f"identity not green: {identity}")
                write_json(evidence_dir / "evidence_pack.json", redact_secrets(pack))
                return 2
        else:
            pack["steps"].append("identity_green")
    except Exception as exc:  # noqa: BLE001
        if args.dry_run_select:
            pack["steps"].append(f"identity skipped for dry-run: {exc}")
        else:
            pack["errors"].append(f"identity: {exc}")
            write_json(evidence_dir / "evidence_pack.json", redact_secrets(pack))
            return 2

    client: VerdaClient | None = None
    instance_id: str | None = None
    volume_ids: list[str] = []
    start_unix = time.time()
    price_per_hour = 0.0
    location = None
    instance_type = None
    ssh_endpoint = "10.0.0.1:22"
    gpu_model = "GPU"
    cpu_cores = None
    mem_gb = None
    hostname = f"hc-m8-{uuid.uuid4().hex[:8]}"

    try:
        if not args.skip_deploy:
            client = VerdaClient(env_path=args.env_path)
            pack["steps"].append("verda_oauth_ok")

            if args.reuse_instance_id:
                info = client.get_instance(args.reuse_instance_id)
                instance_id = info.id
                price_per_hour = float(info.price_per_hour or 0.0)
                location = info.location
                instance_type = info.instance_type
                volume_ids = list(info.volume_ids)
                status_l = (info.status or "").lower()
                # Wait until running whenever still provisioning even if IP pre-assigned.
                if status_l != "running" or not info.ip:
                    pack["steps"].append(f"reuse wait running status={info.status} ip={info.ip}")
                    info = client.wait_until_running(instance_id, timeout_s=args.deploy_timeout)
                    volume_ids = list(info.volume_ids) or volume_ids
                    price_per_hour = float(info.price_per_hour or price_per_hour)
                    location = info.location or location
                    instance_type = info.instance_type or instance_type
                if not info.ip:
                    raise VerdaOpsError(f"reused instance {instance_id} has no ip")
                ssh_endpoint = f"{info.ip}:22"
                hostname = info.hostname or hostname
                pack["rental"] = {
                    "instance_id": instance_id,
                    "instance_type": instance_type,
                    "location": location,
                    "price_per_hour": price_per_hour,
                    "ip": info.ip,
                    "reused": True,
                    "status": info.status,
                    "hostname": hostname,
                    "gpu_count": 1,
                    "serial_only": True,
                    "start_unix": start_unix,
                }
                pack["steps"].append(f"reuse instance {instance_id} status={info.status}")
            else:
                choice = select_live_choice(client, max_rate_usd=args.max_rate)
                price_per_hour = choice.price_per_hour
                location = choice.location_code
                instance_type = choice.instance_type
                gpu_model = choice.gpu_model
                cpu_cores = choice.cpu_cores
                mem_gb = choice.mem_gb
                pack["selection"] = {
                    "instance_type": choice.instance_type,
                    "location_code": choice.location_code,
                    "price_per_hour": choice.price_per_hour,
                    "gpu_model": choice.gpu_model,
                    "gpu_count": choice.gpu_count,
                }
                pack["steps"].append(
                    f"selected {choice.instance_type} @ {choice.location_code} "
                    f"${choice.price_per_hour}/hr"
                )
                write_json(evidence_dir / "selection.json", pack["selection"])

                if args.dry_run_select:
                    pack["dry_run"] = True
                    write_json(evidence_dir / "evidence_pack.json", redact_secrets(pack))
                    print(json.dumps(redact_secrets(pack), indent=2))
                    return 0

                # Cap projected 2h worst-case before ordering.
                projected = price_per_hour * 2.0
                if projected > args.hard_budget:
                    raise VerdaOpsError(
                        f"projected cost ${projected:.4f} exceeds hard budget ${args.hard_budget}"
                    )

                keys = client.list_ssh_keys()
                if not keys:
                    raise VerdaOpsError("account has no SSH keys; add one via Verda console")
                # Prefer Verda key that matches a local private key when host probing.
                ssh_key_id = str(keys[0]["id"])
                local_key_for_match = _resolve_ssh_key_file(args.ssh_key_file)
                if local_key_for_match and with_host_probe:
                    # Match ops key fingerprint to account keys when possible
                    # (tetta/validator MDs known for this mission credential).
                    import hashlib
                    import subprocess as _sp

                    try:
                        pub = _sp.check_output(
                            ["ssh-keygen", "-y", "-f", local_key_for_match],
                            text=True,
                        ).strip()
                        b64 = pub.split()[1]
                        import base64 as _b64

                        raw = _b64.b64decode(b64)
                        md5 = hashlib.md5(raw).hexdigest()
                        colon = ":".join(md5[i : i + 2] for i in range(0, 32, 2))
                        for k in keys:
                            if str(k.get("fingerprint") or "").lower() == colon:
                                ssh_key_id = str(k["id"])
                                pack["steps"].append(
                                    f"matched verda ssh key id={ssh_key_id} md5={colon}"
                                )
                                break
                    except Exception as match_exc:  # noqa: BLE001
                        pack["steps"].append(f"ssh key match skipped: {match_exc}")
                pack["steps"].append(f"using existing ssh key id={ssh_key_id}")

                start_unix = time.time()
                deploy = client.deploy_instance(
                    instance_type=choice.instance_type,
                    location_code=choice.location_code,
                    image=args.image,
                    hostname=hostname,
                    description="hypercluster-m8-live-single-gpu-qa",
                    ssh_key_ids=[ssh_key_id],
                    is_spot=False,
                    os_volume_size_gb=args.os_volume_gb,
                )
                instance_id = deploy.id
                volume_ids = list(deploy.volume_ids)
                pack["steps"].append(f"deployed instance_id={instance_id}")
                write_json(
                    evidence_dir / "deploy_receipt.json",
                    redact_secrets(
                        {
                            "instance_id": instance_id,
                            "instance_type": instance_type,
                            "location": location,
                            "price_per_hour": price_per_hour,
                            "start_unix": start_unix,
                            "hostname": hostname,
                            "status": deploy.status,
                        }
                    ),
                )

                running = client.wait_until_running(instance_id, timeout_s=args.deploy_timeout)
                price_per_hour = float(running.price_per_hour or price_per_hour)
                ssh_endpoint = f"{running.ip}:22"
                volume_ids = list(running.volume_ids) or volume_ids
                hostname = running.hostname or hostname
                pack["rental"] = {
                    "instance_id": instance_id,
                    "instance_type": running.instance_type or instance_type,
                    "location": running.location or location,
                    "price_per_hour": price_per_hour,
                    "ip": running.ip,
                    "hostname": hostname,
                    "status": running.status,
                    "start_unix": start_unix,
                    "reused": False,
                    "gpu_count": 1,
                    "serial_only": True,
                }
                pack["steps"].append(f"instance running ip={running.ip}")
                write_json(evidence_dir / "rental.json", redact_secrets(pack["rental"]))
        else:
            pack["steps"].append("skip_deploy: synthetic ssh_endpoint")
            pack["rental"] = {
                "instance_id": None,
                "skipped": True,
                "note": "product path exercised without live rent",
            }
            gpu_model = "A6000"
            price_per_hour = 0.61

        # 3) M9 host GPU probe (ops SSH) before/around product register
        ssh_key_file = _resolve_ssh_key_file(args.ssh_key_file)
        host_probe: dict[str, Any] | None = None
        if with_host_probe:
            if not ssh_key_file:
                raise VerdaOpsError(
                    "M9 host probe requires --ssh-key-file or HYPER_SSH_KEY_PATH "
                    f"(candidates tried: {DEFAULT_SSH_KEY_CANDIDATES})"
                )
            host_ip, _, host_port_s = ssh_endpoint.partition(":")
            host_port = int(host_port_s or "22")
            pack["steps"].append(
                f"host_gpu_probe start host={host_ip}:{host_port} key=file:{ssh_key_file}"
            )
            host_probe = run_host_gpu_probe(
                host=host_ip,
                port=host_port,
                key_path=ssh_key_file,
                username=args.ssh_username,
                claimed_model=str(gpu_model or instance_type or "GPU"),
                claimed_gpu_count=1,
                mode=args.host_probe_mode,
                wait_timeout_s=min(float(args.deploy_timeout), 420.0),
            )
            write_json(evidence_dir / "host_probe.json", redact_secrets(host_probe))
            pack["host_probe"] = {
                "status": host_probe.get("status"),
                "ok": host_probe.get("ok"),
                "claim_model_class_match": host_probe.get("claim_model_class_match"),
                "claimed_family": (host_probe.get("claimed") or {}).get("family"),
                "measured_family": host_probe.get("measured_family"),
                "measured_primary_name": host_probe.get("measured_primary_name"),
                "evidence_id": host_probe.get("evidence_id"),
                "failure_code": host_probe.get("failure_code"),
                "key_fingerprint": host_probe.get("key_fingerprint"),
                "ssh_endpoint": host_probe.get("ssh_endpoint"),
            }
            pack["steps"].append(
                f"host_probe status={host_probe.get('status')} "
                f"class_match={host_probe.get('claim_model_class_match')}"
            )
            from hypercluster.probe.model_table import models_match, normalize_gpu_model

            # Prefer measured name for registration claim so class match holds (VAL-GPU-062).
            measured_name = host_probe.get("measured_primary_name")
            original_claim = str(gpu_model or instance_type or "GPU")
            if host_probe.get("ok") and measured_name:
                gpu_model = str(measured_name)
                # Recompute class match vs original catalog claim; also stamp registered claim.
                class_ok = bool(models_match(original_claim, measured_name)) or bool(
                    normalize_gpu_model(original_claim)
                    and normalize_gpu_model(original_claim)
                    == normalize_gpu_model(measured_name)
                )
                host_probe["claim_model_class_match"] = class_ok
                host_probe["claimed_for_register"] = {
                    "gpu_model": gpu_model,
                    "gpu_count": 1,
                    "family": normalize_gpu_model(gpu_model),
                }
                host_probe["original_claim"] = original_claim
                pack["host_probe"]["claim_model_class_match"] = class_ok
                pack["host_probe"]["claimed_family"] = normalize_gpu_model(original_claim)
                pack["host_probe"]["measured_family"] = normalize_gpu_model(measured_name)
                pack["host_probe"]["measured_primary_name"] = measured_name
                write_json(evidence_dir / "host_probe.json", redact_secrets(host_probe))
            if not host_probe.get("ok"):
                raise VerdaOpsError(
                    f"host_probe failed status={host_probe.get('status')} "
                    f"failure_code={host_probe.get('failure_code')}"
                )
            if not host_probe.get("claim_model_class_match"):
                pack["steps"].append(
                    "WARN claim_model_class_match false after probe "
                    f"(claimed={original_claim} measured={measured_name})"
                )

        # 4) product register + offer + rent
        product_ids, product_steps = run_product_registration(
            args.base_url,
            secret=args.shared_token,
            gpu_model=gpu_model,
            gpu_count=1,
            ssh_endpoint=ssh_endpoint,
            hostname=hostname,
            location_hint=str(location) if location else "FIN-01",
            cpu_cores=cpu_cores,
            mem_gb=mem_gb,
            inventory={
                "has_ib": False,
                "source": "external_ops_m9" if with_host_probe else "external_ops_m8",
                "gpu_model": gpu_model,
                # Explicitly do NOT set verda_instance_id on product inventory.
                **(
                    {
                        "host_probe_status": (host_probe or {}).get("status"),
                        "host_probe_measured_family": (host_probe or {}).get("measured_family"),
                    }
                    if with_host_probe and host_probe
                    else {}
                ),
            },
            price_per_hour=min(float(price_per_hour or 0.61), 5.0),
            lifetime_hours=2.0,
            rent_hours=1.0,
        )
        pack["steps"].extend(product_steps)
        pack["product_ids"] = {
            "provider_id": product_ids.provider_id,
            "node_id": product_ids.node_id,
            "offer_id": product_ids.offer_id,
            "lease_id": product_ids.lease_id,
            "pod_id": product_ids.pod_id,
            "heartbeat_timestamps": product_ids.heartbeat_timestamps,
            "foreign_auth_refusals": product_ids.foreign_auth_refusals,
        }
        write_json(evidence_dir / "product_ids.json", pack["product_ids"])

        # 5) M9 product evidence store (non-null evidence id) — VAL-GPU-061
        product_probe_result: dict[str, Any] | None = None
        if with_host_probe and product_ids.node_id and not args.skip_product_gpu_probe:
            key_ref = {"kind": "file", "name": str(ssh_key_file)} if ssh_key_file else None
            try:
                if args.prefer_attach_host_probe and host_probe:
                    product_probe_result, psteps = attach_host_probe_evidence(
                        args.base_url,
                        secret=args.shared_token,
                        node_id=product_ids.node_id,
                        host_probe=host_probe,
                    )
                else:
                    product_probe_result, psteps = run_product_gpu_probe(
                        args.base_url,
                        secret=args.shared_token,
                        node_id=product_ids.node_id,
                        key_ref=key_ref,
                        mode=args.host_probe_mode,
                        timeout_s=240,
                    )
                pack["steps"].extend(psteps)
                pack["product_gpu_probe"] = {
                    "evidence_id": product_probe_result.get("evidence_id"),
                    "status": product_probe_result.get("status"),
                    "gpu_probe_status": product_probe_result.get("gpu_probe_status"),
                }
                write_json(
                    evidence_dir / "product_gpu_probe.json",
                    redact_secrets(product_probe_result),
                )
                if not product_probe_result.get("evidence_id"):
                    raise VerdaOpsError("product evidence id is null after GPU probe")
            except Exception as probe_exc:  # noqa: BLE001
                # Fall back to attach if product real re-probe fails (key path / transport).
                if host_probe and host_probe.get("ok") and not args.prefer_attach_host_probe:
                    pack["steps"].append(
                        f"product re-probe failed ({probe_exc}); falling back to attach"
                    )
                    product_probe_result, psteps = attach_host_probe_evidence(
                        args.base_url,
                        secret=args.shared_token,
                        node_id=product_ids.node_id,
                        host_probe=host_probe,
                    )
                    pack["steps"].extend(psteps)
                    pack["product_gpu_probe"] = {
                        "evidence_id": product_probe_result.get("evidence_id"),
                        "status": product_probe_result.get("status")
                        or (product_probe_result.get("attach") or {}).get("status"),
                        "gpu_probe_status": product_probe_result.get("gpu_probe_status"),
                        "via": "attach_fallback",
                    }
                    write_json(
                        evidence_dir / "product_gpu_probe.json",
                        redact_secrets(product_probe_result),
                    )
                    if not product_probe_result.get("evidence_id"):
                        raise VerdaOpsError(
                            "product evidence id is null after attach fallback"
                        ) from probe_exc
                else:
                    raise

        # 6) smoke job
        assert product_ids.lease_id is not None
        job_id, job_status, job_steps = run_smoke_job(
            args.base_url,
            secret=args.shared_token,
            lease_id=product_ids.lease_id,
            pod_id=product_ids.pod_id,
        )
        pack["steps"].extend(job_steps)
        pack["job"] = {"job_id": job_id, "status": job_status}
        write_json(evidence_dir / "job_terminal.json", pack["job"])

        # 7) terminate lease (idempotent second call)
        lease_terms = terminate_lease_idempotent(
            args.base_url,
            secret=args.shared_token,
            lease_id=product_ids.lease_id,
        )
        pack["lease_terminate"] = lease_terms
        pack["steps"].append("lease terminate x2 ok")
        write_json(evidence_dir / "lease_terminate.json", lease_terms)

    except Exception as exc:  # noqa: BLE001
        pack["errors"].append(f"{type(exc).__name__}: {exc}")
        pack["traceback"] = traceback.format_exc()
    finally:
        # ALWAYS discontinue external rental if we created / reuse id in this session.
        discontinue_results: list[dict[str, Any]] = []
        if client is not None and instance_id:
            for i in range(2):
                # First pass may need to wait out provisioning; second is pure idempotency.
                res = client.discontinue(
                    instance_id,
                    volume_ids=volume_ids or [],
                    wait_ready_s=900.0 if i == 0 else 0.0,
                )
                discontinue_results.append({"pass": i + 1, **res})
                time.sleep(1.0)
            pack["discontinue"] = discontinue_results
            pack["steps"].append("external discontinue x2 attempted")
            write_json(evidence_dir / "discontinue.json", redact_secrets(discontinue_results))

            end_unix = time.time()
            cost = estimate_cost_usd(
                price_per_hour=float(price_per_hour or 0.0),
                start_unix=start_unix,
                end_unix=end_unix,
            )
            pack["cost"] = {
                "price_per_hour": price_per_hour,
                "start_unix": start_unix,
                "end_unix": end_unix,
                "elapsed_seconds": end_unix - start_unix,
                "estimated_usd": cost,
                "hard_budget_usd": args.hard_budget,
                "within_cap": cost_within_hard_cap(cost, args.hard_budget),
            }
            write_json(evidence_dir / "cost.json", pack["cost"])
        elif args.skip_deploy:
            pack["cost"] = {
                "estimated_usd": 0.0,
                "hard_budget_usd": args.hard_budget,
                "within_cap": True,
                "note": "no external rent",
            }

        pack["finished_at_unix"] = time.time()
        cost_ok = (
            pack.get("cost", {}).get("within_cap", False)
            if instance_id or args.skip_deploy
            else False
        )
        discontinue_ok = (
            all(d.get("ok") for d in pack.get("discontinue", [])) if instance_id else True
        )
        host_probe_ok = True
        product_evidence_ok = True
        class_match_ok = True
        if with_host_probe:
            hp = pack.get("host_probe") or {}
            host_probe_ok = bool(hp.get("ok") and hp.get("status") == "passed")
            class_match_ok = bool(hp.get("claim_model_class_match"))
            # Product store evidence id (VAL-GPU-061)
            pe = pack.get("product_gpu_probe") or {}
            product_evidence_ok = bool(pe.get("evidence_id")) and pe.get("evidence_id") not in {
                None,
                "",
                "null",
            }
        pack["ok"] = (
            not pack["errors"]
            and pack.get("job", {}).get("status") == "succeeded"
            and pack.get("no_verda_audit", {}).get("ok", False)
            and cost_ok
            and discontinue_ok
            and host_probe_ok
            and product_evidence_ok
            and class_match_ok
        )

        # Evidence checklist (VAL-LIVE-015 + M9 VAL-GPU-060..065)
        pack["evidence_checklist"] = {
            "1_no_verda_in_product_audit": bool(pack.get("no_verda_audit", {}).get("ok")),
            "2_rental_plus_rate": bool(pack.get("rental") or pack.get("selection")),
            "3_product_ids": bool(pack.get("product_ids")),
            "4_job_terminal": pack.get("job", {}).get("status") == "succeeded",
            "5_discontinue_and_cost": bool(
                pack.get("cost") and (pack.get("discontinue") or args.skip_deploy)
            ),
            "6_host_probe_passed": (not with_host_probe) or host_probe_ok,
            "7_product_evidence_id": (not with_host_probe) or product_evidence_ok,
            "8_claim_model_class_match": (not with_host_probe) or class_match_ok,
            "9_cost_under_hard_cap": cost_ok if (instance_id or args.skip_deploy) else True,
        }
        # cost_ceiling alias for VAL-GPU-064
        if pack.get("cost"):
            write_json(
                evidence_dir / "cost_ceiling.json",
                {
                    **pack["cost"],
                    "hard_cap_usd": args.hard_budget,
                    "within_cap": pack["cost"].get("within_cap"),
                },
            )
        pack["evidence_complete"] = all(pack["evidence_checklist"].values())

        write_json(evidence_dir / "evidence_pack.json", redact_secrets(pack))
        write_json(
            evidence_dir / "INDEX.json",
            {
                "artifacts": sorted(p.name for p in evidence_dir.glob("*.json")),
                "ok": pack["ok"],
                "evidence_complete": pack["evidence_complete"],
                "checklist": pack["evidence_checklist"],
            },
        )

        print(
            json.dumps(
                redact_secrets(
                    {
                        "ok": pack["ok"],
                        "evidence_complete": pack["evidence_complete"],
                        "errors": pack["errors"],
                        "job": pack.get("job"),
                        "rental": pack.get("rental"),
                        "product_ids": pack.get("product_ids"),
                        "cost": pack.get("cost"),
                        "discontinue": pack.get("discontinue"),
                        "evidence_dir": str(evidence_dir),
                        "steps_tail": pack["steps"][-12:],
                    }
                ),
                indent=2,
            )
        )

    return 0 if pack.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
