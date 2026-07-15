#!/usr/bin/env python3
"""M8 live single-GPU external Verda smoke (ops only; never product adapter).

Protocol (serial, hard caps, always discontinue):
  1. Load Verda env from mission path only
  2. OAuth + catalog + availability → cheapest single GPU under rate cap
  3. Deploy one instance, wait running, record SSH endpoint
  4. Register host via hypercluster marketplace product APIs (no Verda fields)
  5. Offer + rent + smoke job (combined worker) + heartbeat continuity
  6. Terminate lease (twice for idempotency) + discontinue instance (twice)
  7. Write evidence pack under mission evidence/m8-live-verda/

Usage (ops shell)::

    set -a; source /root/.config/hypercluster-mission/verda.env; set +a
    # Challenge process is started *without* VERDA_* :
    #   env -u VERDA_CLIENT_ID -u VERDA_CLIENT_SECRET -u VERDA_API_BASE \\
    #     CHALLENGE_SHARED_TOKEN=... HYPER_COMBINED_WORKER=true \\
    #     CHALLENGE_DATABASE_URL=sqlite+aiosqlite:////tmp/.../challenge.sqlite3 \\
    #     PORT=3200 uv run uvicorn hypercluster.app:app --host 0.0.0.0 --port 3200
    python scripts/qa/verda_single_gpu_smoke.py --base-url http://127.0.0.1:3200

Never commit tokens. Never leave instances running.
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

from scripts.qa.product_path import (  # noqa: E402
    DEMAND_HK,
    FOREIGN_HK,
    PROVIDER_HK,
    probe_identity,
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
MISSION_EVIDENCE = Path(
    "/root/.factory/missions/4190a233-8fe8-4388-9de9-bb179d6638b7/evidence/m8-live-verda"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="M8 external single-GPU Verda to marketplace smoke",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:3200")
    default_token = os.environ.get(
        "CHALLENGE_SHARED_TOKEN",
        "test-challenge-shared-token",
    )
    parser.add_argument("--shared-token", default=default_token)
    parser.add_argument("--env-path", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--evidence-dir", default=str(MISSION_EVIDENCE))
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
    args = parser.parse_args(argv)

    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    pack: dict[str, Any] = {
        "feature": "m8-live-verda-single-gpu-provider",
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

    # 2) identity gates on product API
    try:
        identity = probe_identity(args.base_url)
        pack["identity"] = identity
        if not identity["ok"]:
            pack["errors"].append(f"identity not green: {identity}")
            write_json(evidence_dir / "evidence_pack.json", redact_secrets(pack))
            return 2
        pack["steps"].append("identity_green")
    except Exception as exc:  # noqa: BLE001
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
                ssh_key_id = str(keys[0]["id"])
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

        # 3) product register + offer + rent
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
                "source": "external_ops_m8",
                "gpu_model": gpu_model,
                # Explicitly do NOT set verda_instance_id on product inventory.
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

        # 4) smoke job
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

        # 5) terminate lease (idempotent second call)
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
        pack["ok"] = (
            not pack["errors"]
            and pack.get("job", {}).get("status") == "succeeded"
            and pack.get("no_verda_audit", {}).get("ok", False)
            and (
                pack.get("cost", {}).get("within_cap", False)
                if instance_id or args.skip_deploy
                else False
            )
            and (all(d.get("ok") for d in pack.get("discontinue", [])) if instance_id else True)
        )

        # Evidence checklist (VAL-LIVE-015)
        pack["evidence_checklist"] = {
            "1_no_verda_in_product_audit": bool(pack.get("no_verda_audit", {}).get("ok")),
            "2_rental_plus_rate": bool(pack.get("rental") or pack.get("selection")),
            "3_product_ids": bool(pack.get("product_ids")),
            "4_job_terminal": pack.get("job", {}).get("status") == "succeeded",
            "5_discontinue_and_cost": bool(
                pack.get("cost") and (pack.get("discontinue") or args.skip_deploy)
            ),
        }
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
