"""Hypercluster Typer CLI — scaffold health/version + sim doctor/smoke.

Architecture outline (expanded in later milestones):
  serve, version, health --url, marketplace, nodes, jobs, fabric, attest,
  score, weights, sim {seed, run-scenario, doctor}
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import typer

from hypercluster import __version__
from hypercluster.sim.ports import (
    DEFAULT_BAREMETAL_PORT,
    assert_mission_port,
    is_mission_port,
    parse_port_from_url,
)

DEFAULT_BASE_URL = f"http://127.0.0.1:{DEFAULT_BAREMETAL_PORT}"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Hypercluster challenge CLI",
)
sim_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local simulator harness (doctor, scenarios).",
)
nodes_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Provider node register / heartbeat / fabric-scan.",
)
fabric_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Fabric plan dry-run and multi-node report views.",
)
fabric_report_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Show fabric report digests for jobs.",
)
app.add_typer(sim_app, name="sim")
app.add_typer(nodes_app, name="nodes")
app.add_typer(fabric_app, name="fabric")
fabric_app.add_typer(fabric_report_app, name="report")


def default_base_url() -> str:
    """Bare-metal default base URL (port 3200 mission band)."""

    return DEFAULT_BASE_URL


def _evaluate_health_payload(payload: dict[str, Any]) -> int:
    """CLI policy: exit 0 only when status is exactly ``ok``.

    Unhealthy / degraded / missing status → non-zero (VAL-SCAF-031).
    """

    status = payload.get("status")
    if status == "ok":
        return 0
    return 1


def _resolve_base_url(
    url: str | None,
    host: str | None,
    port: int | None,
    *,
    enforce_mission_port: bool = False,
) -> str:
    """Resolve CLI base URL from --url or host/port defaults."""

    if url:
        resolved = url.rstrip("/")
        port_from_url = parse_port_from_url(resolved)
        if enforce_mission_port and port_from_url is not None and not is_mission_port(
            port_from_url
        ):
            assert_mission_port(port_from_url)
        return resolved

    resolved_host = host or "127.0.0.1"
    resolved_port = DEFAULT_BAREMETAL_PORT if port is None else int(port)
    if enforce_mission_port:
        assert_mission_port(resolved_port)
    return f"http://{resolved_host}:{resolved_port}"


def _url_option() -> Any:
    return typer.Option(
        None,
        "--url",
        help=f"Live challenge base URL (default {DEFAULT_BASE_URL})",
    )


@app.command("version")
def version_cmd(
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(
        None,
        help="API port when --url omitted (mission band 3200–3299)",
    ),
) -> None:
    """Print package version; with --url, match live GET /version identity."""

    if url is None and host is None and port is None:
        typer.echo(f"hypercluster {__version__}")
        raise typer.Exit(code=0)

    base = _resolve_base_url(url, host, port)
    try:
        response = httpx.get(f"{base}/version", timeout=5.0)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        typer.echo(f"version probe failed for {base}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    challenge_version = body.get("challenge_version", "")
    challenge_slug = body.get("challenge_slug", "")
    role = body.get("role", "")
    # Prefer JSON dump for incorporate tests / jq parity tooling.
    typer.echo(json.dumps(body, indent=2, sort_keys=True))
    typer.echo(
        f"package={__version__} challenge_version={challenge_version} "
        f"slug={challenge_slug} role={role}"
    )
    if challenge_slug and challenge_slug != "hypercluster":
        typer.echo("error: live challenge_slug is not hypercluster", err=True)
        raise typer.Exit(code=1)
    if challenge_version and challenge_version != __version__:
        typer.echo(
            f"warning: package {__version__} != live challenge_version "
            f"{challenge_version}",
            err=True,
        )
    raise typer.Exit(code=0)


@app.command("health")
def health_cmd(
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(
        None,
        help="API port when --url omitted (default 3200, band 3200–3299)",
    ),
) -> None:
    """Probe live ``/health``; exit 0 only when status=ok (VAL-SCAF-031)."""

    base = _resolve_base_url(url, host, port)
    try:
        response = httpx.get(f"{base}/health", timeout=5.0)
    except httpx.HTTPError as exc:
        typer.echo(f"health probe failed for {base}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    text = response.text
    typer.echo(f"{response.status_code} {text}")
    if response.status_code != 200:
        raise typer.Exit(code=1)
    try:
        payload = response.json()
    except ValueError:
        typer.echo("health body is not JSON", err=True)
        raise typer.Exit(code=1) from None

    if not isinstance(payload, dict):
        typer.echo("health body is not a JSON object", err=True)
        raise typer.Exit(code=1)

    code = _evaluate_health_payload(payload)
    if code != 0:
        typer.echo(
            f"health status={payload.get('status')!r} not ok (base_url={base})",
            err=True,
        )
    raise typer.Exit(code=code)


@sim_app.command("doctor")
def sim_doctor_cmd(
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(
        None,
        help="API port when --url omitted (default 3200)",
    ),
) -> None:
    """CI readiness: identity gates + sim backend stubs (VAL-SCAF-036)."""

    from hypercluster.sim.doctor import run_doctor

    base = _resolve_base_url(url, host, port)
    report = run_doctor(base)
    for line in report.summary_lines():
        typer.echo(line)
    raise typer.Exit(code=0 if report.ok else 1)


@sim_app.command("run-scenario")
def sim_run_scenario_cmd(
    name: str = typer.Option(..., "--name", help="Scenario: smoke|marketplace|nccl|..."),
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(
        None,
        help="API port when --url omitted (default 3200)",
    ),
) -> None:
    """Run a local sim scenario. Smoke requires identity green first."""

    from hypercluster.sim.scenarios import run_scenario

    base = _resolve_base_url(url, host, port)
    result = run_scenario(name, base)
    for line in result.summary_lines():
        typer.echo(line)
    raise typer.Exit(code=0 if result.ok else 1)


@sim_app.command("seed")
def sim_seed_cmd(
    node_count: int = typer.Option(4, "--node-count", help="Virtual node count"),
    gpus_per_node: int = typer.Option(2, "--gpus-per-node", help="GPUs per sim node"),
    seed: int = typer.Option(0, "--seed", help="Deterministic inventory seed"),
) -> None:
    """Seed synthetic IB/NVLink multi-node inventory (VAL-FAB-019)."""

    from hypercluster.sim.inventory import plan_readiness, seed_sim_inventory

    inventory = seed_sim_inventory(
        seed=seed,
        node_count=node_count,
        gpus_per_node=gpus_per_node,
    )
    readiness = plan_readiness(
        inventory,
        world_size=min(node_count * gpus_per_node, max(2, gpus_per_node * 2)),
        nnodes=min(node_count, 2),
        nproc_per_node=gpus_per_node,
    )
    summary = {
        "nodes": len(inventory.nodes),
        "ib_edges": len(inventory.ib_edges),
        "nvlink_edges": len(inventory.nvlink_edges),
        "graph_digest": inventory.graph_digest,
        "seed": seed,
        "plan_ready": readiness.ok,
        "plan_nnodes_used": readiness.nnodes_used,
        "report_digests": [n.fabric_report.report_digest for n in inventory.nodes],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    typer.echo(
        f"sim seed: nodes={len(inventory.nodes)} graph_digest={inventory.graph_digest} "
        f"plan_ready={readiness.ok}"
    )
    raise typer.Exit(code=0 if readiness.ok else 1)


@nodes_app.command("fabric-scan")
def nodes_fabric_scan_cmd(
    node_id: str = typer.Option(..., "--node-id", help="Registered node id"),
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    seed: int = typer.Option(0, "--seed", help="Sim scan seed"),
    source: str = typer.Option("sim", "--source", help="sim|scan|inject|manual"),
    hotkey: str | None = typer.Option(
        None,
        "--hotkey",
        help="Provider hotkey (or HYPER_HOTKEY / CHALLENGE_HOTKEY env)",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Challenge shared token (or CHALLENGE_SHARED_TOKEN env)",
    ),
) -> None:
    """Run fabric-scan for a node and print the accepted FabricReport (VAL-FAB-018)."""

    from hypercluster.api.auth import build_signed_headers

    base = _resolve_base_url(url, host, port)
    resolved_hotkey = (
        hotkey
        or os.environ.get("HYPER_HOTKEY")
        or os.environ.get("CHALLENGE_HOTKEY")
        or "sim-fab-cli-hotkey"
    )
    resolved_token = (
        token
        or os.environ.get("CHALLENGE_SHARED_TOKEN")
        or os.environ.get("HYPER_SHARED_TOKEN")
        or ""
    )
    if not resolved_token:
        typer.echo(
            "fabric-scan requires --token or CHALLENGE_SHARED_TOKEN for signed request",
            err=True,
        )
        raise typer.Exit(code=1)

    body = {"source": source, "seed": seed}
    raw = json.dumps(body).encode()
    headers = build_signed_headers(
        secret=resolved_token,
        hotkey=resolved_hotkey,
        body=raw,
    )
    headers["Content-Type"] = "application/json"
    try:
        response = httpx.post(
            f"{base}/v1/nodes/{node_id}/fabric-scan",
            content=raw,
            headers=headers,
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        typer.echo(f"fabric-scan failed for {base}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"{response.status_code} {response.text}")
    if response.status_code == 404:
        raise typer.Exit(code=1)
    if response.status_code >= 400:
        raise typer.Exit(code=1)
    try:
        payload = response.json()
    except ValueError:
        raise typer.Exit(code=1) from None
    digest = payload.get("report_digest")
    if not digest:
        typer.echo("fabric-scan response missing report_digest", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"accepted report_digest={digest} node_id={payload.get('node_id')}")
    raise typer.Exit(code=0)


@fabric_app.command("plan")
def fabric_plan_cmd(
    job_id: str | None = typer.Option(
        None,
        "--job-id",
        help="Optional job id to load world_size/nnodes (dry-run; no mutual job mutate)",
    ),
    world_size: int = typer.Option(4, "--world-size", help="World size for plan (spec path)"),
    nnodes: int = typer.Option(2, "--nnodes", help="Target node count upper bound"),
    nproc_per_node: int = typer.Option(2, "--nproc-per-node", help="Processes per node"),
    policy: str = typer.Option("pack", "--policy", help="pack|spread"),
    fabric: str = typer.Option("auto", "--fabric", help="auto|ib|eth|nvlink_only"),
    seed: int = typer.Option(0, "--seed", help="Sim inventory seed"),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Dry-run only (default): never launches ranks or mutates job state",
    ),
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
) -> None:
    """Topology plan dry-run (VAL-FAB-016). Never advances job to running."""

    from hypercluster.fabric.planner import PlacementRequest, place_ranks
    from hypercluster.sim.inventory import seed_sim_inventory

    _ = dry_run  # dry-run is always true for this command surface
    job_payload: dict[str, Any] | None = None
    if job_id:
        base = _resolve_base_url(url, host, port)
        try:
            response = httpx.get(f"{base}/v1/jobs/{job_id}", timeout=10.0)
            if response.status_code >= 400:
                typer.echo(
                    f"fabric plan: failed to load job {job_id}: "
                    f"{response.status_code} {response.text}",
                    err=True,
                )
                raise typer.Exit(code=1)
            job_payload = response.json()
        except typer.Exit:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            typer.echo(f"fabric plan: failed to load job {job_id}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        world_size = int(job_payload.get("world_size") or world_size)
        nnodes = int(job_payload.get("nnodes") or nnodes)
        nproc_per_node = int(job_payload.get("nproc_per_node") or nproc_per_node)
        policy = str(job_payload.get("placement_policy") or policy)
        fabric = str(job_payload.get("fabric_mode") or job_payload.get("fabric") or fabric)

    inv = seed_sim_inventory(
        seed=seed,
        node_count=max(nnodes, 2),
        gpus_per_node=max(nproc_per_node, 1),
    )
    plan = place_ranks(
        PlacementRequest(
            job_id=job_id or f"dry-run-{seed}",
            world_size=world_size,
            nnodes=nnodes,
            nproc_per_node=nproc_per_node,
            policy=policy if policy in {"pack", "spread"} else "pack",  # type: ignore[arg-type]
            fabric=fabric if fabric in {"auto", "ib", "eth", "nvlink_only"} else "auto",  # type: ignore[arg-type]
            node_reports=inv.reports(),
        )
    )
    out = {
        "ok": plan.ok,
        "dry_run": True,
        "job_id": job_id,
        "rankmap": [b.to_public() for b in plan.rankmap],
        "nccl_env": dict(plan.nccl_env),
        "planner_version": plan.planner_version,
        "graph_digest": plan.graph_digest,
        "nnodes_used": plan.nnodes_used,
        "policy": plan.policy,
        "fabric": plan.fabric,
        "reason": plan.reason,
        "failure_code": plan.failure_code,
        "job_status_unchanged": True,
    }
    typer.echo(json.dumps(out, indent=2, sort_keys=True))
    if not plan.ok:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@fabric_report_app.command("show")
def fabric_report_show_cmd(
    job_id: str = typer.Option(..., "--job-id", help="Completed job id"),
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
) -> None:
    """Echo fabric report_digest for a completed job (VAL-FAB-017)."""

    base = _resolve_base_url(url, host, port)
    try:
        response = httpx.get(f"{base}/v1/jobs/{job_id}/fabric-report", timeout=10.0)
    except httpx.HTTPError as exc:
        typer.echo(f"fabric report show failed for {base}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"{response.status_code} {response.text}")
    if response.status_code != 200:
        raise typer.Exit(code=1)
    try:
        payload = response.json()
    except ValueError:
        typer.echo("fabric report body is not JSON", err=True)
        raise typer.Exit(code=1) from None

    digest = payload.get("report_digest") or payload.get("fabric_report_digest")
    if not digest:
        typer.echo("fabric report missing report_digest", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"report_digest={digest}")
    if payload.get("nodes"):
        typer.echo(f"node_count={len(payload['nodes'])}")
    raise typer.Exit(code=0)


def main() -> None:
    """Console script entry."""

    app()


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_BASE_URL",
    "app",
    "default_base_url",
    "_evaluate_health_payload",
    "main",
]
