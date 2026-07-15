"""Hypercluster Typer CLI — packaging core + domain subcommands.

Groups (VAL-CLI-001):
  serve, version, health --url, db, marketplace, nodes, jobs, fabric, attest,
  score, weights, sim {seed, run-scenario, doctor}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import typer

from hypercluster import __version__
from hypercluster.cli_common import (
    DEFAULT_BASE_URL,
    require_mutate_auth,
    resolve_base_url,
)
from hypercluster.cli_ops import (
    db_app,
    jobs_app,
    marketplace_app,
    register_nodes_mutate,
    register_serve,
)

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
    help="Provider node register / heartbeat / fabric-scan / probe-gpu / evidence.",
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
attest_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="dstack TEE offline verify and compose-hash helpers.",
)
score_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Score recompute and per-hotkey show (VAL-SCORE-019).",
)
weights_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Raw weight preview and push (VAL-SCORE-020). Never set_weights.",
)
# Root groups (VAL-CLI-001 / packaging)
register_serve(app)
app.add_typer(db_app, name="db")
app.add_typer(marketplace_app, name="marketplace")
app.add_typer(jobs_app, name="jobs")
app.add_typer(sim_app, name="sim")
app.add_typer(nodes_app, name="nodes")
app.add_typer(fabric_app, name="fabric")
app.add_typer(attest_app, name="attest")
app.add_typer(score_app, name="score")
app.add_typer(weights_app, name="weights")
fabric_app.add_typer(fabric_report_app, name="report")
register_nodes_mutate(nodes_app)


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

    return resolve_base_url(
        url,
        host,
        port,
        enforce_mission_port=enforce_mission_port,
    )


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
            f"warning: package {__version__} != live challenge_version {challenge_version}",
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
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Skip identity gates; check inventory/launcher/tee fixtures only",
    ),
) -> None:
    """CI readiness: identity gates + inventory/launcher/tee fixtures (VAL-CLI-014)."""

    from hypercluster.sim.doctor import run_doctor

    if offline and url is None and host is None and port is None:
        report = run_doctor(base_url=None, require_identity=False)
    else:
        base = _resolve_base_url(url, host, port)
        report = run_doctor(base, require_identity=not offline)
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
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Deterministic inventory seed (default: HYPER_SIM_SEED or 0)",
    ),
) -> None:
    """Seed synthetic IB/NVLink multi-node inventory (VAL-CLI-013 / VAL-FAB-019).

    Same ``--seed`` or ``HYPER_SIM_SEED`` always yields identical node IDs and
    digests (deterministic fixtures; never random UUIDs).
    """

    from hypercluster.sim.inventory import plan_readiness, seed_sim_inventory
    from hypercluster.sim.seed import inventory_shape_digest, resolve_sim_seed

    resolved = resolve_sim_seed(seed)
    inventory = seed_sim_inventory(
        seed=resolved,
        node_count=node_count,
        gpus_per_node=gpus_per_node,
    )
    readiness = plan_readiness(
        inventory,
        world_size=min(node_count * gpus_per_node, max(2, gpus_per_node * 2)),
        nnodes=min(node_count, 2),
        nproc_per_node=gpus_per_node,
    )
    shape = inventory_shape_digest(inventory)
    summary = {
        "nodes": len(inventory.nodes),
        "node_ids": [n.node_id for n in inventory.nodes],
        "ib_edges": len(inventory.ib_edges),
        "nvlink_edges": len(inventory.nvlink_edges),
        "graph_digest": inventory.graph_digest,
        "shape_digest": shape,
        "seed": resolved,
        "plan_ready": readiness.ok,
        "plan_nnodes_used": readiness.nnodes_used,
        "report_digests": [n.fabric_report.report_digest for n in inventory.nodes],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    typer.echo(
        f"sim seed: seed={resolved} nodes={len(inventory.nodes)} "
        f"graph_digest={inventory.graph_digest} shape_digest={shape} "
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
    """Run fabric-scan for a node and print the accepted FabricReport (VAL-FAB-018).

    Incomplete auth fails closed (VAL-CLI-021); --node-id required (VAL-CLI-026).
    """

    from hypercluster.cli_common import http_request, parse_json_response

    # Fail closed when hotkey/token incomplete — never invent quiet signer identity.
    resolved_hotkey, resolved_token = require_mutate_auth(hotkey=hotkey, token=token)
    base = _resolve_base_url(url, host, port)
    body = {"source": source, "seed": seed}
    response = http_request(
        "POST",
        f"{base}/v1/nodes/{node_id}/fabric-scan",
        json_body=body,
        signed=True,
        hotkey=resolved_hotkey,
        token=resolved_token,
        expect_statuses={200},
    )
    typer.echo(f"{response.status_code} {response.text}")
    payload = parse_json_response(response)
    digest = payload.get("report_digest") if isinstance(payload, dict) else None
    if not digest:
        typer.echo("fabric-scan response missing report_digest", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"accepted report_digest={digest} node_id={payload.get('node_id')}")
    raise typer.Exit(code=0)


def _load_plan_spec(spec: Path) -> dict[str, Any]:
    """Load fabric placement-spec JSON/YAML for ``fabric plan --spec``."""

    text = spec.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"error: cannot parse fabric plan --spec {spec}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    if not isinstance(loaded, dict):
        typer.echo("error: fabric plan --spec must be a JSON/YAML object", err=True)
        raise typer.Exit(code=2)
    return loaded


@fabric_app.command("plan")
def fabric_plan_cmd(
    spec: Path | None = typer.Option(
        None,
        "--spec",
        help="Path to placement-spec JSON/YAML (world_size/nnodes/policy/fabric/seed)",
        exists=False,
        dir_okay=False,
    ),
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
    """Topology plan dry-run (VAL-FAB-016 / VAL-CLI-010). Never advances job to running."""

    from hypercluster.fabric.planner import PlacementRequest, place_ranks
    from hypercluster.sim.inventory import seed_sim_inventory

    _ = dry_run  # dry-run is always true for this command surface

    # --spec wins defaults; explicit flags that follow still override.
    if spec is not None:
        if not spec.exists():
            typer.echo(f"error: fabric plan spec not found: {spec}", err=True)
            raise typer.Exit(code=2)
        loaded = _load_plan_spec(spec)
        world_size = int(loaded.get("world_size", world_size))
        nnodes = int(loaded.get("nnodes", nnodes))
        nproc_per_node = int(loaded.get("nproc_per_node", nproc_per_node))
        policy = str(loaded.get("policy") or loaded.get("placement_policy") or policy)
        fabric = str(
            loaded.get("fabric") or loaded.get("fabric_mode") or fabric
        )
        seed = int(loaded.get("seed", seed))
        if job_id is None and loaded.get("job_id"):
            job_id = str(loaded["job_id"])

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
        "spec": str(spec) if spec is not None else None,
    }
    typer.echo(json.dumps(out, indent=2, sort_keys=True))
    if not plan.ok:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


def _fabric_launch_allowed() -> bool:
    """True only when explicit dev gate env is set (VAL-CLI-010)."""

    raw = (
        os.environ.get("HYPER_ALLOW_FABRIC_LAUNCH")
        or os.environ.get("HYPER_FABRIC_LAUNCH")
        or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}


@fabric_app.command("launch")
def fabric_launch_cmd(
    job_id: str = typer.Option(..., "--job-id", help="Job id that would be launched"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Required with HYPER_ALLOW_FABRIC_LAUNCH=1 (dev-only). No silent prod restarts.",
    ),
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
) -> None:
    """Dev-only fabric launch gate (VAL-CLI-010).

    Production settings refuse by default. Even with the env allowlist, --force
    is required so accidental restarts cannot silently re-kick prod jobs.
    """

    allowed = _fabric_launch_allowed()
    if not allowed or not force:
        reason = (
            "fabric launch denied: gated (set HYPER_ALLOW_FABRIC_LAUNCH=1 and pass --force)"
            if not allowed
            else "fabric launch denied: --force required even when HYPER_ALLOW_FABRIC_LAUNCH is set"
        )
        body = {
            "ok": False,
            "gated": True,
            "denied": True,
            "allowed_env": allowed,
            "force": force,
            "job_id": job_id,
            "message": reason,
        }
        typer.echo(json.dumps(body, indent=2, sort_keys=True))
        typer.echo(reason, err=True)
        raise typer.Exit(code=2)

    # Dev path: still non-destructive by default — dry-check job existence only.
    # Do not call sim_launch from CLI; lifecycle launch is owned by the API worker.
    base = _resolve_base_url(url, host, port)
    try:
        response = httpx.get(f"{base}/v1/jobs/{job_id}", timeout=10.0)
    except httpx.HTTPError as exc:
        typer.echo(f"fabric launch: job probe failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"{response.status_code} {response.text}")
    if response.status_code != 200:
        typer.echo(
            "fabric launch: job not found or API error; refusing to force sim launch",
            err=True,
        )
        raise typer.Exit(code=1)

    # Explicit refusal to call launcher from CLI even with force — operator must
    # re-submit/use job worker. Command exists as a fail-closed surface for gates.
    typer.echo(
        json.dumps(
            {
                "ok": False,
                "gated": False,
                "dev_allow": True,
                "job_id": job_id,
                "message": (
                    "fabric launch CLI does not invoke sim_launch; use the job "
                    "lifecycle worker. Gate opened but launch not performed."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise typer.Exit(code=2)


@attest_app.command("compose-hash")
def attest_compose_hash_cmd(
    compose_file: Path = typer.Option(
        ...,
        "--compose-file",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to app-compose / compose YAML fixture to hash",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit JSON object with compose_hash + path (default: bare hash line)",
    ),
    check_golden: Path | None = typer.Option(
        None,
        "--check-golden",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional golden sha256 file; exit 1 on drift",
    ),
) -> None:
    """Compute deterministic sha256 of a compose file (VAL-TEE-010).

    Offline only — no network, no docker. Two runs over a fixed file yield
    the same ``sha256:<hex>`` string. Optional ``--check-golden`` asserts
    equality against a committed golden hash under tests/fixtures/tee.
    """

    from hypercluster.attest.compose_hash import (
        hash_compose_file,
        load_golden_hash_file,
    )

    try:
        compose_hash = hash_compose_file(compose_file)
    except OSError as exc:
        typer.echo(f"compose-hash failed to read {compose_file}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if check_golden is not None:
        try:
            expected = load_golden_hash_file(check_golden)
        except (OSError, ValueError) as exc:
            typer.echo(f"compose-hash golden load failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if compose_hash != expected:
            typer.echo(
                f"compose-hash drift: got={compose_hash} expected={expected}",
                err=True,
            )
            raise typer.Exit(code=1)

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "compose_hash": compose_hash,
                    "compose_file": str(compose_file),
                    "check_golden": str(check_golden) if check_golden else None,
                    "ok": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        typer.echo(compose_hash)
    raise typer.Exit(code=0)


@attest_app.command("verify-live")
def attest_verify_live_cmd(
    endpoint: str | None = typer.Option(
        None,
        "--endpoint",
        help="Live dstack-verifier URL (unused unless HYPER_TEE_LIVE=1)",
    ),
    quote_fixture: Path | None = typer.Option(
        None,
        "--quote-fixture",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional offline fixture to re-check under live mode",
    ),
) -> None:
    """Live TEE verify path — skip-safe when HYPER_TEE_LIVE is unset (VAL-TEE-014).

    Default CI must not require a live endpoint. When the env flag is unset we
    exit non-zero with an explicit skip message (handlable, never a traceback).
    """

    import os

    live = (os.environ.get("HYPER_TEE_LIVE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not live:
        msg = {
            "ok": False,
            "skipped": True,
            "reason": "live_skipped",
            "message": "HYPER_TEE_LIVE unset; live verify path skipped (VAL-TEE-014)",
            "endpoint": endpoint,
        }
        typer.echo(json.dumps(msg, indent=2, sort_keys=True))
        raise typer.Exit(code=2)

    # Live is enabled but not fully wired — fail closed with a clear code, and
    # never invent a success spoof.
    from hypercluster.attest.models import TeeVerifyRequest
    from hypercluster.attest.verify import verify_tee

    quote_b64 = "bGl2ZS1wbGFjZWhvbGRlcg=="  # base64 "live-placeholder"
    if quote_fixture is not None:
        from hypercluster.attest.offline_fixtures import load_quote_fixture, package_quote_b64

        env = load_quote_fixture(quote_fixture)
        quote_b64 = package_quote_b64(env)

    result = verify_tee(
        TeeVerifyRequest(
            quote_b64=quote_b64,
            report_data_expected=b"\x00" * 64,
            mode="live",
        )
    )
    body = result.to_public()
    body["endpoint"] = endpoint
    typer.echo(json.dumps(body, indent=2, sort_keys=True))
    raise typer.Exit(code=0 if result.is_valid else 1)


@attest_app.command("verify-offline")
def attest_verify_offline_cmd(
    quote_fixture: Path = typer.Option(
        ...,
        "--quote-fixture",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to offline TEE quote fixture JSON",
    ),
    job_id: str | None = typer.Option(None, "--job-id", help="Expected job_id for binding"),
    image_digest: str | None = typer.Option(
        None, "--image-digest", help="Expected image digest for binding"
    ),
    nonce: str | None = typer.Option(None, "--nonce", help="Expected nonce for binding"),
    tcb_enforce: bool | None = typer.Option(
        None,
        "--tcb-enforce/--no-tcb-enforce",
        help="Override HYPER_TEE_TCB_ENFORCE (default from settings)",
    ),
) -> None:
    """Verify an offline TEE quote fixture (VAL-TEE-001..). Exit 0 iff is_valid."""

    from hypercluster.attest.offline_fixtures import load_quote_fixture
    from hypercluster.attest.policy import TeeVerifyPolicy, default_policy_from_settings
    from hypercluster.attest.report_data import build_report_data, parse_report_data
    from hypercluster.attest.verify import verify_offline_fixture_file

    base_policy = default_policy_from_settings()
    policy = (
        TeeVerifyPolicy(
            compose_allowlist=base_policy.compose_allowlist,
            tcb_enforce=bool(tcb_enforce) if tcb_enforce is not None else base_policy.tcb_enforce,
            acceptable_tcb_statuses=base_policy.acceptable_tcb_statuses,
            disallowed_advisory_ids=base_policy.disallowed_advisory_ids,
        )
        if tcb_enforce is not None
        else base_policy
    )

    report_data_expected: bytes | None = None
    if job_id and image_digest and nonce:
        report_data_expected = build_report_data(
            job_id=job_id, image_digest=image_digest, nonce=nonce
        )
    else:
        env = load_quote_fixture(quote_fixture)
        try:
            report_data_expected = parse_report_data(env.report_data_hex).raw
        except Exception:
            report_data_expected = None

    result = verify_offline_fixture_file(
        quote_fixture,
        policy=policy,
        report_data_expected=report_data_expected,
        job_id=job_id,
        image_digest=image_digest,
        nonce=nonce,
    )
    typer.echo(json.dumps(result.to_public(), indent=2, sort_keys=True))
    raise typer.Exit(code=0 if result.is_valid else 1)


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


def _shared_token_opt() -> str:
    return os.environ.get("CHALLENGE_SHARED_TOKEN") or os.environ.get("HYPER_SHARED_TOKEN") or ""


@score_app.command("show")
def score_show_cmd(
    hotkey: str = typer.Option(..., "--hotkey", help="Miner hotkey (ss58)"),
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    limit: int = typer.Option(50, "--limit", help="Max history rows"),
) -> None:
    """Print factor/composite history for a hotkey (VAL-SCORE-019). Never prints tokens."""

    base = _resolve_base_url(url, host, port)
    try:
        response = httpx.get(
            f"{base}/v1/scores/{hotkey}",
            params={"limit": limit},
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        typer.echo(f"score show failed for {base}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(response.text)
    if response.status_code != 200:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@score_app.command("recompute")
def score_recompute_cmd(
    epoch: int | None = typer.Option(
        None, "--epoch", help="Optional epoch bucket for weight snapshot"
    ),
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    master: str | None = typer.Option(
        None,
        "--master",
        help="Master base URL for optional push (env HYPER_MASTER_BASE_URL)",
    ),
) -> None:
    """Refresh aggregates and optionally build a pending weight snapshot (VAL-SCORE-019).

    Uses challenge local DB via process settings when available; prefers live
    API weight-preview for remote base URL mode. Never prints secrets.
    """

    import asyncio

    from hypercluster.db.database import Database
    from hypercluster.settings import get_hyper_settings, get_settings
    from hypercluster.weight_push import (
        WeightPushValidationError,
        create_pending_snapshot,
    )
    from hypercluster.weights import load_raw_weights, weight_preview_payload

    base = _resolve_base_url(url, host, port)
    # Prefer live API preview for remote confirmation, then refresh via DB.
    try:
        preview = httpx.get(f"{base}/v1/weight-preview", timeout=10.0)
        if preview.status_code == 200:
            typer.echo(f"live weight-preview: {preview.text}")
    except httpx.HTTPError as exc:
        typer.echo(f"weight-preview probe warning: {exc}", err=True)

    settings = get_settings()
    hyper = get_hyper_settings()
    database = Database(settings.database_url)

    async def _run() -> dict[str, Any]:
        await database.init()
        try:
            weights = await load_raw_weights(database=database, hyper=hyper)
            body = await weight_preview_payload(database=database, hyper=hyper)
            body["recomputed"] = True
            body["weights"] = weights
            if epoch is not None and weights:
                try:
                    async with database.session() as session:
                        snap = await create_pending_snapshot(
                            session,
                            challenge_slug=settings.slug,
                            epoch=int(epoch),
                            weights=weights,
                            hyper=hyper,
                        )
                    body["snapshot"] = {
                        "epoch": snap.epoch,
                        "revision": snap.revision,
                        "push_status": snap.push_status,
                        "payload_digest": snap.payload_digest,
                    }
                except WeightPushValidationError as exc:
                    body["snapshot_error"] = {"code": exc.code, "detail": exc.message}
            return body
        finally:
            await database.close()

    result = asyncio.run(_run())
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    raise typer.Exit(code=0)


@weights_app.command("preview")
def weights_preview_cmd(
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
) -> None:
    """Print pending/latest raw weight map (VAL-SCORE-020/028). Never prints tokens."""

    base = _resolve_base_url(url, host, port)
    try:
        response = httpx.get(f"{base}/v1/weight-preview", timeout=10.0)
    except httpx.HTTPError as exc:
        typer.echo(f"weights preview failed for {base}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    # Redact auth tokens if present by never echoing env secrets.
    typer.echo(response.text)
    if response.status_code != 200:
        raise typer.Exit(code=1)
    try:
        body = response.json()
    except ValueError:
        raise typer.Exit(code=1) from None
    weights = body.get("weights") or {}
    if not isinstance(weights, dict):
        typer.echo("weights preview shape invalid", err=True)
        raise typer.Exit(code=1)
    for key, val in weights.items():
        try:
            fval = float(val)
        except (TypeError, ValueError):
            typer.echo(f"non-numeric weight for {key}", err=True)
            raise typer.Exit(code=1) from None
        if fval < 0 or fval != fval:  # noqa: PLR0124 — NaN check
            typer.echo(f"illegal weight for {key}: {val}", err=True)
            raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@weights_app.command("push")
def weights_push_cmd(
    epoch: int | None = typer.Option(None, "--epoch", help="Epoch bucket"),
    revision: int | None = typer.Option(None, "--revision", help="Monochronic revision"),
    master: str | None = typer.Option(
        None,
        "--master",
        help="Master base URL (default HYPER_MASTER_BASE_URL or http://127.0.0.1:3201)",
    ),
    url: str | None = _url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Challenge shared token (or CHALLENGE_SHARED_TOKEN). Never printed.",
    ),
    expires_at: str | None = typer.Option(
        None,
        "--expires-at",
        help="Optional ISO expires_at override (for inverted/expired rejection tests)",
    ),
    computed_at: str | None = typer.Option(
        None,
        "--computed-at",
        help="Optional ISO computed_at override",
    ),
) -> None:
    """Authenticated raw-weight push to master/mock-master (VAL-SCORE-015/020/030).

    Never calls on-chain set_weights. Full token is never printed.
    """

    import asyncio
    from datetime import datetime

    from hypercluster.db.database import Database
    from hypercluster.settings import get_hyper_settings, get_settings
    from hypercluster.weight_push import WeightPushClient
    from hypercluster.weights import load_raw_weights

    settings = get_settings()
    hyper = get_hyper_settings()
    resolved_token = token or _shared_token_opt() or (settings.shared_token or "")
    if not resolved_token:
        typer.echo(
            "weights push requires --token or CHALLENGE_SHARED_TOKEN",
            err=True,
        )
        raise typer.Exit(code=1)
    # Never echo the full token; only a redacted fingerprint.
    typer.echo(f"auth=token_set len={len(resolved_token)} redacted=***")

    master_url = (
        master
        or os.environ.get("HYPER_MASTER_BASE_URL")
        or getattr(hyper, "master_base_url", None)
        or "http://127.0.0.1:3201"
    )
    database = Database(settings.database_url)

    force_computed = None
    force_expires = None
    if computed_at:
        force_computed = datetime.fromisoformat(computed_at.replace("Z", "+00:00"))
    if expires_at:
        force_expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))

    async def _run() -> dict[str, Any]:
        await database.init()
        try:
            weights = await load_raw_weights(database=database, hyper=hyper)
            client = WeightPushClient(
                database=database,
                challenge_slug=settings.slug,
                master_base_url=str(master_url),
                shared_token=resolved_token,
                hyper=hyper,
            )
            # Never pass secrets into result dump.
            result = await client.push_once(
                weights=weights if weights else None,
                epoch=epoch,
                revision=revision,
                force_computed_at=force_computed,
                force_expires_at=force_expires,
            )
            return {
                "status": result.status,
                "epoch": result.epoch,
                "revision": result.revision,
                "payload_digest": result.payload_digest,
                "snapshot_id": result.snapshot_id,
                "local_id": result.local_id,
                "push_status": result.push_status,
                "idempotent": result.idempotent,
                "error": result.error,
                "master": str(master_url),
            }
        finally:
            await database.close()

    body = asyncio.run(_run())
    typer.echo(json.dumps(body, indent=2, sort_keys=True))
    # Fail closed for illegal windows / empty / rejections so scripts exit non-zero.
    if body.get("status") in {
        "invalid_window",
        "inverted_window",
        "expired_window",
        "empty_weights",
        "rejected",
        "transport_error",
        "server_error",
        "ack_mismatch",
        "malformed_ack",
    }:
        raise typer.Exit(code=1)
    if body.get("status") not in {"acknowledged", "skipped_empty"}:
        # soft ok for idempotent already-acked is included in acknowledged
        if body.get("push_status") not in {"acked", "sim"}:
            raise typer.Exit(code=1)
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
