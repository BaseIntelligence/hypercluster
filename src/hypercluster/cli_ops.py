"""Core CLI ops: serve, db, marketplace, jobs, and node register/heartbeat.

Registered onto the root Typer app from ``hypercluster.cli`` (VAL-CLI-001/004/020/021).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import typer

from hypercluster.cli_common import (
    emit,
    http_request,
    json_option,
    parse_json_response,
    require_mutate_auth,
    resolve_base_url,
    resolve_hotkey,
    resolve_shared_token,
    url_option,
)

db_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="SQLite database init/migrate under CHALLENGE_DATABASE_URL (/data).",
)
marketplace_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Home-grown marketplace: offers, rent, leases.",
)
marketplace_offers_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Browse and filter marketplace offers.",
)
marketplace_offer_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Create offers of capacity.",
)
marketplace_lease_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Inspect leases.",
)
jobs_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="HyperJob submit / status / list / cancel / logs.",
)

marketplace_app.add_typer(marketplace_offers_app, name="offers")
marketplace_app.add_typer(marketplace_offer_app, name="offer")
marketplace_app.add_typer(marketplace_lease_app, name="lease")


def register_serve(app: typer.Typer) -> None:
    """Attach ``serve`` to the root app."""

    @app.command("serve")
    def serve_cmd(
        host: str | None = typer.Option(
            None,
            "--host",
            help="Bind host (default CHALLENGE_HOST or 127.0.0.1 for local dev)",
        ),
        port: int | None = typer.Option(
            None,
            "--port",
            help="Bind port (default CHALLENGE_PORT or mission 3200)",
        ),
        reload: bool = typer.Option(
            False,
            "--reload/--no-reload",
            help="Uvicorn reload (dev only)",
        ),
    ) -> None:
        """Start the challenge API (dev uvicorn). Binds configured port (VAL-CLI-020)."""

        import uvicorn

        from hypercluster.settings import get_settings
        from hypercluster.sim.ports import DEFAULT_BAREMETAL_PORT

        settings = get_settings()
        # Bare-metal dev prefers 3200 when settings keep the Base image default 8000
        # and the operator did not set CHALLENGE_PORT.
        env_port = os.environ.get("CHALLENGE_PORT")
        env_host = os.environ.get("CHALLENGE_HOST")
        bind_host = host or env_host or "127.0.0.1"
        if port is not None:
            bind_port = int(port)
        elif env_port:
            bind_port = int(env_port)
        elif int(getattr(settings, "port", 8000) or 8000) == 8000:
            bind_port = DEFAULT_BAREMETAL_PORT
        else:
            bind_port = int(settings.port)

        typer.echo(f"serving hypercluster on http://{bind_host}:{bind_port}")
        # Module path supports lifespan + create_challenge_app secrets at boot.
        uvicorn.run(
            "hypercluster.app:app",
            host=bind_host,
            port=bind_port,
            reload=reload,
            log_level="info",
        )


@db_app.command("init")
def db_init_cmd(
    database_url: str | None = typer.Option(
        None,
        "--database-url",
        help="Override CHALLENGE_DATABASE_URL (default sqlite under /data)",
    ),
) -> None:
    """Create schema tables under /data (or URL). Idempotent; does not wipe rows."""

    from hypercluster.db.database import Database
    from hypercluster.settings import DEFAULT_DATABASE_URL, get_settings

    url = (database_url or "").strip() or None
    if url is None:
        try:
            url = get_settings().database_url
        except Exception:  # noqa: BLE001 — missing secrets should still allow path default
            url = os.environ.get("CHALLENGE_DATABASE_URL") or DEFAULT_DATABASE_URL

    async def _run() -> None:
        database = Database(url)
        await database.init()
        await database.close()

    asyncio.run(_run())
    typer.echo(f"db init ok url={url}")


@db_app.command("migrate")
def db_migrate_cmd(
    database_url: str | None = typer.Option(
        None,
        "--database-url",
        help="Override CHALLENGE_DATABASE_URL",
    ),
) -> None:
    """Alias for schema ensure (create_all). Idempotent migrate shim for v1."""

    db_init_cmd(database_url=database_url)


@marketplace_offers_app.command("list")
def marketplace_offers_list_cmd(
    gpu_model: str | None = typer.Option(None, "--gpu-model", help="Filter by GPU model"),
    require_ib: bool = typer.Option(
        False,
        "--require-ib",
        help="Only offers requiring / providing IB",
    ),
    tee: str | None = typer.Option(None, "--tee", help="Filter tee tier"),
    url: str | None = url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    as_json: bool = json_option(),
) -> None:
    """List marketplace offers (VAL-CLI-005). Non-zero when API unreachable."""

    base = resolve_base_url(url, host, port)
    params: dict[str, Any] = {}
    if gpu_model:
        params["gpu_model"] = gpu_model
    if require_ib:
        params["require_ib"] = "true"
    if tee:
        params["tee"] = tee
    response = http_request(
        "GET",
        f"{base}/v1/offers",
        params=params or None,
        expect_statuses={200},
    )
    payload = parse_json_response(response)
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if as_json:
        emit(items if isinstance(items, list) else payload, as_json=True)
    else:
        count = len(items) if isinstance(items, list) else "?"
        emit(payload, as_json=False, human_line=f"offers={count}")
    raise typer.Exit(code=0)


@marketplace_offer_app.command("create")
def marketplace_offer_create_cmd(
    node_ids: str = typer.Option(
        ...,
        "--node-ids",
        help="Comma-separated node ids to advertise",
    ),
    price: float = typer.Option(..., "--price", help="price_per_hour (>0)"),
    lifetime: float = typer.Option(..., "--lifetime", help="max_lifetime_hours (>0)"),
    mode: str = typer.Option("single", "--mode", help="single|cluster"),
    require_ib: bool = typer.Option(False, "--require-ib", help="Require IB on nodes"),
    gpu_model: str | None = typer.Option(None, "--gpu-model"),
    gpu_count: int | None = typer.Option(None, "--gpu-count"),
    tee: str = typer.Option("none", "--tee"),
    create_hotkey: str | None = typer.Option(None, "--hotkey", help="Provider hotkey"),
    token: str | None = typer.Option(None, "--token", help="Challenge shared token"),
    url: str | None = url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    as_json: bool = json_option(),
) -> None:
    """Create a listed offer (signed). Incomplete auth fails closed."""

    hotkey, secret = require_mutate_auth(hotkey=create_hotkey, token=token)
    base = resolve_base_url(url, host, port)
    body: dict[str, Any] = {
        "node_ids": [part.strip() for part in node_ids.split(",") if part.strip()],
        "price_per_hour": float(price),
        "max_lifetime_hours": float(lifetime),
        "mode": mode,
        "require_ib": bool(require_ib),
        "tee": tee,
    }
    if gpu_model:
        body["gpu_model"] = gpu_model
    if gpu_count is not None:
        body["gpu_count"] = int(gpu_count)
    response = http_request(
        "POST",
        f"{base}/v1/offers",
        json_body=body,
        signed=True,
        hotkey=hotkey,
        token=secret,
        expect_statuses={200, 201},
    )
    payload = parse_json_response(response)
    emit(payload, as_json=as_json, human_line=f"offer_id={payload.get('id')}")
    raise typer.Exit(code=0)


@marketplace_app.command("rent")
def marketplace_rent_cmd(
    offer_id: str = typer.Option(..., "--offer-id", help="Listed offer id"),
    max_hours: float | None = typer.Option(None, "--max-hours", help="lifetime_hours bound"),
    max_price: float | None = typer.Option(None, "--max-price", help="renter price cap"),
    rent_hotkey: str | None = typer.Option(None, "--hotkey", help="Renter hotkey"),
    token: str | None = typer.Option(None, "--token", help="Challenge shared token"),
    url: str | None = url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    as_json: bool = json_option(),
) -> None:
    """Rent listed offer → lease + pod (signed)."""

    hotkey, secret = require_mutate_auth(hotkey=rent_hotkey, token=token)
    base = resolve_base_url(url, host, port)
    body: dict[str, Any] = {}
    if max_hours is not None:
        body["lifetime_hours"] = float(max_hours)
    if max_price is not None:
        body["max_price"] = float(max_price)
    response = http_request(
        "POST",
        f"{base}/v1/offers/{offer_id}/rent",
        json_body=body or {},
        signed=True,
        hotkey=hotkey,
        token=secret,
        expect_statuses={200, 201},
    )
    payload = parse_json_response(response)
    lease = payload.get("lease") if isinstance(payload, dict) else None
    lease_id = lease.get("id") if isinstance(lease, dict) else None
    emit(payload, as_json=as_json, human_line=f"lease_id={lease_id}")
    raise typer.Exit(code=0)


@marketplace_lease_app.command("show")
def marketplace_lease_show_cmd(
    lease_id: str = typer.Option(..., "--id", help="Lease id"),
    url: str | None = url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    as_json: bool = json_option(),
) -> None:
    """Show lease detail."""

    base = resolve_base_url(url, host, port)
    response = http_request(
        "GET",
        f"{base}/v1/leases/{lease_id}",
        expect_statuses={200},
    )
    payload = parse_json_response(response)
    emit(payload, as_json=as_json)
    raise typer.Exit(code=0)


@marketplace_app.command("terminate")
def marketplace_terminate_cmd(
    lease_id: str = typer.Option(..., "--lease-id", help="Active lease id"),
    reason: str = typer.Option("cli_terminate", "--reason"),
    term_hotkey: str | None = typer.Option(None, "--hotkey", help="Renter/provider hotkey"),
    token: str | None = typer.Option(None, "--token", help="Challenge shared token"),
    url: str | None = url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    as_json: bool = json_option(),
) -> None:
    """Terminate lease (signed)."""

    hotkey, secret = require_mutate_auth(hotkey=term_hotkey, token=token)
    base = resolve_base_url(url, host, port)
    response = http_request(
        "POST",
        f"{base}/v1/leases/{lease_id}/terminate",
        json_body={"reason": reason},
        signed=True,
        hotkey=hotkey,
        token=secret,
        expect_statuses={200},
    )
    payload = parse_json_response(response)
    emit(payload, as_json=as_json, human_line=f"terminated lease_id={lease_id}")
    raise typer.Exit(code=0)


def register_nodes_mutate(nodes_app: typer.Typer) -> None:
    """Attach register / heartbeat that fail closed without auth."""

    @nodes_app.command("register")
    def nodes_register_cmd(
        ssh: str | None = typer.Option(None, "--ssh", help="ssh_endpoint host:port"),
        gpus: int = typer.Option(..., "--gpus", help="gpu_count"),
        gpu_model: str = typer.Option("sim-gpu", "--gpu-model"),
        hostname: str | None = typer.Option(None, "--hostname"),
        tee: str = typer.Option("none", "--tee", help="tee_capability"),
        ib: bool = typer.Option(False, "--ib", help="Mark inventory IB-capable"),
        reg_hotkey: str | None = typer.Option(None, "--hotkey"),
        token: str | None = typer.Option(None, "--token"),
        url: str | None = url_option(),
        host: str | None = typer.Option(None, help="API host when --url omitted"),
        port: int | None = typer.Option(None, help="API port when --url omitted"),
        as_json: bool = json_option(),
    ) -> None:
        """Register a provider node (signed). Incomplete auth rejected."""

        hotkey, secret = require_mutate_auth(hotkey=reg_hotkey, token=token)
        base = resolve_base_url(url, host, port)
        inventory: dict[str, Any] = {}
        if ib:
            inventory["ib_devices"] = [{"name": "mlx5_0", "port": 1, "rate_gbps": 100.0}]
            inventory["ib_rate_gbps"] = 100.0
        body: dict[str, Any] = {
            "gpu_model": gpu_model,
            "gpu_count": int(gpus),
            "tee_capability": tee,
        }
        if ssh:
            body["ssh_endpoint"] = ssh
        if hostname:
            body["hostname"] = hostname
        if inventory:
            body["inventory"] = inventory
        response = http_request(
            "POST",
            f"{base}/v1/nodes",
            json_body=body,
            signed=True,
            hotkey=hotkey,
            token=secret,
            expect_statuses={200, 201},
        )
        payload = parse_json_response(response)
        emit(payload, as_json=as_json, human_line=f"node_id={payload.get('id')}")
        raise typer.Exit(code=0)

    @nodes_app.command("heartbeat")
    def nodes_heartbeat_cmd(
        node_id: str | None = typer.Option(None, "--node-id", help="Optional single node"),
        hb_hotkey: str | None = typer.Option(None, "--hotkey"),
        token: str | None = typer.Option(None, "--token"),
        url: str | None = url_option(),
        host: str | None = typer.Option(None, help="API host when --url omitted"),
        port: int | None = typer.Option(None, help="API port when --url omitted"),
        as_json: bool = json_option(),
    ) -> None:
        """Provider/node heartbeat (signed)."""

        hotkey, secret = require_mutate_auth(hotkey=hb_hotkey, token=token)
        base = resolve_base_url(url, host, port)
        body: dict[str, Any] = {}
        if node_id:
            body["node_id"] = node_id
        response = http_request(
            "POST",
            f"{base}/v1/nodes/heartbeat",
            json_body=body or {},
            signed=True,
            hotkey=hotkey,
            token=secret,
            expect_statuses={200},
        )
        payload = parse_json_response(response)
        emit(payload, as_json=as_json)
        raise typer.Exit(code=0)


@jobs_app.command("submit")
def jobs_submit_cmd(
    spec: Path | None = typer.Option(
        None,
        "--spec",
        help="Path to job JSON/YAML spec",
        exists=False,
        dir_okay=False,
    ),
    image_digest: str | None = typer.Option(None, "--image-digest"),
    entrypoint: str | None = typer.Option(
        None,
        "--entrypoint",
        help="Comma-separated entrypoint argv",
    ),
    world_size: int | None = typer.Option(None, "--world-size"),
    nnodes: int | None = typer.Option(None, "--nnodes"),
    nproc_per_node: int | None = typer.Option(None, "--nproc-per-node"),
    timeout_s: int | None = typer.Option(None, "--timeout-s"),
    fabric: str = typer.Option("auto", "--fabric"),
    tee: str = typer.Option("none", "--tee"),
    lease_id: str | None = typer.Option(None, "--lease-id"),
    client_request_id: str | None = typer.Option(None, "--client-request-id"),
    submit_hotkey: str | None = typer.Option(None, "--hotkey"),
    token: str | None = typer.Option(None, "--token"),
    url: str | None = url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    as_json: bool = json_option(),
) -> None:
    """Submit a HyperJob (signed). Requires --spec or direct fields + auth."""

    hotkey, secret = require_mutate_auth(hotkey=submit_hotkey, token=token)
    body: dict[str, Any] = {}
    if spec is not None:
        if not spec.exists():
            typer.echo(f"error: job spec not found: {spec}", err=True)
            raise typer.Exit(code=2)
        text = spec.read_text(encoding="utf-8")
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore

                loaded = yaml.safe_load(text)
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"error: cannot parse job spec {spec}: {exc}", err=True)
                raise typer.Exit(code=2) from exc
        if not isinstance(loaded, dict):
            typer.echo("error: job spec must be a JSON/YAML object", err=True)
            raise typer.Exit(code=2)
        body.update(loaded)
    # CLI flags override spec.
    if image_digest:
        body["image_digest"] = image_digest
    if entrypoint:
        body["entrypoint"] = [p for p in entrypoint.split(",") if p != ""]
    if world_size is not None:
        body["world_size"] = int(world_size)
    if nnodes is not None:
        body["nnodes"] = int(nnodes)
    if nproc_per_node is not None:
        body["nproc_per_node"] = int(nproc_per_node)
    if timeout_s is not None:
        body["timeout_s"] = int(timeout_s)
    if fabric:
        body.setdefault("fabric", fabric)
    if tee:
        body.setdefault("tee", tee)
    if lease_id:
        body["lease_id"] = lease_id
    if client_request_id:
        body["client_request_id"] = client_request_id
    # Defaults for automatic smoke when only --spec placeholders omitted.
    body.setdefault(
        "image_digest",
        "sha256:sim000000000000000000000000000000000000000000000000000000000001",
    )
    body.setdefault("entrypoint", ["python", "-c", "print('hypercluster')"])
    body.setdefault("world_size", 1)
    body.setdefault("nnodes", 1)
    body.setdefault("nproc_per_node", 1)
    body.setdefault("timeout_s", 60)
    body.setdefault("resource", {"gpus": 1, "nodes": 1})

    required = (
        "image_digest",
        "entrypoint",
        "world_size",
        "nnodes",
        "nproc_per_node",
        "timeout_s",
        "resource",
    )
    missing = [k for k in required if k not in body]
    if missing:
        typer.echo(
            f"error: job submit missing required fields: {', '.join(missing)} "
            "(provide --spec or flags)",
            err=True,
        )
        raise typer.Exit(code=2)

    base = resolve_base_url(url, host, port)
    response = http_request(
        "POST",
        f"{base}/v1/jobs",
        json_body=body,
        signed=True,
        hotkey=hotkey,
        token=secret,
        expect_statuses={200, 201},
    )
    payload = parse_json_response(response)
    job_id = payload.get("id") or payload.get("job_id")
    emit(payload, as_json=as_json, human_line=f"job_id={job_id}")
    raise typer.Exit(code=0)


@jobs_app.command("status")
def jobs_status_cmd(
    job_id: str = typer.Option(..., "--id", help="Job id"),
    url: str | None = url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    as_json: bool = json_option(),
) -> None:
    """Get job status. Non-zero when API down (VAL-CLI-023/026)."""

    base = resolve_base_url(url, host, port)
    response = http_request(
        "GET",
        f"{base}/v1/jobs/{job_id}",
        expect_statuses={200},
    )
    payload = parse_json_response(response)
    status = payload.get("status") if isinstance(payload, dict) else None
    emit(payload, as_json=as_json, human_line=f"job_id={job_id} status={status}")
    raise typer.Exit(code=0)


@jobs_app.command("list")
def jobs_list_cmd(
    status_filter: str | None = typer.Option(None, "--status", help="Filter status"),
    list_hotkey: str | None = typer.Option(
        None,
        "--hotkey",
        help="Submitter hotkey scope (optional; fail-closed empty without)",
    ),
    url: str | None = url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    as_json: bool = json_option(),
) -> None:
    """List submitter jobs. Connection errors exit non-zero."""

    base = resolve_base_url(url, host, port)
    params: dict[str, Any] = {}
    if status_filter:
        params["status"] = status_filter
    headers: dict[str, str] = {}
    hk = resolve_hotkey(list_hotkey, required=False)
    if hk:
        headers["X-Hotkey"] = hk
    response = http_request(
        "GET",
        f"{base}/v1/jobs",
        params=params or None,
        headers=headers or None,
        expect_statuses={200},
    )
    payload = parse_json_response(response)
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if as_json:
        emit(items if isinstance(items, list) else payload, as_json=True)
    else:
        count = len(items) if isinstance(items, list) else "?"
        emit(payload, as_json=False, human_line=f"jobs={count}")
    raise typer.Exit(code=0)


@jobs_app.command("cancel")
def jobs_cancel_cmd(
    job_id: str = typer.Option(..., "--id", help="Job id"),
    cancel_hotkey: str | None = typer.Option(None, "--hotkey"),
    token: str | None = typer.Option(None, "--token"),
    url: str | None = url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    as_json: bool = json_option(),
) -> None:
    """Cancel a non-terminal job (signed)."""

    hotkey, secret = require_mutate_auth(hotkey=cancel_hotkey, token=token)
    base = resolve_base_url(url, host, port)
    response = http_request(
        "POST",
        f"{base}/v1/jobs/{job_id}/cancel",
        json_body={},
        signed=True,
        hotkey=hotkey,
        token=secret,
        expect_statuses={200},
    )
    payload = parse_json_response(response)
    emit(payload, as_json=as_json, human_line=f"cancelled job_id={job_id}")
    raise typer.Exit(code=0)


@jobs_app.command("logs")
def jobs_logs_cmd(
    job_id: str = typer.Option(..., "--id", help="Job id"),
    attempt: int = typer.Option(1, "--attempt", help="Attempt number (1-based)"),
    url: str | None = url_option(),
    host: str | None = typer.Option(None, help="API host when --url omitted"),
    port: int | None = typer.Option(None, help="API port when --url omitted"),
    as_json: bool = json_option(),
) -> None:
    """Show attempt metrics digests / safe log excerpts (no secrets)."""

    base = resolve_base_url(url, host, port)
    response = http_request(
        "GET",
        f"{base}/v1/jobs/{job_id}/attempts/{attempt}",
        expect_statuses={200},
    )
    payload = parse_json_response(response)
    emit(payload, as_json=as_json, human_line=f"job_id={job_id} attempt={attempt}")
    raise typer.Exit(code=0)


__all__ = [
    "db_app",
    "jobs_app",
    "marketplace_app",
    "register_nodes_mutate",
    "register_serve",
    "resolve_shared_token",
]
