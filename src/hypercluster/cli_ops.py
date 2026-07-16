"""Core CLI ops: serve, db, marketplace, jobs, and node register/probe.

Registered onto the root Typer app from ``hypercluster.cli`` (VAL-CLI-001/004/020/021).
Includes M9 GPU probe + evidence commands (VAL-GPU-040/041).
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

# Map design §5 probe-gpu-sim --fail CHECK_ID → FakeSsh fixture names.
_PROBE_SIM_FAIL_FIXTURES: dict[str, str] = {
    "ssh_connect": "ssh_timeout",
    "nvidia_smi_list": "no_gpu",
    "gpu_count": "no_gpu",
    "gpu_model_match": "wrong_model",
    "gpu_uuid_valid": "no_gpu",
    "gpu_uuid_unique": "uuid_clone",
    "vram_window": "vram_lie",
    "driver_present": "no_gpu",
    "cuda_microbench": "bench_fail",
    "docker_runtime": "docker_missing",
    "fingerprint_stable": "fingerprint_churn",
    "claim_consistency": "wrong_model",
}

# M9 design §5 exit codes for nodes probe-gpu
PROBE_EXIT_PASS = 0
PROBE_EXIT_USAGE = 1
PROBE_EXIT_FAILED_CHECKS = 2
PROBE_EXIT_TRANSPORT = 3

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
        display_name: str | None = typer.Option(
            None,
            "--display-name",
            help="Provider display name (auto-registers provider when missing)",
        ),
        reg_hotkey: str | None = typer.Option(None, "--hotkey"),
        token: str | None = typer.Option(None, "--token"),
        url: str | None = url_option(),
        host: str | None = typer.Option(None, help="API host when --url omitted"),
        port: int | None = typer.Option(None, help="API port when --url omitted"),
        as_json: bool = json_option(),
    ) -> None:
        """Register a provider node (signed). Incomplete auth rejected.

        Ensures the provider hotkey is onboarded first (VAL-CLI-007 / VAL-MKT-004)
        so ``nodes register`` is a true one-shot CLI path without a separate
        provider subcommand.
        """

        hotkey, secret = require_mutate_auth(hotkey=reg_hotkey, token=token)
        base = resolve_base_url(url, host, port)
        # Auto-register provider (idempotent) so register works without prior setup.
        provider_body: dict[str, Any] = {
            "display_name": display_name or f"provider-{hotkey[:12]}",
        }
        http_request(
            "POST",
            f"{base}/v1/providers/register",
            json_body=provider_body,
            signed=True,
            hotkey=hotkey,
            token=secret,
            expect_statuses={200, 201},
        )
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

    _register_nodes_probe_commands(nodes_app)


def _parse_key_ref(raw: str | None) -> dict[str, str] | None:
    """Parse ``env:NAME`` / ``file:PATH`` / bare path into API key_ref body.

    Never accepts raw PEM (private keys forbidden on API body).
    """

    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if "BEGIN " in text.upper() and "PRIVATE KEY" in text.upper():
        typer.echo(
            "error: private key PEM not allowed on CLI; use --key-ref env:NAME or file:PATH",
            err=True,
        )
        raise typer.Exit(code=PROBE_EXIT_USAGE)
    if text.startswith("env:"):
        name = text[4:].strip()
        if not name:
            typer.echo("error: empty env key_ref name", err=True)
            raise typer.Exit(code=PROBE_EXIT_USAGE)
        return {"kind": "env", "name": name}
    if text.startswith("file:"):
        name = text[5:].strip()
        if not name:
            typer.echo("error: empty file key_ref path", err=True)
            raise typer.Exit(code=PROBE_EXIT_USAGE)
        return {"kind": "file", "name": name}
    # Bare path → file ref
    return {"kind": "file", "name": text}


def _probe_exit_code(payload: dict[str, Any], *, http_status: int) -> int:
    """Map probe API response → CLI exit code (design §5 / VAL-GPU-040).

    0 pass, 2 failed checks, 3 transport/error, 1 usage-class client errors.
    """

    if http_status in {400, 401, 403, 404, 422}:
        return PROBE_EXIT_USAGE
    if http_status in {503}:
        return PROBE_EXIT_TRANSPORT
    if http_status >= 500:
        return PROBE_EXIT_TRANSPORT

    status_val = str(payload.get("status") or "").lower()
    if status_val == "passed":
        return PROBE_EXIT_PASS
    if status_val == "error":
        # Transport/connect errors map to exit 3; residual check failures → 2
        code = str(payload.get("failure_code") or "").lower()
        if any(
            token in code
            for token in (
                "ssh",
                "timeout",
                "transport",
                "connect",
                "unavailable",
            )
        ):
            return PROBE_EXIT_TRANSPORT
        # FakeSsh ssh_timeout often surfaces as failed/error with ssh_connect
        checks = payload.get("checks") or []
        for c in checks:
            if not isinstance(c, dict):
                continue
            if c.get("id") == "ssh_connect" and c.get("passed") is False:
                return PROBE_EXIT_TRANSPORT
        return PROBE_EXIT_FAILED_CHECKS
    if status_val == "failed":
        return PROBE_EXIT_FAILED_CHECKS
    # Unknown status with evidence still treated as fail-closed non-zero
    return PROBE_EXIT_FAILED_CHECKS


def _run_probe_gpu(
    *,
    node_id: str,
    mode: str,
    timeout_s: int | None,
    key_ref_raw: str | None,
    fixture: str | None,
    hotkey_flag: str | None,
    token_flag: str | None,
    url: str | None,
    host: str | None,
    port: int | None,
    as_json: bool,
) -> None:
    """Shared body for probe-gpu / probe-gpu-sim (signed product base-url)."""

    hotkey, secret = require_mutate_auth(hotkey=hotkey_flag, token=token_flag)
    base = resolve_base_url(url, host, port)
    body: dict[str, Any] = {
        "mode": "quick" if str(mode).lower() == "quick" else "full",
    }
    if timeout_s is not None:
        body["timeout_s"] = int(timeout_s)
    key_ref = _parse_key_ref(key_ref_raw)
    if key_ref is not None:
        body["key_ref"] = key_ref
    if fixture:
        body["fixture"] = str(fixture)

    # Probes can take longer than default marketplace calls when real SSH is used;
    # FakeSsh is still fast. Keep a higher HTTP timeout without shipping Verda.
    response = http_request(
        "POST",
        f"{base}/v1/nodes/{node_id}/probes/gpu",
        json_body=body,
        signed=True,
        hotkey=hotkey,
        token=secret,
        timeout=max(30.0, float(timeout_s or 30)),
        expect_statuses=None,  # map statuses ourselves for exit codes
    )
    # Client-visible 4xx/5xx: still try to parse JSON detail, then exit.
    try:
        payload = parse_json_response(response)
    except typer.Exit:
        code = (
            PROBE_EXIT_TRANSPORT
            if response.status_code >= 500 or response.status_code == 503
            else PROBE_EXIT_USAGE
        )
        raise typer.Exit(code=code) from None

    if not isinstance(payload, dict):
        typer.echo("error: probe response was not a JSON object", err=True)
        raise typer.Exit(code=PROBE_EXIT_USAGE)

    if response.status_code != 200:
        detail = payload.get("detail")
        err_code = None
        if isinstance(detail, dict):
            err_code = detail.get("code")
        elif isinstance(payload.get("code"), str):
            err_code = payload.get("code")
        emit(
            payload,
            as_json=as_json,
            human_line=(f"probe-gpu http={response.status_code} code={err_code}"),
        )
        raise typer.Exit(code=_probe_exit_code(payload, http_status=response.status_code))

    exit_code = _probe_exit_code(payload, http_status=200)
    evidence_id = payload.get("evidence_id") or payload.get("id")
    status_val = payload.get("status")
    emit(
        payload,
        as_json=as_json,
        human_line=(
            f"probe-gpu status={status_val} evidence_id={evidence_id} "
            f"checks_failed={payload.get('checks_failed')}"
        ),
    )
    raise typer.Exit(code=exit_code)


def _register_nodes_probe_commands(nodes_app: typer.Typer) -> None:
    """Attach probe-gpu / probe-gpu-sim / evidence (VAL-GPU-040/041)."""

    evidence_app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        help="GPU host evidence list/show/latest (API field parity).",
    )
    nodes_app.add_typer(evidence_app, name="evidence")

    @nodes_app.command("probe-gpu")
    def nodes_probe_gpu_cmd(
        node_id: str = typer.Argument(..., help="Registered node id"),
        mode: str = typer.Option("full", "--mode", help="full|quick"),
        timeout_s: int | None = typer.Option(
            None,
            "--timeout",
            help="Wall timeout seconds (mapped into probe request)",
        ),
        key_ref: str | None = typer.Option(
            None,
            "--key-ref",
            help="SSH key ref: env:NAME | file:PATH (never raw PEM)",
        ),
        fixture: str | None = typer.Option(
            None,
            "--fixture",
            help="FakeSsh fixture name override (CI only; ignored on real transport)",
        ),
        probe_hotkey: str | None = typer.Option(None, "--hotkey"),
        token: str | None = typer.Option(None, "--token"),
        url: str | None = url_option(),
        host: str | None = typer.Option(None, help="API host when --url omitted"),
        port: int | None = typer.Option(None, help="API port when --url omitted"),
        as_json: bool = json_option(),
    ) -> None:
        """Run owner-signed GPU probe via product API (VAL-GPU-040).

        Exit codes (design §5): 0 pass, 2 failed checks, 3 transport/error, 1 usage.
        Wraps ``POST /v1/nodes/{id}/probes/gpu`` only — no Verda client, no set_weights.
        """

        if not node_id.strip():
            typer.echo("error: NODE_ID is required", err=True)
            raise typer.Exit(code=PROBE_EXIT_USAGE)
        _run_probe_gpu(
            node_id=node_id.strip(),
            mode=mode,
            timeout_s=timeout_s,
            key_ref_raw=key_ref,
            fixture=fixture,
            hotkey_flag=probe_hotkey,
            token_flag=token,
            url=url,
            host=host,
            port=port,
            as_json=as_json,
        )

    @nodes_app.command("probe-gpu-sim")
    def nodes_probe_gpu_sim_cmd(
        node_id: str = typer.Argument(..., help="Registered node id"),
        pass_all: bool = typer.Option(
            False,
            "--pass-all",
            help="Run FakeSsh pass_all fixture",
        ),
        fail: str | None = typer.Option(
            None,
            "--fail",
            help="Inject fatal fail for CHECK_ID via FakeSsh fixture bank",
        ),
        mode: str = typer.Option("full", "--mode", help="full|quick"),
        sim_hotkey: str | None = typer.Option(None, "--hotkey"),
        token: str | None = typer.Option(None, "--token"),
        url: str | None = url_option(),
        host: str | None = typer.Option(None, help="API host when --url omitted"),
        port: int | None = typer.Option(None, help="API port when --url omitted"),
        as_json: bool = json_option(),
    ) -> None:
        """Ops FakeSsh helper: --pass-all or --fail CHECK_ID (VAL-GPU-040).

        Same signed product path as ``probe-gpu``; fixture forces CI outcomes.
        """

        if pass_all and fail:
            typer.echo("error: use either --pass-all or --fail, not both", err=True)
            raise typer.Exit(code=PROBE_EXIT_USAGE)
        if not pass_all and not fail:
            typer.echo(
                "error: probe-gpu-sim requires --pass-all or --fail CHECK_ID",
                err=True,
            )
            raise typer.Exit(code=PROBE_EXIT_USAGE)

        fixture: str
        if pass_all:
            fixture = "pass_all"
        else:
            assert fail is not None
            check_id = fail.strip()
            fixture = _PROBE_SIM_FAIL_FIXTURES.get(check_id, check_id)
            if check_id not in _PROBE_SIM_FAIL_FIXTURES and check_id not in {
                "pass_all",
                "no_gpu",
                "wrong_model",
                "uuid_clone",
                "vram_lie",
                "bench_fail",
                "docker_missing",
                "ssh_timeout",
                "fingerprint_churn",
            }:
                # Still allow direct fixture names; unknown ids also forwarded as fixture
                typer.echo(
                    f"probe-gpu-sim: mapping --fail {check_id!r} → fixture {fixture!r}",
                    err=True,
                )

        _run_probe_gpu(
            node_id=node_id.strip(),
            mode=mode,
            timeout_s=None,
            key_ref_raw=None,
            fixture=fixture,
            hotkey_flag=sim_hotkey,
            token_flag=token,
            url=url,
            host=host,
            port=port,
            as_json=as_json,
        )

    @evidence_app.command("list")
    def evidence_list_cmd(
        node_id: str = typer.Argument(..., help="Node id to list evidence for"),
        limit: int = typer.Option(50, "--limit", min=1, max=200),
        url: str | None = url_option(),
        host: str | None = typer.Option(None, help="API host when --url omitted"),
        port: int | None = typer.Option(None, help="API port when --url omitted"),
        as_json: bool = json_option(),
    ) -> None:
        """List GPU evidence newest-first (VAL-GPU-041; GET probes/gpu)."""

        base = resolve_base_url(url, host, port)
        response = http_request(
            "GET",
            f"{base}/v1/nodes/{node_id}/probes/gpu",
            params={"limit": int(limit)},
            expect_statuses={200},
        )
        payload = parse_json_response(response)
        # Keep API shape {items: [...]} for field parity; also accept raw list.
        if isinstance(payload, list):
            payload = {"items": payload}
        emit(
            payload,
            as_json=as_json,
            human_line=(
                f"evidence items={len((payload or {}).get('items') or [])} node_id={node_id}"
            ),
        )
        raise typer.Exit(code=0)

    @evidence_app.command("latest")
    def evidence_latest_cmd(
        node_id: str = typer.Argument(..., help="Node id"),
        url: str | None = url_option(),
        host: str | None = typer.Option(None, help="API host when --url omitted"),
        port: int | None = typer.Option(None, help="API port when --url omitted"),
        as_json: bool = json_option(),
    ) -> None:
        """Show latest GPU evidence for a node (VAL-GPU-041)."""

        base = resolve_base_url(url, host, port)
        response = http_request(
            "GET",
            f"{base}/v1/nodes/{node_id}/probes/gpu/latest",
            expect_statuses={200, 404},
        )
        if response.status_code == 404:
            typer.echo(f"error: no GPU evidence for node {node_id}", err=True)
            raise typer.Exit(code=PROBE_EXIT_USAGE)
        payload = parse_json_response(response)
        if not isinstance(payload, dict):
            typer.echo("error: latest evidence response was not a JSON object", err=True)
            raise typer.Exit(code=PROBE_EXIT_USAGE)
        evidence_id = payload.get("evidence_id") or payload.get("id")
        emit(
            payload,
            as_json=as_json,
            human_line=(f"latest evidence_id={evidence_id} status={payload.get('status')}"),
        )
        raise typer.Exit(code=0)

    @evidence_app.command("show")
    def evidence_show_cmd(
        evidence_id: str = typer.Argument(..., help="Evidence id"),
        node_id: str | None = typer.Option(
            None,
            "--node-id",
            help="Optional node scope (uses node route when set)",
        ),
        url: str | None = url_option(),
        host: str | None = typer.Option(None, help="API host when --url omitted"),
        port: int | None = typer.Option(None, help="API port when --url omitted"),
        as_json: bool = json_option(),
    ) -> None:
        """Show full evidence by id (checks + digests; VAL-GPU-041).

        Prefer global ``GET /v1/evidence/gpu/{id}``; optional ``--node-id`` uses
        node-scoped GET for the same document shape.
        """

        base = resolve_base_url(url, host, port)
        if node_id:
            path = f"{base}/v1/nodes/{node_id}/probes/gpu/{evidence_id}"
        else:
            path = f"{base}/v1/evidence/gpu/{evidence_id}"
        response = http_request(
            "GET",
            path,
            expect_statuses={200, 404},
        )
        if response.status_code == 404:
            typer.echo(f"error: evidence not found: {evidence_id}", err=True)
            raise typer.Exit(code=PROBE_EXIT_USAGE)
        payload = parse_json_response(response)
        status_val = payload.get("status") if isinstance(payload, dict) else None
        emit(
            payload,
            as_json=as_json,
            human_line=f"evidence_id={evidence_id} status={status_val}",
        )
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
    """Show attempt digests / safe log URIs without flooding secrets (VAL-CLI-009).

    Pre-collect (404 attempt_not_found) exits non-zero with a clear message —
    never dumps binary garbage or private material.
    """

    base = resolve_base_url(url, host, port)
    response = http_request(
        "GET",
        f"{base}/v1/jobs/{job_id}/attempts/{attempt}",
        # 404 => pre-collect empty / unknown attempt handled below.
        expect_statuses=None,
    )
    if response.status_code == 404:
        typer.echo(
            f"logs empty: job_id={job_id} attempt={attempt} not ready "
            f"(collect not finished or unknown job/attempt)",
            err=True,
        )
        raise typer.Exit(code=1)
    if response.status_code != 200:
        typer.echo(
            f"API error {response.status_code} for logs job_id={job_id}: {response.text[:400]}",
            err=True,
        )
        raise typer.Exit(code=1)
    payload = parse_json_response(response)
    # Redact any accidental secret-ish keys if present in metrics/contract.
    safe = _safe_logs_payload(payload if isinstance(payload, dict) else {"raw": payload})
    if as_json:
        emit(safe, as_json=True)
    else:
        digest_line = (
            f"job_id={job_id} attempt={attempt} "
            f"status={safe.get('status')} "
            f"output_digest={safe.get('output_digest')} "
            f"fabric_report_digest={safe.get('fabric_report_digest')} "
            f"launcher_log_uri={safe.get('launcher_log_uri')}"
        )
        emit(safe, as_json=False, human_line=digest_line)
    raise typer.Exit(code=0)


_SECRET_KEY_HINTS = (
    "password",
    "secret",
    "private_key",
    "ssh_private",
    "token",
    "authorization",
    "wallet",
)


def _safe_logs_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Project attempt payload to digests/excerpts; strip secret-looking keys."""

    def _scrub(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                lowered = str(key).lower()
                if any(h in lowered for h in _SECRET_KEY_HINTS):
                    out[key] = "***redacted***"
                else:
                    out[key] = _scrub(item)
            return out
        if isinstance(value, list):
            return [_scrub(item) for item in value]
        if isinstance(value, (bytes, bytearray)):
            return f"<bytes:{len(value)}>"
        if isinstance(value, str) and "\x00" in value:
            return f"<binary-text:{len(value)}>"
        return value

    preferred = {
        "id": payload.get("id"),
        "job_id": payload.get("job_id"),
        "attempt_no": payload.get("attempt_no"),
        "status": payload.get("status"),
        "launcher_log_uri": payload.get("launcher_log_uri"),
        "fabric_report_digest": payload.get("fabric_report_digest"),
        "output_digest": payload.get("output_digest"),
        "result_digest": payload.get("result_digest"),
        "failure_code": payload.get("failure_code"),
        "metrics": _scrub(payload.get("metrics")),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
    }
    # Drop Nones for a compact operator-readable view.
    return {k: v for k, v in preferred.items() if v is not None}


__all__ = [
    "PROBE_EXIT_FAILED_CHECKS",
    "PROBE_EXIT_PASS",
    "PROBE_EXIT_TRANSPORT",
    "PROBE_EXIT_USAGE",
    "db_app",
    "jobs_app",
    "marketplace_app",
    "register_nodes_mutate",
    "register_serve",
    "resolve_shared_token",
]
