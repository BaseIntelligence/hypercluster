"""Hypercluster Typer CLI — scaffold health/version + sim doctor/smoke.

Architecture outline (expanded in later milestones):
  serve, version, health --url, marketplace, nodes, jobs, fabric, attest,
  score, weights, sim {seed, run-scenario, doctor}
"""

from __future__ import annotations

import json
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
app.add_typer(sim_app, name="sim")


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
def sim_seed_cmd() -> None:
    """Placeholder seed command (deterministic fixtures land later milestones)."""

    typer.echo("sim seed: stub ok (no-op in M1 scaffold)")
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
