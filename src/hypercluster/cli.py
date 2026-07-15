"""Minimal Typer CLI entry (scaffold); expanded in later milestones."""

from __future__ import annotations

import typer

from hypercluster import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Hypercluster challenge CLI")


@app.command("version")
def version_cmd() -> None:
    """Print package version."""

    typer.echo(__version__)


@app.command("health")
def health_cmd(
    host: str = typer.Option("127.0.0.1", help="API host"),
    port: int = typer.Option(3200, help="API port (mission band 3200–3299)"),
) -> None:
    """Probe remote /health (requires a running API process)."""

    import httpx

    url = f"http://{host}:{port}/health"
    response = httpx.get(url, timeout=5.0)
    typer.echo(f"{response.status_code} {response.text}")
    raise typer.Exit(code=0 if response.status_code == 200 else 1)


if __name__ == "__main__":
    app()
