"""Shared Typer helpers: base URL, auth flags, HTTP call, JSON output mode.

Used by marketplace/nodes/jobs mutate + query surfaces so incomplete auth,
connection errors, and ``--json`` one-shot machine mode behave consistently
(VAL-CLI-021/023/024/026).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import typer

from hypercluster.api.auth import build_signed_headers
from hypercluster.sim.ports import (
    DEFAULT_BAREMETAL_PORT,
    assert_mission_port,
    is_mission_port,
    parse_port_from_url,
)

DEFAULT_BASE_URL = f"http://127.0.0.1:{DEFAULT_BAREMETAL_PORT}"

# Typer option placeholders (factories) to keep help text consistent.


def url_option() -> Any:
    return typer.Option(
        None,
        "--url",
        help=f"Live challenge base URL (default {DEFAULT_BASE_URL})",
    )


def json_option() -> Any:
    return typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON only (no rich table noise)",
    )


def resolve_base_url(
    url: str | None,
    host: str | None = None,
    port: int | None = None,
    *,
    enforce_mission_port: bool = False,
) -> str:
    """Resolve CLI base URL from --url or host/port defaults."""

    if url:
        resolved = url.rstrip("/")
        port_from_url = parse_port_from_url(resolved)
        if (
            enforce_mission_port
            and port_from_url is not None
            and not is_mission_port(port_from_url)
        ):
            assert_mission_port(port_from_url)
        return resolved

    resolved_host = host or "127.0.0.1"
    resolved_port = DEFAULT_BAREMETAL_PORT if port is None else int(port)
    if enforce_mission_port:
        assert_mission_port(resolved_port)
    return f"http://{resolved_host}:{resolved_port}"


def resolve_hotkey(
    hotkey: str | None,
    *,
    required: bool = True,
    flag_label: str = "--hotkey",
) -> str | None:
    """Resolve miner/provider hotkey from flag or env.

    When *required* and missing, exits with a clear usage error (VAL-CLI-021).
    Never invents a silent default for mutate paths — fail closed.
    """

    resolved = (
        (hotkey or "").strip()
        or (os.environ.get("HYPER_HOTKEY") or "").strip()
        or (os.environ.get("CHALLENGE_HOTKEY") or "").strip()
    )
    if resolved:
        return resolved
    if required:
        typer.echo(
            f"error: missing miner auth hotkey ({flag_label} or "
            "HYPER_HOTKEY / CHALLENGE_HOTKEY env required for mutate commands)",
            err=True,
        )
        raise typer.Exit(code=2)
    return None


def resolve_shared_token(
    token: str | None,
    *,
    required: bool = True,
) -> str | None:
    """Resolve challenge shared token for signing (not printed in full)."""

    resolved = (
        (token or "").strip()
        or (os.environ.get("CHALLENGE_SHARED_TOKEN") or "").strip()
        or (os.environ.get("HYPER_SHARED_TOKEN") or "").strip()
    )
    if resolved:
        return resolved
    if required:
        typer.echo(
            "error: missing auth token (--token or CHALLENGE_SHARED_TOKEN / "
            "HYPER_SHARED_TOKEN env required for signed mutate requests)",
            err=True,
        )
        raise typer.Exit(code=2)
    return None


def require_mutate_auth(
    *,
    hotkey: str | None,
    token: str | None,
) -> tuple[str, str]:
    """Return (hotkey, token) or exit with clear incomplete-auth error."""

    hk = resolve_hotkey(hotkey, required=True)
    tok = resolve_shared_token(token, required=True)
    assert hk is not None and tok is not None
    return hk, tok


def emit(payload: Any, *, as_json: bool = False, human_line: str | None = None) -> None:
    """Print resulting payload.

    When *as_json* is True, stdout is exactly one JSON document (VAL-CLI-024).
    Rich/table chatter must not interleave with machine mode.
    """

    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if human_line is not None:
        typer.echo(human_line)
    if isinstance(payload, (dict, list)):
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    elif payload is not None:
        typer.echo(str(payload))


def http_request(
    method: str,
    url: str,
    *,
    json_body: dict[str, Any] | list[Any] | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 10.0,
    signed: bool = False,
    hotkey: str | None = None,
    token: str | None = None,
    expect_statuses: set[int] | None = None,
) -> httpx.Response:
    """Perform an HTTP request; fail closed on connection / 5xx (VAL-CLI-023).

    *signed* attaches X-Hotkey/X-Signature/X-Nonce/X-Timestamp using the
    challenge shared token (HMAC-dev). Auth material must already be present —
    caller is responsible for require_mutate_auth on write paths.
    """

    raw = b""
    req_headers: dict[str, str] = dict(headers or {})
    content: bytes | None = None
    if json_body is not None:
        raw = json.dumps(json_body).encode()
        content = raw
        req_headers.setdefault("Content-Type", "application/json")
    if signed:
        if not hotkey or not token:
            typer.echo(
                "error: signed request missing hotkey/token (incomplete auth flags)",
                err=True,
            )
            raise typer.Exit(code=2)
        signed_headers = build_signed_headers(secret=token, hotkey=hotkey, body=raw)
        req_headers.update(signed_headers)

    try:
        response = httpx.request(
            method.upper(),
            url,
            content=content,
            headers=req_headers,
            params=params,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        typer.echo(f"connection error for {url}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if response.status_code >= 500:
        typer.echo(
            f"API server error {response.status_code} for {url}: {response.text[:500]}",
            err=True,
        )
        raise typer.Exit(code=1)

    if expect_statuses is not None and response.status_code not in expect_statuses:
        typer.echo(
            f"API error {response.status_code} for {url}: {response.text[:500]}",
            err=True,
        )
        raise typer.Exit(code=1)

    return response


def parse_json_response(response: httpx.Response) -> Any:
    """Parse JSON body or exit non-zero with clear error."""

    try:
        return response.json()
    except ValueError as exc:
        typer.echo(f"API returned non-JSON body: {response.text[:300]!r}", err=True)
        raise typer.Exit(code=1) from exc


__all__ = [
    "DEFAULT_BASE_URL",
    "emit",
    "http_request",
    "json_option",
    "parse_json_response",
    "require_mutate_auth",
    "resolve_base_url",
    "resolve_hotkey",
    "resolve_shared_token",
    "url_option",
]
