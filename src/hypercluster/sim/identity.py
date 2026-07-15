"""Identity probes for local sim doctor / smoke (health + ready gates).

VAL-SCAF-036 requires smoke/doctor to fail when /health or /ready is not green.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(slots=True)
class IdentityReport:
    """Result of probing challenge identity surfaces."""

    base_url: str
    ok: bool
    health_status: str | None = None
    ready: bool | None = None
    ready_http_status: int | None = None
    health_http_status: int | None = None
    slug: str | None = None
    role: str | None = None
    version: str | None = None
    errors: list[str] = field(default_factory=list)
    health_body: dict[str, Any] | None = None
    ready_body: dict[str, Any] | None = None

    def summary_lines(self) -> list[str]:
        lines = [
            f"base_url={self.base_url}",
            f"ok={self.ok}",
            f"health_http={self.health_http_status} status={self.health_status}",
            f"ready_http={self.ready_http_status} ready={self.ready}",
        ]
        if self.slug is not None:
            lines.append(f"slug={self.slug}")
        if self.version is not None:
            lines.append(f"version={self.version}")
        for err in self.errors:
            lines.append(f"error={err}")
        return lines


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def probe_identity_gates(
    base_url: str,
    *,
    timeout: float = 5.0,
) -> IdentityReport:
    """Probe ``/health`` and ``/ready`` against a live base URL.

    Pass criteria (green):
    - ``GET /health`` returns 200, JSON with ``status == "ok"``,
      ``slug == "hypercluster"``, and ``ready is True``
    - ``GET /ready`` returns 200 with ``ready is True``

    Any network error, non-JSON body, wrong slug, or not-ready condition fails.
    """

    normalized = _normalize_base_url(base_url)
    errors: list[str] = []
    health_status: str | None = None
    health_http: int | None = None
    ready_flag: bool | None = None
    ready_http: int | None = None
    slug: str | None = None
    role: str | None = None
    version: str | None = None
    health_body: dict[str, Any] | None = None
    ready_body: dict[str, Any] | None = None

    try:
        with httpx.Client(timeout=timeout) as client:
            try:
                health_resp = client.get(f"{normalized}/health")
                health_http = health_resp.status_code
                try:
                    health_body = health_resp.json()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"health non-JSON: {exc}")
                    health_body = None
                if health_resp.status_code != 200:
                    errors.append(f"health HTTP {health_resp.status_code}")
                if isinstance(health_body, dict):
                    health_status = str(health_body.get("status", ""))
                    slug = health_body.get("slug") if isinstance(
                        health_body.get("slug"), str
                    ) else None
                    role = health_body.get("role") if isinstance(
                        health_body.get("role"), str
                    ) else None
                    version = (
                        str(health_body["version"])
                        if health_body.get("version") is not None
                        else None
                    )
                    if health_status != "ok":
                        errors.append(f"health status={health_status!r} not ok")
                    if slug != "hypercluster":
                        errors.append(f"health slug={slug!r} expected hypercluster")
                    if role is not None and role != "challenge":
                        errors.append(f"health role={role!r} expected challenge")
                    if health_body.get("ready") is not True:
                        errors.append("health ready is not true")
                else:
                    errors.append("health body missing object")
            except httpx.HTTPError as exc:
                errors.append(f"health request failed: {exc}")

            try:
                ready_resp = client.get(f"{normalized}/ready")
                ready_http = ready_resp.status_code
                try:
                    ready_body = ready_resp.json()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"ready non-JSON: {exc}")
                    ready_body = None
                if ready_resp.status_code != 200:
                    errors.append(f"ready HTTP {ready_resp.status_code}")
                if isinstance(ready_body, dict):
                    ready_flag = bool(ready_body.get("ready"))
                    if ready_body.get("ready") is not True:
                        errors.append("ready ready flag is not true")
                else:
                    errors.append("ready body missing object")
            except httpx.HTTPError as exc:
                errors.append(f"ready request failed: {exc}")
    except httpx.HTTPError as exc:
        errors.append(f"client error: {exc}")

    ok = not errors
    return IdentityReport(
        base_url=normalized,
        ok=ok,
        health_status=health_status,
        ready=ready_flag,
        ready_http_status=ready_http,
        health_http_status=health_http,
        slug=slug,
        role=role,
        version=version,
        errors=errors,
        health_body=health_body,
        ready_body=ready_body,
    )


__all__ = ["IdentityReport", "probe_identity_gates"]
