"""Minimal external Verda HTTP client for M8 single-GPU QA (NOT product code).

Loads credentials only from ``/root/.config/hypercluster-mission/verda.env``
(or paths passed explicitly). Never hangs secrets into product settings.
All token material stays process-local and is redacted from evidence dumps.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_ENV_PATH = Path("/root/.config/hypercluster-mission/verda.env")
DEFAULT_API_BASE = "https://api.verda.com"
# Cloudflare rejects some bot-like UAs when combined with loopback cf-connecting-ip.
DEFAULT_USER_AGENT = (
    "hypercluster-m8-ops/1.0 (+mission-qa; https://github.com/BaseIntelligence/hypercluster)"
)
# Prefer a non-loopback client IP header; 127.0.0.1 triggers CF error 1000 on this edge.
DEFAULT_CF_CONNECTING_IP = "203.0.113.10"

# Hard safety for this mission feature (serial, short, cheap).
DEFAULT_MAX_RATE_USD_PER_HR = 1.50
DEFAULT_HARD_BUDGET_USD = 5.0
DEFAULT_MAX_LIFETIME_MINUTES = 60
DEFAULT_MAX_GPUS = 1


class VerdaOpsError(RuntimeError):
    """External ops failure (auth, stock, deploy, teardown)."""


@dataclass(slots=True)
class InstanceChoice:
    """Selected on-demand instance type under budget."""

    instance_type: str
    location_code: str
    price_per_hour: float
    gpu_model: str
    gpu_count: int
    gpu_memory_gb: float | None
    cpu_cores: int | None
    mem_gb: float | None
    raw_type: dict[str, Any]


@dataclass(slots=True)
class InstanceInfo:
    """Running or known Verda instance summary (non-secret)."""

    id: str
    status: str
    ip: str | None
    instance_type: str | None
    location: str | None
    price_per_hour: float | None
    hostname: str | None
    volume_ids: list[str]
    raw: dict[str, Any]


def load_env_file(path: Path | str) -> dict[str, str]:
    """Parse KEY=VALUE file (no shell expansion). Values are not printed."""

    env_path = Path(path)
    if not env_path.is_file():
        raise VerdaOpsError(f"verda env file missing: {env_path}")
    out: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            out[key] = value
    return out


def parse_price(value: Any) -> float | None:
    """Cast catalog prices that may arrive as strings."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pick_cheapest_single_gpu(
    types: list[dict[str, Any]],
    availability: list[dict[str, Any]],
    *,
    max_rate_usd: float = DEFAULT_MAX_RATE_USD_PER_HR,
    max_gpus: int = DEFAULT_MAX_GPUS,
) -> InstanceChoice | None:
    """Pick cheapest available single-GPU under rate cap; re-query each call."""

    loc_for: dict[str, list[str]] = {}
    for loc in availability:
        code = str(loc.get("location_code") or loc.get("location") or "")
        for type_name in loc.get("availabilities") or []:
            loc_for.setdefault(str(type_name), []).append(code)

    candidates: list[InstanceChoice] = []
    for row in types:
        name = str(row.get("instance_type") or row.get("name") or "")
        if not name or name not in loc_for:
            continue
        gpu = row.get("gpu") or {}
        try:
            n_gpu = int(gpu.get("number_of_gpus") or row.get("gpu_count") or 0)
        except (TypeError, ValueError):
            n_gpu = 0
        if n_gpu != max_gpus:
            continue
        # Prefer fixed on-demand; spot may be cheaper but eviction-risky for smoke.
        price = parse_price(row.get("price_per_hour"))
        if price is None or price > max_rate_usd or price <= 0:
            continue
        model = str(row.get("model") or gpu.get("description") or "GPU")
        gmem = (row.get("gpu_memory") or {}).get("size_in_gigabytes")
        cpu = (row.get("cpu") or {}).get("number_of_cores")
        mem = (row.get("memory") or {}).get("size_in_gigabytes")
        # Prefer FIN-01 style stability order as secondary key.
        locations = sorted(loc_for[name])
        location = locations[0]
        candidates.append(
            InstanceChoice(
                instance_type=name,
                location_code=location,
                price_per_hour=float(price),
                gpu_model=model,
                gpu_count=n_gpu,
                gpu_memory_gb=float(gmem) if gmem is not None else None,
                cpu_cores=int(cpu) if cpu is not None else None,
                mem_gb=float(mem) if mem is not None else None,
                raw_type=row,
            )
        )

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c.price_per_hour, c.instance_type, c.location_code))
    return candidates[0]


def estimate_cost_usd(
    *,
    price_per_hour: float,
    start_unix: float,
    end_unix: float | None = None,
    min_billable_minutes: float = 1.0,
) -> float:
    """Upper-bound cost estimate (rate × elapsed hours, minute floor)."""

    end = end_unix if end_unix is not None else time.time()
    elapsed_s = max(0.0, end - start_unix)
    hours = max(elapsed_s / 3600.0, min_billable_minutes / 60.0)
    return round(price_per_hour * hours, 6)


def cost_within_hard_cap(cost_usd: float, hard_cap_usd: float) -> bool:
    return cost_usd <= hard_cap_usd + 1e-9


def redact_secrets(payload: Any) -> Any:
    """Recursively redact token-like fields for evidence logs."""

    secret_keys = {
        "access_token",
        "token",
        "refresh_token",
        "client_secret",
        "authorization",
        "password",
        "jupyter_token",
        "secret",
    }
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            if str(key).lower() in secret_keys:
                out[key] = "***REDACTED***"
            else:
                out[key] = redact_secrets(value)
        return out
    if isinstance(payload, list):
        return [redact_secrets(item) for item in payload]
    if isinstance(payload, str) and len(payload) > 40 and payload.count(".") >= 2:
        # JWT-ish: keep shape only.
        if payload.startswith("eyJ"):
            return "***REDACTED_JWT***"
    return payload


class VerdaClient:
    """Thin urllib client against Verda public API.

    Intentionally NOT imported by ``hypercluster`` product package.
    """

    def __init__(
        self,
        *,
        env_path: Path | str = DEFAULT_ENV_PATH,
        api_base: str | None = None,
        timeout_s: float = 60.0,
        user_agent: str = DEFAULT_USER_AGENT,
        cf_connecting_ip: str = DEFAULT_CF_CONNECTING_IP,
    ) -> None:
        file_env = load_env_file(env_path)
        self.client_id = (
            os.environ.get("VERDA_CLIENT_ID") or file_env.get("VERDA_CLIENT_ID") or ""
        ).strip()
        self.client_secret = (
            os.environ.get("VERDA_CLIENT_SECRET") or file_env.get("VERDA_CLIENT_SECRET") or ""
        ).strip()
        base = (
            api_base
            or os.environ.get("VERDA_API_BASE")
            or file_env.get("VERDA_API_BASE")
            or DEFAULT_API_BASE
        )
        self.api_base = base.rstrip("/")
        self.timeout_s = timeout_s
        self.user_agent = user_agent
        self.cf_connecting_ip = cf_connecting_ip
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        if not self.client_id or not self.client_secret:
            raise VerdaOpsError("VERDA_CLIENT_ID / VERDA_CLIENT_SECRET missing")

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        auth: bool = True,
        form: dict[str, str] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        # OpenAPI lists cf-connecting-ip as required on mutations, but sending any
        # explicit value (including public probes) triggers Cloudflare error 1000
        # on this edge. Only attach it when the caller opts in via extra_headers.
        data: bytes | None = None
        if form is not None:
            data = urlencode(form).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if auth:
            headers["Authorization"] = f"Bearer {self.access_token()}"
        if extra_headers:
            headers.update(extra_headers)
        req = Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - external QA only
                raw = resp.read()
                if not raw:
                    return None
                try:
                    return json.loads(raw.decode())
                except json.JSONDecodeError:
                    return {"raw": raw.decode(errors="replace")[:2000]}
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise VerdaOpsError(f"HTTP {exc.code} {method} {path}: {detail}") from exc
        except URLError as exc:
            raise VerdaOpsError(f"network error {method} {path}: {exc}") from exc

    def access_token(self) -> str:
        """Fetch / refresh OAuth client_credentials token (short TTL)."""

        now = time.time()
        if self._token and now < self._token_expires_at - 30:
            return self._token
        payload = self._request(
            "POST",
            "/v1/oauth2/token",
            form={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            auth=False,
        )
        if not isinstance(payload, dict) or not payload.get("access_token"):
            raise VerdaOpsError("oauth token response missing access_token")
        self._token = str(payload["access_token"])
        expires_in = float(payload.get("expires_in") or 590)
        self._token_expires_at = now + max(60.0, expires_in)
        return self._token

    def list_instance_types(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/v1/instance-types")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            items = data.get("data") or data.get("items") or data.get("instance_types")
            if isinstance(items, list):
                return items
        raise VerdaOpsError("unexpected instance-types shape")

    def list_availability(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/v1/instance-availability")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            items = data.get("data") or data.get("items")
            if isinstance(items, list):
                return items
        raise VerdaOpsError("unexpected instance-availability shape")

    def list_ssh_keys(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/v1/ssh-keys")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            items = data.get("data") or data.get("items") or data.get("keys")
            if isinstance(items, list):
                return items
        return []

    def list_instances(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/v1/instances")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            items = data.get("data") or data.get("items")
            if isinstance(items, list):
                return items
        return []

    def get_instance(self, instance_id: str) -> InstanceInfo:
        data = self._request("GET", f"/v1/instances/{instance_id}")
        if not isinstance(data, dict):
            raise VerdaOpsError(f"bad instance payload for {instance_id}")
        return _to_instance_info(data)

    def deploy_instance(
        self,
        *,
        instance_type: str,
        location_code: str,
        image: str,
        hostname: str,
        description: str,
        ssh_key_ids: list[str],
        is_spot: bool = False,
        os_volume_size_gb: int = 50,
    ) -> InstanceInfo:
        os_volume: dict[str, Any] = {
            "name": f"hc-m8-{hostname[:20]}-{uuid.uuid4().hex[:6]}",
            "size": os_volume_size_gb,
        }
        # on_spot_discontinue is rejected for non-spot contracts (HTTP 400).
        if is_spot:
            os_volume["on_spot_discontinue"] = "delete_permanently"
        body: dict[str, Any] = {
            "instance_type": instance_type,
            "image": image,
            "hostname": hostname,
            "description": description,
            "location_code": location_code,
            "ssh_key_ids": ssh_key_ids,
            "is_spot": is_spot,
            "os_volume": os_volume,
        }
        data = self._request("POST", "/v1/instances", body=body)
        # Wire responses observed:
        #   - bare uuid string
        #   - {"raw": "<uuid>"} when body is non-JSON text
        #   - full instance object /{"instance": {...}}
        instance_id: str | None = None
        if isinstance(data, str) and data.strip():
            instance_id = data.strip().strip('"')
        elif isinstance(data, dict):
            if isinstance(data.get("instance"), dict) and data["instance"].get("id"):
                return _to_instance_info(data["instance"])
            if data.get("id"):
                return _to_instance_info(data)
            raw = data.get("raw")
            if isinstance(raw, str) and raw.strip():
                # Non-JSON body (uuid text)
                candidate = raw.strip().strip('"')
                if len(candidate) >= 32:
                    instance_id = candidate
        if not instance_id:
            raise VerdaOpsError(f"deploy missing instance id: {redact_secrets(data)!r}")
        # Fetch canonical instance record.
        return self.get_instance(instance_id)

    def instance_action(
        self,
        *,
        action: str,
        instance_id: str | list[str],
        volume_ids: list[str] | None = None,
        delete_permanently: bool = True,
    ) -> Any:
        body: dict[str, Any] = {
            "action": action,
            "id": instance_id,
            "delete_permanently": delete_permanently,
        }
        if volume_ids is not None:
            body["volume_ids"] = volume_ids
        return self._request("PUT", "/v1/instances", body=body)

    def discontinue(
        self,
        instance_id: str,
        *,
        volume_ids: list[str] | None = None,
        delete_permanently: bool = True,
        wait_ready_s: float = 600.0,
    ) -> dict[str, Any]:
        """Idempotent-ish discontinue: second call returns clear non-crash status.

        Verda rejects discontinue while status is ``provisioning``; wait for a
        non-provisioning state (or terminal error) before DELETE/discontinue.
        """

        try:
            info = self.get_instance(instance_id)
        except VerdaOpsError as exc:
            msg = str(exc).lower()
            if any(t in msg for t in ("notfound", "not found", "404", "410")):
                return {
                    "ok": True,
                    "action": "discontinue",
                    "instance_id": instance_id,
                    "idempotent": True,
                    "detail": str(exc)[:500],
                }
            info = None

        if info is not None and (info.status or "").lower() == "provisioning" and wait_ready_s > 0:
            deadline = time.time() + wait_ready_s
            while time.time() < deadline:
                try:
                    info = self.get_instance(instance_id)
                except VerdaOpsError as exc:
                    msg = str(exc).lower()
                    if any(t in msg for t in ("notfound", "not found", "404", "410")):
                        return {
                            "ok": True,
                            "action": "discontinue",
                            "instance_id": instance_id,
                            "idempotent": True,
                            "detail": str(exc)[:500],
                        }
                    break
                status = (info.status or "").lower()
                if status != "provisioning":
                    break
                time.sleep(8.0)
            if info is not None:
                volume_ids = volume_ids if volume_ids is not None else list(info.volume_ids)

        # Prefer discontinue; fall back to delete if API prefers that action name.
        for action in ("discontinue", "delete"):
            try:
                result = self.instance_action(
                    action=action,
                    instance_id=instance_id,
                    volume_ids=volume_ids if volume_ids is not None else [],
                    delete_permanently=delete_permanently,
                )
                return {
                    "ok": True,
                    "action": action,
                    "instance_id": instance_id,
                    "result": redact_secrets(result),
                }
            except VerdaOpsError as exc:
                msg = str(exc).lower()
                if any(
                    token in msg
                    for token in (
                        "notfound",
                        "not found",
                        "discontinued",
                        "already",
                        "404",
                        "410",
                        "no such",
                    )
                ):
                    return {
                        "ok": True,
                        "action": action,
                        "instance_id": instance_id,
                        "idempotent": True,
                        "detail": str(exc)[:500],
                    }
                last_error = str(exc)[:500]
                # Try alternate action once.
                if action == "discontinue" and (
                    "can't discontinue" in msg or "cannot discontinue" in msg or "forbidden" in msg
                ):
                    continue
                return {
                    "ok": False,
                    "action": action,
                    "instance_id": instance_id,
                    "error": last_error,
                }
        return {
            "ok": False,
            "action": "discontinue",
            "instance_id": instance_id,
            "error": last_error if "last_error" in locals() else "unknown",
        }

    def wait_until_running(
        self,
        instance_id: str,
        *,
        timeout_s: float = 600.0,
        poll_s: float = 8.0,
    ) -> InstanceInfo:
        deadline = time.time() + timeout_s
        last: InstanceInfo | None = None
        while time.time() < deadline:
            last = self.get_instance(instance_id)
            status = (last.status or "").lower()
            if status == "running" and last.ip:
                return last
            if status in {
                "error",
                "installation_failed",
                "discontinued",
                "notfound",
                "no_capacity",
            }:
                raise VerdaOpsError(f"instance {instance_id} entered terminal status={last.status}")
            time.sleep(poll_s)
        raise VerdaOpsError(
            f"timeout waiting for instance {instance_id} running "
            f"(last_status={getattr(last, 'status', None)})"
        )


def _to_instance_info(data: dict[str, Any]) -> InstanceInfo:
    volumes = data.get("volume_ids") or []
    if not isinstance(volumes, list):
        volumes = []
    price = parse_price(data.get("price_per_hour"))
    return InstanceInfo(
        id=str(data.get("id") or ""),
        status=str(data.get("status") or "unknown"),
        ip=str(data["ip"]) if data.get("ip") else None,
        instance_type=str(data["instance_type"]) if data.get("instance_type") else None,
        location=str(data["location"]) if data.get("location") else None,
        price_per_hour=price,
        hostname=str(data["hostname"]) if data.get("hostname") else None,
        volume_ids=[str(v) for v in volumes],
        raw=data,
    )


def select_live_choice(
    client: VerdaClient,
    *,
    max_rate_usd: float = DEFAULT_MAX_RATE_USD_PER_HR,
) -> InstanceChoice:
    """Re-query catalog + availability and select cheapest single GPU under cap."""

    types = client.list_instance_types()
    avail = client.list_availability()
    choice = pick_cheapest_single_gpu(types, avail, max_rate_usd=max_rate_usd)
    if choice is None:
        raise VerdaOpsError(f"no available single-GPU under ${max_rate_usd}/hr rate cap")
    return choice
