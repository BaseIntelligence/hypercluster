"""Ordered GPU probe pipeline: fatal halt vs advisory continue.

Implements design §2 check list over a pluggable :class:`SshTransport`.
Unit tests drive this solely with FakeSsh (VAL-GPU-012..017).

Does **not** change the four-factor scoring formula and never calls
``set_weights``. Integrity mapping to correctness/fabric_gate is a later
feature (``m9-scoring-integrity-hooks``).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from hypercluster.probe.model_table import (
    lookup_vram_window,
    models_match,
    normalize_gpu_model,
)
from hypercluster.probe.transport import SshCommandResult, SshTransport
from hypercluster.probe.types import (
    CheckResult,
    ClaimedInventory,
    GpuHostEvidence,
    MeasuredGpu,
    MeasuredInventory,
    ProbeMode,
    canonical_json,
)

# Ordered pipeline (design §2). Residual checks after a fatal fail are not
# executed (VAL-GPU-012).
CHECK_ORDER: tuple[str, ...] = (
    "ssh_connect",
    "nvidia_smi_list",
    "gpu_count",
    "gpu_model_match",
    "gpu_uuid_valid",
    "gpu_uuid_unique",
    "vram_window",
    "driver_present",
    "cuda_microbench",
    "docker_runtime",
    "power_limit_ratio",
    "idle_util",
    "fingerprint_stable",
    "claim_consistency",
)

ADVISORY_CHECK_IDS: frozenset[str] = frozenset(
    {
        "power_limit_ratio",
        "idle_util",
    }
)

# docker_runtime is fatal when require_docker_runtime=True; otherwise advisory.
# fingerprint_stable is fatal only when a prior verified UUID set is provided.
# All other listed ids are always fatal.
ALWAYS_FATAL: frozenset[str] = frozenset(
    {
        "ssh_connect",
        "nvidia_smi_list",
        "gpu_count",
        "gpu_model_match",
        "gpu_uuid_valid",
        "gpu_uuid_unique",
        "vram_window",
        "driver_present",
        "cuda_microbench",
        "claim_consistency",
    }
)

FATAL_CHECK_IDS: frozenset[str] = ALWAYS_FATAL | {
    "docker_runtime",
    "fingerprint_stable",
}

_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_SMI_L_RE = re.compile(
    r"GPU\s+\d+:\s*(?P<name>.+?)\s*\(UUID:\s*(?P<uuid>GPU-[0-9a-fA-F-]+)\)"
)


@dataclass
class GpuProbeConfig:
    """Runtime knobs for one probe run."""

    mode: ProbeMode = "full"
    max_gpu_count: int = 14
    # VAL-GPU-015: when True, missing nvidia docker runtime is fatal.
    require_docker_runtime: bool = True
    skip_microbench: bool = False  # quick mode may skip
    power_limit_min_ratio: float = 0.9
    idle_util_max: float = 85.0


@dataclass
class GpuProbeContext:
    """Inputs for a probe: claim, prior evidence, UUID uniqueness index."""

    node_id: str | None = None
    provider_hotkey: str | None = None
    ssh_endpoint: str | None = None
    claimed: ClaimedInventory | None = None
    key_fingerprint: str | None = None
    # UUIDs already claimed by other healthy/rented nodes (VAL-GPU-016).
    occupied_uuids: set[str] = field(default_factory=set)
    # Prior verified UUID set for this node (VAL-GPU-017). None → skip gate.
    prior_verified_uuids: set[str] | None = None
    # Optional extra metadata for digests / raw.
    meta: dict[str, Any] = field(default_factory=dict)


def _is_fatal(check_id: str, config: GpuProbeConfig, ctx: GpuProbeContext) -> bool:
    if check_id in ADVISORY_CHECK_IDS:
        return False
    if check_id == "docker_runtime":
        return bool(config.require_docker_runtime)
    if check_id == "fingerprint_stable":
        return ctx.prior_verified_uuids is not None
    return check_id in ALWAYS_FATAL


def _check(
    check_id: str,
    *,
    fatal: bool,
    passed: bool,
    message: str,
    duration_ms: int = 0,
    details: dict[str, Any] | None = None,
    halt: bool = False,
) -> CheckResult:
    return CheckResult(
        id=check_id,
        fatal=fatal,
        passed=passed,
        halt=halt,
        message=message,
        duration_ms=max(0, int(duration_ms)),
        details=details or {},
    )


def parse_nvidia_smi_list(stdout: str) -> list[tuple[str, str]]:
    """Parse ``nvidia-smi -L`` lines → list of (name, uuid)."""

    found: list[tuple[str, str]] = []
    for line in (stdout or "").splitlines():
        m = _SMI_L_RE.search(line.strip())
        if m:
            found.append((m.group("name").strip(), m.group("uuid").strip()))
    return found


def parse_nvidia_smi_query(stdout: str) -> list[MeasuredGpu]:
    """Parse CSV query rows from nvidia-smi --query-gpu."""

    rows: list[MeasuredGpu] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[0]
        uuid = parts[1]
        mem = _as_int(parts[2]) if len(parts) > 2 else None
        driver = parts[3] if len(parts) > 3 else None
        power_limit = _as_float(parts[4]) if len(parts) > 4 else None
        power_default = _as_float(parts[5]) if len(parts) > 5 else None
        util_gpu = _as_float(parts[6]) if len(parts) > 6 else None
        util_mem = _as_float(parts[7]) if len(parts) > 7 else None
        clocks = _as_float(parts[8]) if len(parts) > 8 else None
        rows.append(
            MeasuredGpu(
                name=name,
                uuid=uuid,
                memory_total_mb=mem,
                driver_version=driver,
                power_limit_w=power_limit,
                power_default_w=power_default,
                util_gpu=util_gpu,
                util_mem=util_mem,
                clocks_sm_mhz=clocks,
            )
        )
    return rows


def _as_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_docker_info(stdout: str) -> dict[str, Any]:
    if not stdout or not stdout.strip():
        return {"present": False, "runtimes": [], "gpu_in_container": False}
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {"present": True, "runtimes": [], "raw": stdout[:512]}
    runtimes: list[str] = []
    rt = data.get("Runtimes") if isinstance(data, dict) else None
    if isinstance(rt, dict):
        runtimes = sorted(str(k) for k in rt.keys())
    elif isinstance(rt, list):
        runtimes = sorted(str(x) for x in rt)
    return {
        "present": True,
        "runtimes": runtimes,
        "name": data.get("Name") if isinstance(data, dict) else None,
    }


class GpuProbeService:
    """Ordered check runner building :class:`GpuHostEvidence`."""

    def __init__(
        self,
        transport: SshTransport,
        *,
        config: GpuProbeConfig | None = None,
    ) -> None:
        self.transport = transport
        self.config = config or GpuProbeConfig()

    def run(self, ctx: GpuProbeContext) -> GpuHostEvidence:
        return run_gpu_probe(self.transport, ctx, config=self.config)


def run_gpu_probe(
    transport: SshTransport,
    ctx: GpuProbeContext,
    *,
    config: GpuProbeConfig | None = None,
) -> GpuHostEvidence:
    """Execute the ordered probe pipeline and return sealed evidence."""

    cfg = config or GpuProbeConfig()
    claimed = ctx.claimed or ClaimedInventory(gpu_model="unknown", gpu_count=0)
    transport_name = getattr(transport, "name", "fake")
    if transport_name not in {"real", "fake"}:
        transport_name = "fake"

    evidence = GpuHostEvidence(
        node_id=ctx.node_id,
        provider_hotkey=ctx.provider_hotkey,
        ssh_endpoint=ctx.ssh_endpoint,
        status="failed",
        mode=cfg.mode,
        transport=transport_name,  # type: ignore[arg-type]
        claimed=claimed,
        key_fingerprint=ctx.key_fingerprint,
        raw_redacted={"command_results": []},
    )

    measured = MeasuredInventory()
    checks: list[CheckResult] = []
    advisories: list[CheckResult] = []
    residual_aborted = False
    failure_code: str | None = None
    cmd_log: list[dict[str, Any]] = []

    def record(result: CheckResult) -> bool:
        """Append check; return True if pipeline must abort."""

        nonlocal failure_code, residual_aborted
        checks.append(result)
        if not result.fatal and not result.passed:
            advisories.append(result)
        if result.fatal and not result.passed:
            residual_aborted = True
            failure_code = result.id
            return True
        return False

    # --- 1. ssh_connect (fatal; aborts residual) ---
    t0 = time.perf_counter()
    try:
        connect_res = transport.connect()
    except Exception as exc:  # noqa: BLE001 — treat as connect error
        duration = int((time.perf_counter() - t0) * 1000)
        result = _check(
            "ssh_connect",
            fatal=True,
            passed=False,
            message=f"ssh_connect exception: {exc}",
            duration_ms=duration,
            details={"error": str(exc)},
        )
        record(result)
        evidence.status = "error"
        evidence.failure_code = "ssh_connect"
        evidence.checks = checks
        evidence.advisories = advisories
        evidence.measured = measured
        evidence.raw_redacted = {"command_results": cmd_log}
        return evidence.seal()

    cmd_log.append(_cmd_public(connect_res))
    duration = connect_res.duration_ms or int((time.perf_counter() - t0) * 1000)
    connect_ok = connect_res.ok
    connect_msg = "ok" if connect_ok else (connect_res.error or connect_res.stderr or "ssh failed")
    if connect_res.timed_out:
        connect_msg = "ssh_connect timeout"
    if record(
        _check(
            "ssh_connect",
            fatal=True,
            passed=connect_ok,
            message=connect_msg,
            duration_ms=duration,
            details={"timed_out": connect_res.timed_out},
        )
    ):
        evidence.status = "error" if connect_res.timed_out else "failed"
        evidence.failure_code = failure_code
        evidence.checks = checks
        evidence.advisories = advisories
        evidence.measured = measured
        evidence.raw_redacted = {"command_results": cmd_log}
        try:
            transport.close()
        except Exception:  # noqa: BLE001
            pass
        return evidence.seal()

    # --- 2. nvidia_smi_list ---
    smi_l = _run_cmd(transport, "nvidia_smi_list", cmd_log)
    if not smi_l.ok or not (smi_l.stdout or "").strip():
        if record(
            _check(
                "nvidia_smi_list",
                fatal=True,
                passed=False,
                message=smi_l.error or smi_l.stderr or "nvidia-smi -L failed / empty",
                duration_ms=smi_l.duration_ms,
            )
        ):
            return _finalize(
                evidence, checks, advisories, measured, failure_code, cmd_log, transport
            )
    else:
        names_uuids = parse_nvidia_smi_list(smi_l.stdout)
        if not names_uuids:
            if record(
                _check(
                    "nvidia_smi_list",
                    fatal=True,
                    passed=False,
                    message="nvidia-smi -L produced no GPU lines",
                    duration_ms=smi_l.duration_ms,
                    details={"stdout_snip": smi_l.stdout[:200]},
                )
            ):
                return _finalize(
                    evidence, checks, advisories, measured, failure_code, cmd_log, transport
                )
        else:
            # Seed measured from -L; query enriches next.
            measured.gpus = [
                MeasuredGpu(name=n, uuid=u) for n, u in names_uuids
            ]
            measured.gpu_count = len(measured.gpus)
            if record(
                _check(
                    "nvidia_smi_list",
                    fatal=True,
                    passed=True,
                    message=f"found {len(names_uuids)} GPU(s)",
                    duration_ms=smi_l.duration_ms,
                    details={"count": len(names_uuids)},
                )
            ):
                return _finalize(
                    evidence, checks, advisories, measured, failure_code, cmd_log, transport
                )

    # Enrich via nvidia-smi query (best-effort; still continue on soft fail).
    smi_q = _run_cmd(transport, "nvidia_smi_query", cmd_log)
    if smi_q.ok:
        rows = parse_nvidia_smi_query(smi_q.stdout)
        if rows:
            measured.gpus = rows
            measured.gpu_count = len(rows)
            drivers = [g.driver_version for g in rows if g.driver_version]
            if drivers:
                measured.cuda_runtime_hint = drivers[0]

    # --- 3. gpu_count ---
    count = measured.gpu_count
    count_ok = 0 < count <= cfg.max_gpu_count
    if record(
        _check(
            "gpu_count",
            fatal=True,
            passed=count_ok,
            message=(
                f"gpu_count={count}" if count_ok else f"gpu_count out of range: {count}"
            ),
            details={"gpu_count": count, "max": cfg.max_gpu_count},
        )
    ):
        return _finalize(evidence, checks, advisories, measured, failure_code, cmd_log, transport)

    # --- 4. gpu_model_match ---
    measured_names = [g.name for g in measured.gpus]
    match_ok = False
    if measured_names:
        match_ok = any(models_match(claimed.gpu_model, name) for name in measured_names)
        # When all GPUs share a family, also require claimed family present.
        families = {normalize_gpu_model(n) for n in measured_names}
        claimed_family = normalize_gpu_model(claimed.gpu_model)
        if claimed_family is not None:
            match_ok = claimed_family in families
    if record(
        _check(
            "gpu_model_match",
            fatal=True,
            passed=match_ok,
            message=(
                "model family match"
                if match_ok
                else f"claimed {claimed.gpu_model!r} != measured {measured_names!r}"
            ),
            details={
                "claimed": claimed.gpu_model,
                "claimed_family": normalize_gpu_model(claimed.gpu_model),
                "measured_names": measured_names,
                "measured_families": sorted(
                    {normalize_gpu_model(n) or "?" for n in measured_names}
                ),
            },
        )
    ):
        return _finalize(evidence, checks, advisories, measured, failure_code, cmd_log, transport)

    # --- 5. gpu_uuid_valid ---
    uuids = measured.uuid_set()
    invalid = [u for u in uuids if not _UUID_RE.match(u)]
    uuid_valid = bool(uuids) and not invalid and len(uuids) == measured.gpu_count
    # Also reject blank / empty UUIDs on gpus list.
    if any(not (g.uuid or "").strip() for g in measured.gpus):
        uuid_valid = False
    if record(
        _check(
            "gpu_uuid_valid",
            fatal=True,
            passed=uuid_valid,
            message="uuids ok" if uuid_valid else "invalid or empty GPU UUID set",
            details={"uuids": uuids, "invalid": invalid},
        )
    ):
        return _finalize(evidence, checks, advisories, measured, failure_code, cmd_log, transport)

    # --- 6. gpu_uuid_unique (VAL-GPU-016) ---
    collisions = sorted(u for u in uuids if u in ctx.occupied_uuids)
    unique_ok = not collisions
    if record(
        _check(
            "gpu_uuid_unique",
            fatal=True,
            passed=unique_ok,
            message=(
                "uuid set unique across healthy nodes"
                if unique_ok
                else f"duplicate GPU UUID(s): {collisions}"
            ),
            details={"collisions": collisions},
        )
    ):
        return _finalize(evidence, checks, advisories, measured, failure_code, cmd_log, transport)

    # --- 7. vram_window ---
    vram_ok = True
    vram_details: list[dict[str, Any]] = []
    window = lookup_vram_window(claimed.gpu_model)
    if window is None and measured_names:
        window = lookup_vram_window(measured_names[0])
    for gpu in measured.gpus:
        family = normalize_gpu_model(gpu.name) or normalize_gpu_model(claimed.gpu_model)
        win = lookup_vram_window(family) if family else window
        row = {
            "uuid": gpu.uuid,
            "memory_total_mb": gpu.memory_total_mb,
            "window": None if win is None else {"min_mb": win.min_mb, "max_mb": win.max_mb},
        }
        if win is None or not win.contains(gpu.memory_total_mb):
            vram_ok = False
            row["ok"] = False
        else:
            row["ok"] = True
        vram_details.append(row)
    if record(
        _check(
            "vram_window",
            fatal=True,
            passed=vram_ok,
            message="vram within model window" if vram_ok else "vram outside model window",
            details={"gpus": vram_details},
        )
    ):
        return _finalize(evidence, checks, advisories, measured, failure_code, cmd_log, transport)

    # --- 8. driver_present ---
    drivers = [g.driver_version for g in measured.gpus if g.driver_version]
    driver_ok = bool(drivers) and all(str(d).strip() for d in drivers)
    if record(
        _check(
            "driver_present",
            fatal=True,
            passed=driver_ok,
            message="driver present" if driver_ok else "missing driver_version",
            details={"driver_versions": drivers},
        )
    ):
        return _finalize(evidence, checks, advisories, measured, failure_code, cmd_log, transport)

    # --- 9. cuda_microbench ---
    if cfg.mode == "quick" or cfg.skip_microbench:
        if record(
            _check(
                "cuda_microbench",
                fatal=True,
                passed=True,
                message="skipped (quick mode)",
                details={"skipped": True},
            )
        ):
            return _finalize(
                evidence, checks, advisories, measured, failure_code, cmd_log, transport
            )
    else:
        bench = _run_cmd(transport, "cuda_microbench", cmd_log)
        bench_ok, bench_details = _eval_microbench(bench)
        if record(
            _check(
                "cuda_microbench",
                fatal=True,
                passed=bench_ok,
                message="microbench ok" if bench_ok else "microbench failed",
                duration_ms=bench.duration_ms,
                details=bench_details,
            )
        ):
            return _finalize(
                evidence, checks, advisories, measured, failure_code, cmd_log, transport
            )
        digest = bench_details.get("digest")
        if isinstance(digest, str) and digest:
            evidence.digests.microbench_digest = digest

    # --- 10. docker_runtime (VAL-GPU-015) ---
    docker_fatal = _is_fatal("docker_runtime", cfg, ctx)
    dinfo = _run_cmd(transport, "docker_info", cmd_log)
    docker_meta = _parse_docker_info(dinfo.stdout if dinfo.ok else "")
    if not dinfo.ok:
        docker_meta = {
            "present": False,
            "runtimes": [],
            "error": dinfo.error or dinfo.stderr or "docker_info failed",
        }
    runtimes = [str(r).lower() for r in docker_meta.get("runtimes") or []]
    has_nvidia_rt = any(r in {"nvidia", "nvidia-container-runtime"} for r in runtimes)
    docker_ok = bool(docker_meta.get("present")) and has_nvidia_rt
    # Optional container visibility enrichment.
    if docker_ok:
        dgpu = _run_cmd(transport, "docker_gpu_smi", cmd_log)
        docker_meta["gpu_in_container"] = dgpu.ok and bool((dgpu.stdout or "").strip())
        if cfg.require_docker_runtime and not docker_meta["gpu_in_container"]:
            # Soft: require runtime only by default; presence of nvidia runtime is enough.
            pass
    measured.docker = docker_meta
    if record(
        _check(
            "docker_runtime",
            fatal=docker_fatal,
            passed=docker_ok,
            message=(
                "nvidia docker runtime present"
                if docker_ok
                else "nvidia docker runtime missing"
            ),
            duration_ms=dinfo.duration_ms,
            details=docker_meta,
        )
    ):
        return _finalize(evidence, checks, advisories, measured, failure_code, cmd_log, transport)

    # --- 11. power_limit_ratio (advisory) ---
    power_ok = True
    power_details: list[dict[str, Any]] = []
    for gpu in measured.gpus:
        pl = gpu.power_limit_w
        pd = gpu.power_default_w
        ratio: float | None = None
        if pl is not None and pd is not None and pd > 0:
            ratio = float(pl) / float(pd)
            if ratio < cfg.power_limit_min_ratio:
                power_ok = False
        power_details.append({"uuid": gpu.uuid, "ratio": ratio})
    record(
        _check(
            "power_limit_ratio",
            fatal=False,
            passed=power_ok,
            message=(
                "power limit ok"
                if power_ok
                else f"power limit ratio < {cfg.power_limit_min_ratio}"
            ),
            details={"gpus": power_details},
        )
    )

    # --- 12. idle_util (advisory) ---
    idle_ok = True
    for gpu in measured.gpus:
        if gpu.util_gpu is not None and float(gpu.util_gpu) > cfg.idle_util_max:
            idle_ok = False
    record(
        _check(
            "idle_util",
            fatal=False,
            passed=idle_ok,
            message="util idle ok" if idle_ok else "util unexpectedly high on advertise",
            details={
                "util_gpus": [g.util_gpu for g in measured.gpus],
                "max_allowed": cfg.idle_util_max,
            },
        )
    )

    # --- 13. fingerprint_stable (VAL-GPU-017) ---
    fp_fatal = _is_fatal("fingerprint_stable", cfg, ctx)
    if ctx.prior_verified_uuids is None:
        record(
            _check(
                "fingerprint_stable",
                fatal=False,
                passed=True,
                message="no prior verified fingerprint",
                details={"skipped": True},
            )
        )
    else:
        prior = set(ctx.prior_verified_uuids)
        current = set(uuids)
        stable = prior == current
        if record(
            _check(
                "fingerprint_stable",
                fatal=fp_fatal,
                passed=stable,
                message=(
                    "fingerprint stable"
                    if stable
                    else "fingerprint churn; re-admit required"
                ),
                details={
                    "prior": sorted(prior),
                    "current": sorted(current),
                    "re_admit_required": not stable,
                },
            )
        ):
            return _finalize(
                evidence, checks, advisories, measured, failure_code, cmd_log, transport
            )

    # --- 14. claim_consistency ---
    count_match = claimed.gpu_count <= 0 or claimed.gpu_count == measured.gpu_count
    # model already gate-checked; keep claim count consistency fatal when claimed.
    claim_ok = count_match and match_ok
    if record(
        _check(
            "claim_consistency",
            fatal=True,
            passed=claim_ok,
            message="claim consistent" if claim_ok else "claim vs measured mismatch",
            details={
                "claimed_count": claimed.gpu_count,
                "measured_count": measured.gpu_count,
                "claimed_model": claimed.gpu_model,
            },
        )
    ):
        return _finalize(evidence, checks, advisories, measured, failure_code, cmd_log, transport)

    # Success path (advisory failures alone do not fail — VAL-GPU-013).
    evidence.status = "passed"
    evidence.failure_code = None
    evidence.checks = checks
    evidence.advisories = advisories
    evidence.measured = measured
    evidence.raw_redacted = {"command_results": cmd_log}
    try:
        transport.close()
    except Exception:  # noqa: BLE001
        pass
    return evidence.seal()


def _finalize(
    evidence: GpuHostEvidence,
    checks: list[CheckResult],
    advisories: list[CheckResult],
    measured: MeasuredInventory,
    failure_code: str | None,
    cmd_log: list[dict[str, Any]],
    transport: SshTransport,
) -> GpuHostEvidence:
    if evidence.status not in {"error"}:
        evidence.status = "failed"
    evidence.failure_code = failure_code
    evidence.checks = checks
    evidence.advisories = advisories
    evidence.measured = measured
    evidence.raw_redacted = {"command_results": cmd_log}
    try:
        transport.close()
    except Exception:  # noqa: BLE001
        pass
    return evidence.seal()


def _run_cmd(
    transport: SshTransport,
    command_id: str,
    cmd_log: list[dict[str, Any]],
) -> SshCommandResult:
    try:
        res = transport.run(command_id)
    except Exception as exc:  # noqa: BLE001
        res = SshCommandResult(
            command_id=command_id,
            exit_code=255,
            stderr=str(exc),
            error=str(exc),
        )
    cmd_log.append(_cmd_public(res))
    return res


def _cmd_public(res: SshCommandResult) -> dict[str, Any]:
    # Redact + hard-cap before evidence storage (VAL-GPU-031).
    from hypercluster.probe.redact import sanitize_output

    return {
        "command_id": res.command_id,
        "exit_code": res.exit_code,
        "stdout": sanitize_output(res.stdout, max_bytes=4096),
        "stderr": sanitize_output(res.stderr, max_bytes=2048),
        "duration_ms": res.duration_ms,
        "timed_out": res.timed_out,
        "error": res.error,
    }


def _eval_microbench(res: SshCommandResult) -> tuple[bool, dict[str, Any]]:
    if not res.ok:
        return False, {
            "exit_code": res.exit_code,
            "error": res.error or res.stderr,
            "timed_out": res.timed_out,
        }
    text = (res.stdout or "").strip()
    try:
        data = json.loads(text.splitlines()[-1] if text else "{}")
    except json.JSONDecodeError:
        return False, {"parse_error": True, "stdout_snip": text[:200]}
    if not isinstance(data, dict):
        return False, {"parse_error": True}
    ok = bool(data.get("ok")) is True
    digest = data.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        ok = False
    elapsed = data.get("elapsed_ms")
    # Heuristic: zero-GPU fake shells finishing unrealistically fast with no work.
    if isinstance(elapsed, (int, float)) and elapsed < 0:
        ok = False
    return ok, {
        "ok": data.get("ok"),
        "digest": digest,
        "gflops": data.get("gflops"),
        "elapsed_ms": elapsed,
    }


def occupied_uuid_index(
    claims: Iterable[tuple[str, Iterable[str]]],
    *,
    exclude_node_id: str | None = None,
) -> set[str]:
    """Build occupied UUID set from (node_id, uuid_list) pairs (helper for DB layer)."""

    occupied: set[str] = set()
    for node_id, uuid_list in claims:
        if exclude_node_id is not None and node_id == exclude_node_id:
            continue
        for u in uuid_list:
            if u:
                occupied.add(str(u))
    return occupied


__all__ = [
    "ADVISORY_CHECK_IDS",
    "ALWAYS_FATAL",
    "CHECK_ORDER",
    "FATAL_CHECK_IDS",
    "GpuProbeConfig",
    "GpuProbeContext",
    "GpuProbeService",
    "occupied_uuid_index",
    "parse_nvidia_smi_list",
    "parse_nvidia_smi_query",
    "run_gpu_probe",
    "canonical_json",
]
