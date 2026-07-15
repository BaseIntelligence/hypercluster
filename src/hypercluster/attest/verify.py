"""TEE verify pipeline (architecture §9.2) — offline_fixture primary path.

Pipeline stages (all reason_codes collected, fail-closed conjunction):
1. Decode quote envelope (offline JSON layout; junk → quote invalid)
2. Parse quote report_data layout (truncated/extra → layout codes)
3. Bind report_data_expected vs quote report_data (nonce / job_digest)
4. Compose hash vs expected + allowlist
5. TCB / advisory policy
6. Event-log / os image flags (fixture-level offline)
"""

from __future__ import annotations

from pathlib import Path

from hypercluster.attest.models import TeeVerifyRequest, TeeVerifyResult
from hypercluster.attest.offline_fixtures import (
    OfflineQuoteEnvelope,
    load_quote_fixture,
    package_quote_b64,
    unpack_quote_b64,
)
from hypercluster.attest.policy import TeeVerifyPolicy, default_policy_from_settings
from hypercluster.attest.report_data import (
    ReportDataLayoutError,
    parse_report_data,
)


def _unique_codes(codes: list[str]) -> list[str]:
    return list(dict.fromkeys(c for c in codes if c))


def verify_tee(
    request: TeeVerifyRequest,
    *,
    policy: TeeVerifyPolicy | None = None,
) -> TeeVerifyResult:
    """Verify a TeeVerifyRequest under the given policy.

    ``offline_fixture`` mode is fully in-process and never dials the network.
    ``live`` is not implemented here (skip/unavailable — later slice).
    ``sim`` accepts only envelopes marked quote_sig_ok with sim tee_type if
    policy allowlist passes; otherwise same pipeline as offline.
    """

    pol = policy if policy is not None else default_policy_from_settings()
    mode = request.mode
    reasons: list[str] = []

    if mode == "live":
        # Live path owned by later M5 feature / optional HYPER_TEE_LIVE.
        return TeeVerifyResult(
            is_valid=False,
            quote_verified=False,
            tcb_status="unknown",
            reason_codes=["live_not_available"],
            verify_mode="live",
        )

    # 1. Decode offline envelope
    env: OfflineQuoteEnvelope | None = None
    try:
        env = unpack_quote_b64(request.quote_b64)
    except (ValueError, Exception) as exc:  # noqa: BLE001
        reasons.append("quote_invalid")
        reasons.append("quote_decode_failed")
        return TeeVerifyResult(
            is_valid=False,
            quote_verified=False,
            reason_codes=_unique_codes(reasons + [f"detail:{type(exc).__name__}"]),
            verify_mode=mode,
        )

    compose_hash = env.compose_hash
    tcb_status = env.tcb_status or "unknown"
    advisory_ids = list(env.advisory_ids or [])

    quote_ok = bool(env.quote_sig_ok)
    if not quote_ok:
        reasons.append("quote_sig_invalid")

    # 2. Parse quote-carried report_data layout
    quote_parsed = None
    try:
        quote_parsed = parse_report_data(env.report_data_hex)
    except ReportDataLayoutError as exc:
        msg = str(exc).lower()
        reasons.append("report_data_layout")
        reasons.append("report_data_invalid")
        if "truncated" in msg:
            reasons.append("report_data_truncated")
        if "extra" in msg:
            reasons.append("report_data_extra")
        quote_ok = False

    # 3. Expected report_data layout + bind
    expected_parsed = None
    try:
        expected_parsed = parse_report_data(request.report_data_expected)
    except ReportDataLayoutError as exc:
        msg = str(exc).lower()
        reasons.append("report_data_layout")
        reasons.append("report_data_invalid")
        if "truncated" in msg:
            reasons.append("report_data_truncated")
        if "extra" in msg:
            reasons.append("report_data_extra")
        quote_ok = False

    if quote_parsed is not None and expected_parsed is not None:
        if quote_parsed.raw != expected_parsed.raw:
            reasons.append("report_data_mismatch")
            reasons.append("binding_mismatch")
            if quote_parsed.job_digest != expected_parsed.job_digest:
                reasons.append("job_digest_mismatch")
                reasons.append("cross_job_binding")
            if quote_parsed.nonce_digest != expected_parsed.nonce_digest:
                reasons.append("nonce_mismatch")
                reasons.append("stale_nonce")

    # 4. Compose hash consistency vs expected_compose_hash + allowlist
    expected_compose = env.expected_compose_hash
    if expected_compose is not None and expected_compose != compose_hash:
        reasons.append("compose_hash_mismatch")
        reasons.append("measurement_mismatch")

    if not pol.is_compose_allowed(compose_hash):
        reasons.append("compose_not_allowlisted")
        reasons.append("allowlist_miss")
        reasons.append("measurement_not_allowlisted")

    # 5. TCB / advisory policy
    tcb_fail = False
    if pol.tcb_enforce:
        if tcb_status not in pol.acceptable_tcb_statuses:
            reasons.append("tcb_status_reject")
            reasons.append("tcb_fail")
            tcb_fail = True
        disallowed_hit = [a for a in advisory_ids if a in pol.disallowed_advisory_ids]
        if disallowed_hit:
            reasons.append("advisory_disallowed")
            reasons.append("tcb_fail")
            tcb_fail = True
    else:
        # Soft path: document advisories / non-UpToDate without failing.
        if tcb_status not in pol.acceptable_tcb_statuses or any(
            a in pol.disallowed_advisory_ids for a in advisory_ids
        ):
            reasons.append("tcb_advisory_soft")

    # 6. Offline event log / os image flags
    event_log_ok = bool(env.event_log_ok)
    if not event_log_ok:
        reasons.append("event_log_invalid")
    os_ok = bool(env.os_image_hash) if env.os_image_hash else False
    if request.event_log is not None and env.event_log is not None:
        if request.event_log != env.event_log:
            reasons.append("event_log_mismatch")
            event_log_ok = False

    # Conjunctive validity: any hard reason fails closed (soft tags excluded).
    soft_only = frozenset({"tcb_advisory_soft"})
    hard = [c for c in reasons if c not in soft_only and not c.startswith("detail:")]
    is_valid = bool(quote_ok) and not hard and not tcb_fail
    if is_valid:
        # Keep only soft documentation codes on success.
        reasons = [c for c in reasons if c in soft_only]
    else:
        reasons = [c for c in reasons if not c.startswith("detail:")]

    return TeeVerifyResult(
        is_valid=is_valid,
        quote_verified=bool(quote_ok and is_valid),
        event_log_verified=bool(event_log_ok and is_valid),
        os_image_hash_verified=bool(os_ok and is_valid),
        tcb_status=tcb_status,
        advisory_ids=advisory_ids,
        compose_hash=compose_hash,
        reason_codes=_unique_codes(reasons),
        verify_mode=mode,
    )


def verify_offline_fixture_file(
    path: str | Path,
    *,
    policy: TeeVerifyPolicy | None = None,
    report_data_expected: bytes | None = None,
    job_id: str | None = None,
    image_digest: str | None = None,
    nonce: str | None = None,
) -> TeeVerifyResult:
    """Load a JSON fixture file and verify under offline_fixture mode."""

    from hypercluster.attest.report_data import build_report_data

    env = load_quote_fixture(path)
    if report_data_expected is None:
        if job_id and image_digest and nonce:
            report_data_expected = build_report_data(
                job_id=job_id, image_digest=image_digest, nonce=nonce
            )
        else:
            try:
                report_data_expected = parse_report_data(env.report_data_hex).raw
            except ReportDataLayoutError:
                # Intentionally garbage so layout reject path still runs.
                report_data_expected = b"\x00" * 64

    req = TeeVerifyRequest(
        quote_b64=package_quote_b64(env),
        event_log=env.event_log,
        vm_config=env.vm_config,
        report_data_expected=report_data_expected,
        gpu_evidence=env.gpu_evidence,
        mode="offline_fixture",
    )
    return verify_tee(req, policy=policy)


__all__ = [
    "verify_offline_fixture_file",
    "verify_tee",
]
