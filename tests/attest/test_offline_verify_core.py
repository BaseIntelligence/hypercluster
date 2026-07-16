"""VAL-TEE-001/002/003/004/011/016/017/018: offline TEE fixture verify core.

Fail-closed offline path: positive fixture, mutated compose, nonce/report_data
bind, TCB enforce, compose allowlist, job_digest cross-bind, multi reason_codes.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import pytest

from hypercluster.attest.models import TeeVerifyRequest, TeeVerifyResult
from hypercluster.attest.offline_fixtures import (
    OfflineQuoteEnvelope,
    load_quote_fixture,
    make_offline_envelope,
    package_quote_b64,
)
from hypercluster.attest.policy import TeeVerifyPolicy
from hypercluster.attest.report_data import (
    REPORT_DATA_SIZE,
    ReportDataLayoutError,
    build_job_digest,
    build_report_data,
    parse_report_data,
)
from hypercluster.attest.verify import verify_tee

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tee"
POSITIVE = FIXTURES / "positive_tdx_v1.json"

# Positive golden binding inputs (must match fixture).
JOB_A = "job-offline-positive-0001"
IMAGE_A = "sha256:sim000000000000000000000000000000000000000000000000000000000001"
NONCE_A = "n0nce-posit1ve-aaaa-bbbb-cccc-111111111111"
COMPOSE_GOLDEN = "sha256:0c0ffeec0a5eabcdef0123456789abcdef0123456789abcdef0123456789ab"


def _policy(**overrides: object) -> TeeVerifyPolicy:
    defaults: dict[str, object] = {
        "compose_allowlist": frozenset({COMPOSE_GOLDEN}),
        "tcb_enforce": True,
        "acceptable_tcb_statuses": frozenset({"UpToDate"}),
        "disallowed_advisory_ids": frozenset(),
    }
    defaults.update(overrides)
    return TeeVerifyPolicy(**defaults)  # type: ignore[arg-type]


def _expected_report_data(
    *,
    job_id: str = JOB_A,
    image_digest: str = IMAGE_A,
    nonce: str = NONCE_A,
) -> bytes:
    return build_report_data(
        job_id=job_id,
        image_digest=image_digest,
        nonce=nonce,
    )


def _request_from_fixture(
    path: Path,
    *,
    report_data_expected: bytes | None = None,
    mode: str = "offline_fixture",
) -> TeeVerifyRequest:
    env = load_quote_fixture(path)
    return TeeVerifyRequest(
        quote_b64=package_quote_b64(env),
        event_log=env.event_log,
        vm_config=env.vm_config,
        report_data_expected=report_data_expected
        if report_data_expected is not None
        else bytes.fromhex(env.report_data_hex),
        gpu_evidence=env.gpu_evidence,
        mode=mode,  # type: ignore[arg-type]
    )


# ----- VAL-TEE-001 -----------------------------------------------------------


def test_offline_positive_fixture_is_valid() -> None:
    """VAL-TEE-001: positive offline fixture verifies is_valid true."""

    req = _request_from_fixture(POSITIVE)
    result = verify_tee(req, policy=_policy())
    assert isinstance(result, TeeVerifyResult)
    assert result.is_valid is True
    assert result.quote_verified is True
    assert result.compose_hash == COMPOSE_GOLDEN
    assert result.tcb_status == "UpToDate"
    assert result.reason_codes == []


def test_verify_offline_file_helper_matches_positive() -> None:
    """VAL-TEE-001 API equivalent path: verify fixture file end-to-end."""

    from hypercluster.attest.verify import verify_offline_fixture_file

    result = verify_offline_fixture_file(
        POSITIVE,
        policy=_policy(),
        report_data_expected=_expected_report_data(),
    )
    assert result.is_valid is True
    assert result.quote_verified is True


# ----- VAL-TEE-002 -----------------------------------------------------------


def test_mutated_compose_hash_rejected() -> None:
    """VAL-TEE-002: wrong compose_hash / measurement fails closed."""

    env = load_quote_fixture(POSITIVE)
    bad = env.model_copy(
        update={
            "compose_hash": (
                "sha256:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
            )
        }
    )
    # Mutated hash must still be "in" allowlist sense by default policy? No —
    # allowlist still has golden only; also field compose_mismatch vs envelope.
    req = TeeVerifyRequest(
        quote_b64=package_quote_b64(bad),
        event_log=bad.event_log,
        report_data_expected=bytes.fromhex(env.report_data_hex),
        mode="offline_fixture",
    )
    result = verify_tee(req, policy=_policy())
    assert result.is_valid is False
    assert result.quote_verified is False or result.is_valid is False
    codes = set(result.reason_codes)
    assert codes & {
        "compose_hash_mismatch",
        "compose_not_allowlisted",
        "measurement_mismatch",
    }


def test_wrong_compose_with_allowlist_also_documents_mismatch_when_fixture_marked() -> None:
    """VAL-TEE-002: when fixture internal expected_compose differs, emit compose mismatch."""

    env = load_quote_fixture(POSITIVE)
    # Keep allowlisted so we isolate measurement/compose field consistency
    # via intentional expected_compose_hash on envelope.
    env2 = env.model_copy(
        update={
            "compose_hash": COMPOSE_GOLDEN,
            "expected_compose_hash": (
                "sha256:1111111111111111111111111111111111111111111111111111111111111111"
            ),
        }
    )
    req = TeeVerifyRequest(
        quote_b64=package_quote_b64(env2),
        report_data_expected=bytes.fromhex(env.report_data_hex),
        mode="offline_fixture",
    )
    # Allowlist includes golden so allowlist alone is ok; internal measure mismatch.
    result = verify_tee(req, policy=_policy())
    assert result.is_valid is False
    assert "compose_hash_mismatch" in result.reason_codes


# ----- VAL-TEE-003 -----------------------------------------------------------


def test_wrong_nonce_report_data_rejected() -> None:
    """VAL-TEE-003: stale/wrong nonce binding fails (report_data bind)."""

    req = _request_from_fixture(
        POSITIVE,
        report_data_expected=_expected_report_data(nonce="stale-or-wrong-nonce-zzzzzzzzzzzzzz"),
    )
    result = verify_tee(req, policy=_policy())
    assert result.is_valid is False
    codes = set(result.reason_codes)
    assert codes & {"nonce_mismatch", "binding_mismatch", "stale_nonce", "report_data_mismatch"}


# ----- VAL-TEE-004 -----------------------------------------------------------


def test_bad_tcb_fail_closed_when_enforce_on() -> None:
    """VAL-TEE-004: enforce-on rejects bad tcb_status / disallowed advisories."""

    env = load_quote_fixture(POSITIVE)
    bad = env.model_copy(
        update={
            "tcb_status": "OutOfDate",
            "advisory_ids": ["INTEL-SA-00000"],
        }
    )
    req = TeeVerifyRequest(
        quote_b64=package_quote_b64(bad),
        report_data_expected=bytes.fromhex(env.report_data_hex),
        mode="offline_fixture",
    )
    result = verify_tee(
        req,
        policy=_policy(
            tcb_enforce=True,
            disallowed_advisory_ids=frozenset({"INTEL-SA-00000"}),
        ),
    )
    assert result.is_valid is False
    codes = set(result.reason_codes)
    assert codes & {
        "tcb_status_reject",
        "advisory_disallowed",
        "tcb_fail",
    }


def test_bad_tcb_may_allow_when_enforce_off() -> None:
    """VAL-TEE-004: enforce-off may accept with flags (not default CI path)."""

    env = load_quote_fixture(POSITIVE)
    bad = env.model_copy(update={"tcb_status": "OutOfDate", "advisory_ids": ["SA-X"]})
    req = TeeVerifyRequest(
        quote_b64=package_quote_b64(bad),
        report_data_expected=bytes.fromhex(env.report_data_hex),
        mode="offline_fixture",
    )
    result = verify_tee(
        req,
        policy=_policy(tcb_enforce=False, disallowed_advisory_ids=frozenset({"SA-X"})),
    )
    # Soft path: may still be valid with reason flags documenting advisories.
    assert result.is_valid is True
    assert result.tcb_status == "OutOfDate"
    assert "tcb_advisory_soft" in result.reason_codes or result.advisory_ids


# ----- VAL-TEE-011 -----------------------------------------------------------


def test_report_data_parser_rejects_truncated() -> None:
    """VAL-TEE-011: truncated report_data fails with layout reason."""

    good = build_report_data(job_id=JOB_A, image_digest=IMAGE_A, nonce=NONCE_A)
    with pytest.raises(ReportDataLayoutError) as exc:
        parse_report_data(good[:16])
    assert "truncated" in str(exc.value).lower() or "layout" in str(exc.value).lower()

    # Verify path also surfaces layout reason_codes (not crypto accept).
    env = load_quote_fixture(POSITIVE)
    truncated_hex = env.report_data_hex[:20]
    mutated = env.model_copy(update={"report_data_hex": truncated_hex})
    req = TeeVerifyRequest(
        quote_b64=package_quote_b64(mutated),
        report_data_expected=good,
        mode="offline_fixture",
    )
    result = verify_tee(req, policy=_policy())
    assert result.is_valid is False
    assert any(
        c in result.reason_codes
        for c in ("report_data_layout", "report_data_truncated", "report_data_invalid")
    )


def test_report_data_parser_rejects_extra_trailing() -> None:
    """VAL-TEE-011: extra trailing bytes / oversize field rejected."""

    good = build_report_data(job_id=JOB_A, image_digest=IMAGE_A, nonce=NONCE_A)
    bloated = good + b"\xff\xee"
    with pytest.raises(ReportDataLayoutError):
        parse_report_data(bloated)

    env = load_quote_fixture(POSITIVE)
    extra_hex = env.report_data_hex + "deadbeef"
    mutated = env.model_copy(update={"report_data_hex": extra_hex})
    req = TeeVerifyRequest(
        quote_b64=package_quote_b64(mutated),
        report_data_expected=good,
        mode="offline_fixture",
    )
    result = verify_tee(req, policy=_policy())
    assert result.is_valid is False
    assert any(
        c in result.reason_codes
        for c in ("report_data_layout", "report_data_extra", "report_data_invalid")
    )


def test_random_blob_not_valid_tdx() -> None:
    """VAL-TEE-011: arbitrary blob is never accepted as valid TDX offline quote."""

    junk = base64.b64encode(b"not-a-quote-blob" + b"\x00" * 40).decode()
    req = TeeVerifyRequest(
        quote_b64=junk,
        report_data_expected=_expected_report_data(),
        mode="offline_fixture",
    )
    result = verify_tee(req, policy=_policy())
    assert result.is_valid is False
    assert result.quote_verified is False


# ----- VAL-TEE-016 -----------------------------------------------------------


def test_unknown_compose_hash_fail_closed_allowlist() -> None:
    """VAL-TEE-016: compose_hash not on allowlist rejected."""

    env = load_quote_fixture(POSITIVE)
    foreign = "sha256:ffffffffffffeeeeeeeeeeeeeeeeeeeeddddddddddddddddcccccccccccccc"
    env2 = env.model_copy(
        update={
            "compose_hash": foreign,
            "expected_compose_hash": foreign,
        }
    )
    req = TeeVerifyRequest(
        quote_b64=package_quote_b64(env2),
        report_data_expected=bytes.fromhex(env.report_data_hex),
        mode="offline_fixture",
    )
    result = verify_tee(req, policy=_policy(compose_allowlist=frozenset({COMPOSE_GOLDEN})))
    assert result.is_valid is False
    assert any(
        c in result.reason_codes
        for c in ("compose_not_allowlisted", "allowlist_miss", "measurement_not_allowlisted")
    )


# ----- VAL-TEE-017 -----------------------------------------------------------


def test_job_digest_mutation_invalidates_prior_quote() -> None:
    """VAL-TEE-017: quote for job A fails when rebound to job B."""

    job_b = "job-offline-other-9999"
    expected_b = _expected_report_data(job_id=job_b)
    # Quote still carries job A's report_data; expected for B must reject.
    req = _request_from_fixture(POSITIVE, report_data_expected=expected_b)
    result = verify_tee(req, policy=_policy())
    assert result.is_valid is False
    codes = set(result.reason_codes)
    assert codes & {
        "job_digest_mismatch",
        "binding_mismatch",
        "report_data_mismatch",
        "cross_job_binding",
    }


def test_build_job_digest_differs_across_jobs() -> None:
    d1 = build_job_digest(job_id=JOB_A, image_digest=IMAGE_A)
    d2 = build_job_digest(job_id="job-other", image_digest=IMAGE_A)
    assert d1 != d2
    assert len(d1) == 32


# ----- VAL-TEE-018 -----------------------------------------------------------


def test_multiple_reason_codes_on_compound_failure() -> None:
    """VAL-TEE-018: wrong compose + bad TCB returns multiple reason_codes."""

    env = load_quote_fixture(POSITIVE)
    compound = env.model_copy(
        update={
            "compose_hash": (
                "sha256:badcompose00badcompose00badcompose00badcompose00badcompose00badcomp00"
            ),
            "expected_compose_hash": COMPOSE_GOLDEN,  # internal vs actual mismatch
            "tcb_status": "Revoked",
            "advisory_ids": ["INTEL-SA-99999"],
        }
    )
    req = TeeVerifyRequest(
        quote_b64=package_quote_b64(compound),
        report_data_expected=bytes.fromhex(env.report_data_hex),
        mode="offline_fixture",
    )
    result = verify_tee(
        req,
        policy=_policy(
            tcb_enforce=True,
            disallowed_advisory_ids=frozenset({"INTEL-SA-99999"}),
        ),
    )
    assert result.is_valid is False
    assert len(result.reason_codes) >= 2
    # Must surface both families, not only first.
    joined = " ".join(result.reason_codes)
    assert "compose" in joined or "allowlist" in joined or "measurement" in joined
    assert "tcb" in joined or "advisory" in joined


# ----- report_data layout unit core ----------------------------------------


def test_report_data_roundtrip_fixed_size() -> None:
    field = build_report_data(job_id=JOB_A, image_digest=IMAGE_A, nonce=NONCE_A)
    assert len(field) == REPORT_DATA_SIZE
    parsed = parse_report_data(field)
    assert parsed.job_digest == build_job_digest(job_id=JOB_A, image_digest=IMAGE_A)
    assert parsed.nonce_digest  # opaque but present


def test_make_envelope_helper_builds_verifiable_positive() -> None:
    """Synthetic envelope path stays self-consistent for mutations in other tests."""

    rd = build_report_data(job_id=JOB_A, image_digest=IMAGE_A, nonce=NONCE_A)
    env = make_offline_envelope(
        compose_hash=COMPOSE_GOLDEN,
        report_data=rd,
        tcb_status="UpToDate",
        job_id=JOB_A,
        image_digest=IMAGE_A,
        nonce=NONCE_A,
    )
    assert isinstance(env, OfflineQuoteEnvelope)
    result = verify_tee(
        TeeVerifyRequest(
            quote_b64=package_quote_b64(env),
            report_data_expected=rd,
            mode="offline_fixture",
        ),
        policy=_policy(),
    )
    assert result.is_valid is True


def test_offline_mode_does_not_require_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """offline_fixture path must not dial remote dstack HTTP (local fail hard)."""

    import httpx

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("offline_fixture must not open network")

    monkeypatch.setattr(httpx, "Client", _boom)
    monkeypatch.setattr(httpx, "AsyncClient", _boom)
    result = verify_tee(_request_from_fixture(POSITIVE), policy=_policy())
    assert result.is_valid is True


def test_positive_fixture_file_json_stable() -> None:
    """Vendor fixture file is loadable JSON with required fields."""

    raw = json.loads(POSITIVE.read_text(encoding="utf-8"))
    assert raw["compose_hash"] == COMPOSE_GOLDEN
    assert raw["tcb_status"] == "UpToDate"
    assert len(raw["report_data_hex"]) == REPORT_DATA_SIZE * 2


def test_copy_helper_preserved() -> None:
    """Sanity: copy fixture doesn't mutate original."""

    env = load_quote_fixture(POSITIVE)
    env2 = copy.deepcopy(env)
    assert env2.compose_hash == env.compose_hash
