"""VAL-TEE-005..009, 012, 014, 015, 019, 020: tee_bonus + proof persistence."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

from hypercluster.attest.gpu_evidence import mock_gpu_evidence, validate_gpu_evidence
from hypercluster.attest.models import TeeVerifyRequest
from hypercluster.attest.offline_fixtures import (
    make_offline_envelope,
    package_quote_b64,
)
from hypercluster.attest.policy import TeeVerifyPolicy
from hypercluster.attest.report_data import build_report_data
from hypercluster.attest.verify import verify_tee
from hypercluster.domain.scoring_tee import (
    compute_tee_bonus,
    four_factor_composite,
    persist_score_for_attempt,
)
from hypercluster.settings import HyperSettings

COMPOSE_GOLDEN = (
    "sha256:0c0ffeec0a5eabcdef0123456789abcdef0123456789abcdef0123456789ab"
)
IMAGE = "sha256:sim000000000000000000000000000000000000000000000000000000000001"
NONCE = "n0nce-bonus-path-aaaa-bbbb-cccc-222222222222"


def _policy() -> TeeVerifyPolicy:
    return TeeVerifyPolicy(
        compose_allowlist=frozenset({COMPOSE_GOLDEN}),
        tcb_enforce=True,
        acceptable_tcb_statuses=frozenset({"UpToDate"}),
        disallowed_advisory_ids=frozenset(),
    )


def _hyper(**overrides: Any) -> HyperSettings:
    base = {
        "tee_bonus_tdx": 1.08,
        "tee_bonus_tdx_gpu": 1.20,
        "tee_live": False,
    }
    base.update(overrides)
    return HyperSettings(**base)


# ----- VAL-TEE-005 -----------------------------------------------------------


def test_sim_proof_tier_never_receives_live_tee_bonus() -> None:
    """VAL-TEE-005: pure sim → tee_bonus == 1.0 even when verified."""

    decision = compute_tee_bonus(
        proof_tier="sim",
        verified=True,
        verify_mode="sim",
        tee_mode="tdx",
        hyper=_hyper(),
    )
    assert decision.tee_bonus == pytest.approx(1.0)
    assert "sim_no_live_bonus" in decision.reason_codes
    # Even if caller pretends offline mode, sim tier wins.
    decision2 = compute_tee_bonus(
        proof_tier="sim",
        verified=True,
        verify_mode="offline_fixture",
        hyper=_hyper(),
    )
    assert decision2.tee_bonus == pytest.approx(1.0)


# ----- VAL-TEE-006 -----------------------------------------------------------


def test_valid_offline_tdx_applies_bonus_tdx() -> None:
    """VAL-TEE-006: verified offline TDX → HYPER_TEE_BONUS_TDX."""

    decision = compute_tee_bonus(
        proof_tier="tdx",
        verified=True,
        verify_mode="offline_fixture",
        tee_mode="tdx",
        hyper=_hyper(tee_bonus_tdx=1.08),
        is_valid_verdict=True,
    )
    assert decision.tee_bonus == pytest.approx(1.08)
    assert decision.bonus_applied
    assert decision.applied_tier == "tdx"


# ----- VAL-TEE-007 -----------------------------------------------------------


def test_valid_tdx_gpu_cc_applies_higher_bonus() -> None:
    """VAL-TEE-007: verified tdx+gpu_cc → HYPER_TEE_BONUS_TDX_GPU > TDX."""

    hyper = _hyper(tee_bonus_tdx=1.08, tee_bonus_tdx_gpu=1.20)
    tdx = compute_tee_bonus(
        proof_tier="tdx",
        verified=True,
        verify_mode="offline_fixture",
        hyper=hyper,
        is_valid_verdict=True,
    )
    gpu = compute_tee_bonus(
        proof_tier="tdx+gpu_cc",
        verified=True,
        verify_mode="offline_fixture",
        hyper=hyper,
        is_valid_verdict=True,
    )
    assert gpu.tee_bonus == pytest.approx(1.20)
    assert gpu.tee_bonus > tdx.tee_bonus


# ----- VAL-TEE-008 -----------------------------------------------------------


def test_unverified_tee_claim_no_bonus_and_integrity_zero() -> None:
    """VAL-TEE-008: tee=tdx + garbage quote → bonus 1.0, integrity_zero."""

    decision = compute_tee_bonus(
        proof_tier="tdx",
        verified=False,
        verify_mode="offline_fixture",
        tee_mode="tdx",
        hyper=_hyper(),
        is_valid_verdict=False,
    )
    assert decision.tee_bonus == pytest.approx(1.0)
    assert not decision.bonus_applied
    assert decision.integrity_zero is True
    composite = four_factor_composite(
        correctness=1.0,
        efficiency=1.0,
        fabric_gate=1.0,
        tee_bonus=decision.tee_bonus,
        integrity_zero=decision.integrity_zero,
    )
    assert composite == pytest.approx(0.0)


# ----- VAL-TEE-012 -----------------------------------------------------------


def test_gpu_evidence_nonce_echo_good_and_bad() -> None:
    """VAL-TEE-012: mock NRAS good nonce echo verifies; mismatch fails."""

    good = mock_gpu_evidence(nonce=NONCE)
    ok, reasons, evidence = validate_gpu_evidence(good, expected_nonce=NONCE, require=True)
    assert ok is True
    assert evidence is not None
    assert reasons == []

    bad = mock_gpu_evidence(nonce=NONCE, nonce_echo="wrong-echo")
    ok2, reasons2, _ = validate_gpu_evidence(bad, expected_nonce=NONCE, require=True)
    assert ok2 is False
    assert any("nonce" in r for r in reasons2)


def test_gpu_evidence_end_to_end_in_verify_tee() -> None:
    """VAL-TEE-012: offline tdx+gpu_cc rejects mismatched GPU nonce."""

    job_id = "job-gpu-cc-0001"
    report = build_report_data(job_id=job_id, image_digest=IMAGE, nonce=NONCE)
    good_gpu = mock_gpu_evidence(nonce=NONCE)
    from hypercluster.attest.offline_fixtures import OfflineQuoteEnvelope

    env = OfflineQuoteEnvelope(
        **{
            **make_offline_envelope(
                compose_hash=COMPOSE_GOLDEN,
                report_data=report,
                job_id=job_id,
                image_digest=IMAGE,
                nonce=NONCE,
                gpu_evidence=good_gpu,
                fixture_id="gpu_ok",
            ).model_dump(),
            "tee_type": "tdx+gpu_cc",
            "gpu_evidence": good_gpu,
        }
    )
    req = TeeVerifyRequest(
        quote_b64=package_quote_b64(env),
        report_data_expected=report,
        gpu_evidence=good_gpu,
        mode="offline_fixture",
    )
    result = verify_tee(
        req,
        policy=_policy(),
        require_gpu_evidence=True,
        expected_gpu_nonce=NONCE,
    )
    assert result.is_valid is True

    bad_gpu = mock_gpu_evidence(nonce="other-nonce-zzzz", nonce_echo="other-nonce-zzzz")
    env_bad = OfflineQuoteEnvelope(
        **{**env.model_dump(), "gpu_evidence": bad_gpu, "tee_type": "tdx+gpu_cc"}
    )
    req_bad = TeeVerifyRequest(
        quote_b64=package_quote_b64(env_bad),
        report_data_expected=report,
        gpu_evidence=bad_gpu,
        mode="offline_fixture",
    )
    result_bad = verify_tee(
        req_bad,
        policy=_policy(),
        require_gpu_evidence=True,
        expected_gpu_nonce=NONCE,
    )
    assert result_bad.is_valid is False
    assert any("gpu" in c or "nonce" in c for c in result_bad.reason_codes)


# ----- VAL-TEE-014 -----------------------------------------------------------


def test_live_verify_skip_safe_when_hyper_tee_live_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-TEE-014: live mode without HYPER_TEE_LIVE → skip reasons, no crash."""

    monkeypatch.delenv("HYPER_TEE_LIVE", raising=False)
    monkeypatch.setenv("HYPER_TEE_LIVE", "")
    from hypercluster.settings import clear_settings_cache

    clear_settings_cache()
    req = TeeVerifyRequest(
        quote_b64="bm90LWEteWVhci1xdW90ZQ==",
        report_data_expected=b"\x00" * 64,
        mode="live",
    )
    result = verify_tee(req, policy=_policy())
    assert result.is_valid is False
    assert "live_skipped" in result.reason_codes or "hyper_tee_live_unset" in result.reason_codes
    assert result.verify_mode == "live"


def test_cli_verify_live_skip_exit_code() -> None:
    """VAL-TEE-014: CLI verify-live exit 2 and explicit skip when unset."""

    from typer.testing import CliRunner

    from hypercluster.cli import app

    runner = CliRunner()
    env = {k: v for k, v in os.environ.items() if k != "HYPER_TEE_LIVE"}
    env["HYPER_TEE_LIVE"] = ""
    result = runner.invoke(app, ["attest", "verify-live"], env=env)
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["skipped"] is True
    assert payload["reason"] == "live_skipped"


# ----- VAL-TEE-015 -----------------------------------------------------------


def test_capability_includes_tee_verification_and_ordinary() -> None:
    """VAL-TEE-015: Settings enable tee_verification + ordinary proof."""

    from hypercluster.settings import Settings, clear_settings_cache

    clear_settings_cache()
    s = Settings(shared_token="t", shared_token_file=None)
    assert "challenge.tee_verification" in s.capabilities
    assert "challenge.ordinary_proof" in s.capabilities
    assert "challenge.scoring" in s.capabilities


def test_ordinary_path_scores_without_quote() -> None:
    """VAL-TEE-015: tee=none → ordinary tier, bonus 1.0, no verifier demand."""

    decision = compute_tee_bonus(
        proof_tier="ordinary",
        verified=True,
        verify_mode="sim",
        tee_mode="none",
        hyper=_hyper(),
    )
    assert decision.tee_bonus == pytest.approx(1.0)
    assert decision.applied_tier == "ordinary"


# ----- VAL-TEE-019 -----------------------------------------------------------


def test_offline_fixture_mode_never_dials_http() -> None:
    """VAL-TEE-019: mode=offline_fixture never uses httpx client."""

    job_id = "job-offline-no-http"
    report = build_report_data(job_id=job_id, image_digest=IMAGE, nonce=NONCE)
    env = make_offline_envelope(
        compose_hash=COMPOSE_GOLDEN,
        report_data=report,
        job_id=job_id,
        image_digest=IMAGE,
        nonce=NONCE,
    )
    client = MagicMock()
    client.post = MagicMock(side_effect=AssertionError("must not dial network"))
    client.get = MagicMock(side_effect=AssertionError("must not dial network"))
    req = TeeVerifyRequest(
        quote_b64=package_quote_b64(env),
        report_data_expected=report,
        mode="offline_fixture",
    )
    result = verify_tee(req, policy=_policy(), httpx_client=client)
    assert result.is_valid is True
    assert result.verify_mode == "offline_fixture"
    client.post.assert_not_called()
    client.get.assert_not_called()


# ----- VAL-TEE-009 / VAL-TEE-020 (persistence) --------------------------------


@pytest.mark.asyncio
async def test_job_proofs_persist_verdict_and_score_bonus_invariant(
    settings_factory, tmp_path: Path
) -> None:
    """VAL-TEE-009 + VAL-TEE-020: proofs store verdict/mode; bonus iff verified."""

    from hypercluster.app import create_app
    from hypercluster.db.models import Job, JobAttempt, JobProof, Score
    from hypercluster.domain.tee_proofs import (
        build_sim_proof,
        verify_and_build_proof,
    )

    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'tee-persist.sqlite3'}",
        shared_token="t",
        shared_token_file=None,
    )
    hyper = _hyper()
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        database = app.state.database
        async with database.session() as session:
            # --- sim job never gets live bonus ---
            job_sim = Job(
                id=str(uuid.uuid4()),
                submitter_hotkey="hk-sim",
                status="succeeded",
                image_digest=IMAGE,
                entrypoint_json=json.dumps(["python", "-m", "x"]),
                world_size=1,
                nnodes=1,
                nproc_per_node=1,
                backend="nccl",
                fabric_mode="auto",
                tee_mode="none",
                resource_json=json.dumps({"gpus": 1}),
                timeout_s=60,
            )
            session.add(job_sim)
            attempt_sim = JobAttempt(
                id=str(uuid.uuid4()),
                job_id=job_sim.id,
                attempt_no=1,
                status="succeeded",
            )
            session.add(attempt_sim)
            await session.flush()
            proof_sim = build_sim_proof(attempt_id=attempt_sim.id, job=job_sim)
            session.add(proof_sim)
            await session.flush()
            assert proof_sim.verify_mode == "sim"
            assert proof_sim.dstack_verdict_json is not None
            verdict_sim = json.loads(proof_sim.dstack_verdict_json)
            assert "is_valid" in verdict_sim
            score_sim = await persist_score_for_attempt(
                session,
                attempt_id=attempt_sim.id,
                hotkey=job_sim.submitter_hotkey,
                correctness=1.0,
                efficiency=1.0,
                fabric_gate=1.0,
                proof=proof_sim,
                tee_mode="none",
                hyper=hyper,
            )
            assert score_sim.tee_bonus == pytest.approx(1.0)

            # --- offline verified TDX bonus ---
            job_id = str(uuid.uuid4())
            nonce = NONCE
            report = build_report_data(job_id=job_id, image_digest=IMAGE, nonce=nonce)
            env = make_offline_envelope(
                compose_hash=COMPOSE_GOLDEN,
                report_data=report,
                job_id=job_id,
                image_digest=IMAGE,
                nonce=nonce,
                fixture_id="bonus_tdx",
            )
            job_tdx = Job(
                id=job_id,
                submitter_hotkey="hk-tdx",
                status="succeeded",
                image_digest=IMAGE,
                entrypoint_json=json.dumps(["python", "-m", "x"]),
                world_size=1,
                nnodes=1,
                nproc_per_node=1,
                backend="nccl",
                fabric_mode="auto",
                tee_mode="tdx",
                resource_json=json.dumps({"gpus": 1}),
                timeout_s=60,
            )
            session.add(job_tdx)
            attempt_tdx = JobAttempt(
                id=str(uuid.uuid4()),
                job_id=job_tdx.id,
                attempt_no=1,
                status="succeeded",
            )
            session.add(attempt_tdx)
            await session.flush()
            proof_tdx, result = verify_and_build_proof(
                attempt_id=attempt_tdx.id,
                job=job_tdx,
                quote_b64=package_quote_b64(env),
                report_data_expected=report,
                mode="offline_fixture",
                policy=_policy(),
            )
            assert result.is_valid is True
            session.add(proof_tdx)
            await session.flush()
            assert proof_tdx.verified == 1
            assert proof_tdx.verify_mode == "offline_fixture"
            assert proof_tdx.dstack_verdict_json is not None
            v = json.loads(proof_tdx.dstack_verdict_json)
            assert v["is_valid"] is True
            score_tdx = await persist_score_for_attempt(
                session,
                attempt_id=attempt_tdx.id,
                hotkey=job_tdx.submitter_hotkey,
                correctness=1.0,
                efficiency=1.0,
                fabric_gate=1.0,
                proof=proof_tdx,
                tee_mode="tdx",
                hyper=hyper,
            )
            assert score_tdx.tee_bonus == pytest.approx(1.08)
            assert score_tdx.composite == pytest.approx(1.08)

            # --- GPU tier higher bonus ---
            job_gpu_id = str(uuid.uuid4())
            report_g = build_report_data(
                job_id=job_gpu_id, image_digest=IMAGE, nonce=nonce
            )
            gpu_ev = mock_gpu_evidence(nonce=nonce)
            from hypercluster.attest.offline_fixtures import OfflineQuoteEnvelope

            env_g = OfflineQuoteEnvelope(
                **{
                    **make_offline_envelope(
                        compose_hash=COMPOSE_GOLDEN,
                        report_data=report_g,
                        job_id=job_gpu_id,
                        image_digest=IMAGE,
                        nonce=nonce,
                        gpu_evidence=gpu_ev,
                    ).model_dump(),
                    "tee_type": "tdx+gpu_cc",
                    "gpu_evidence": gpu_ev,
                }
            )
            job_gpu = Job(
                id=job_gpu_id,
                submitter_hotkey="hk-gpu",
                status="succeeded",
                image_digest=IMAGE,
                entrypoint_json=json.dumps(["python", "-m", "x"]),
                world_size=1,
                nnodes=1,
                nproc_per_node=1,
                backend="nccl",
                fabric_mode="auto",
                tee_mode="tdx+gpu_cc",
                resource_json=json.dumps({"gpus": 1}),
                timeout_s=60,
            )
            session.add(job_gpu)
            attempt_gpu = JobAttempt(
                id=str(uuid.uuid4()),
                job_id=job_gpu.id,
                attempt_no=1,
                status="succeeded",
            )
            session.add(attempt_gpu)
            await session.flush()
            proof_gpu, result_g = verify_and_build_proof(
                attempt_id=attempt_gpu.id,
                job=job_gpu,
                quote_b64=package_quote_b64(env_g),
                report_data_expected=report_g,
                gpu_evidence=gpu_ev,
                mode="offline_fixture",
                policy=_policy(),
                require_gpu_evidence=True,
                expected_gpu_nonce=nonce,
            )
            assert result_g.is_valid is True
            session.add(proof_gpu)
            await session.flush()
            score_gpu = await persist_score_for_attempt(
                session,
                attempt_id=attempt_gpu.id,
                hotkey=job_gpu.submitter_hotkey,
                correctness=1.0,
                efficiency=1.0,
                fabric_gate=1.0,
                proof=proof_gpu,
                tee_mode="tdx+gpu_cc",
                hyper=hyper,
            )
            assert score_gpu.tee_bonus == pytest.approx(1.20)
            assert score_gpu.tee_bonus > score_tdx.tee_bonus

            # --- unverified claim + garbage ---
            job_bad_id = str(uuid.uuid4())
            job_bad = Job(
                id=job_bad_id,
                submitter_hotkey="hk-bad",
                status="succeeded",
                image_digest=IMAGE,
                entrypoint_json=json.dumps(["python", "-m", "x"]),
                world_size=1,
                nnodes=1,
                nproc_per_node=1,
                backend="nccl",
                fabric_mode="auto",
                tee_mode="tdx",
                resource_json=json.dumps({"gpus": 1}),
                timeout_s=60,
            )
            session.add(job_bad)
            attempt_bad = JobAttempt(
                id=str(uuid.uuid4()),
                job_id=job_bad.id,
                attempt_no=1,
                status="succeeded",
            )
            session.add(attempt_bad)
            await session.flush()
            proof_bad, result_bad = verify_and_build_proof(
                attempt_id=attempt_bad.id,
                job=job_bad,
                quote_b64="bm90LWEtdmFsaWQtcXVvdGU=",
                report_data_expected=b"\x00" * 64,
                mode="offline_fixture",
                policy=_policy(),
            )
            assert result_bad.is_valid is False
            session.add(proof_bad)
            await session.flush()
            assert proof_bad.verified == 0
            assert proof_bad.dstack_verdict_json is not None
            score_bad = await persist_score_for_attempt(
                session,
                attempt_id=attempt_bad.id,
                hotkey=job_bad.submitter_hotkey,
                correctness=1.0,
                efficiency=1.0,
                fabric_gate=1.0,
                proof=proof_bad,
                tee_mode="tdx",
                hyper=hyper,
            )
            assert score_bad.tee_bonus == pytest.approx(1.0)
            assert score_bad.composite == pytest.approx(0.0)

            await session.commit()

            # VAL-TEE-020 hard join invariant over scores ↔ proofs.
            rows = (
                await session.execute(
                    select(Score, JobProof).join(
                        JobProof, JobProof.attempt_id == Score.attempt_id
                    )
                )
            ).all()
            assert rows
            for score, proof in rows:
                if score.tee_bonus > 1.0 + 1e-12:
                    assert proof.verified == 1
                    verdict = json.loads(proof.dstack_verdict_json or "{}")
                    assert verdict.get("is_valid") is True
                    assert proof.verify_mode in {"offline_fixture", "live"}


@pytest.mark.asyncio
async def test_lifecycle_sim_job_tee_bonus_one(
    settings_factory, tmp_path: Path
) -> None:
    """VAL-TEE-005 + VAL-TEE-015: lifecycle sim collect stores score bonus 1.0."""


    from httpx import ASGITransport, AsyncClient

    from hypercluster.api.auth import build_signed_headers
    from hypercluster.app import create_app
    from hypercluster.db.models import Score
    from hypercluster.domain.job_lifecycle import (
        get_latest_attempt,
        get_proofs_for_attempt,
        run_job_to_terminal,
    )
    from hypercluster.settings import HyperSettings

    TOKEN = "test-challenge-shared-token"
    HK = "tee-life-submitter-aaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    settings = settings_factory(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'tee-life.sqlite3'}",
        shared_token=TOKEN,
        shared_token_file=None,
    )
    hyper = HyperSettings(
        allow_insecure_signatures=True,
        combined_worker=False,
        job_image_allowlist=IMAGE,
        sim_auto_capacity=True,
        tee_bonus_tdx=1.08,
        tee_bonus_tdx_gpu=1.20,
    )
    app = create_app(settings, hyper_settings=hyper)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            body = {
                "image_digest": IMAGE,
                "entrypoint": ["python", "-m", "x"],
                "world_size": 1,
                "nnodes": 1,
                "nproc_per_node": 1,
                "timeout_s": 60,
                "resource": {"gpus": 1, "nodes": 1},
                "tee": "none",
                "fabric": "auto",
            }
            raw = json.dumps(body).encode()
            headers = build_signed_headers(secret=TOKEN, hotkey=HK, body=raw)
            headers["content-type"] = "application/json"
            resp = await client.post("/v1/jobs", content=raw, headers=headers)
            assert resp.status_code == 200, resp.text
            job_id = resp.json()["id"]

        async with app.state.database.session() as session:
            job = await run_job_to_terminal(session, job_id, hyper=hyper)
            assert job.status == "succeeded"
            attempt = await get_latest_attempt(session, job_id)
            assert attempt is not None
            proofs = await get_proofs_for_attempt(session, attempt.id)
            assert proofs
            proof = proofs[0]
            assert proof.verify_mode == "sim"
            assert proof.dstack_verdict_json is not None
            # sim tier / ordinary — never TDX live bonus
            assert proof.proof_tier in {"sim", "ordinary"}
            score_row = (
                await session.execute(select(Score).where(Score.attempt_id == attempt.id))
            ).scalar_one_or_none()
            assert score_row is not None
            assert score_row.tee_bonus == pytest.approx(1.0)
