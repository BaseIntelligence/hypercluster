"""VAL-TEE-010 compose-hash golden CLI + VAL-TEE-013 tee-offline scenario.

CI must stay green without TEE silicon or live dstack-verifier network.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hypercluster.attest.compose_hash import (
    hash_compose_file,
    load_golden_hash_file,
)
from hypercluster.attest.policy import DEFAULT_COMPOSE_HASH_GOLDEN
from hypercluster.cli import app as cli_app
from hypercluster.sim.scenarios import TEE_OFFLINE, run_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_COMPOSE = REPO_ROOT / "tests" / "fixtures" / "tee" / "golden_compose.yml"
GOLDEN_HASH_FILE = REPO_ROOT / "tests" / "fixtures" / "tee" / "golden_compose.sha256"
POSITIVE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tee" / "positive_tdx_v1.json"

runner = CliRunner()


# ----- VAL-TEE-010 -----------------------------------------------------------


def test_compose_hash_deterministic_two_runs() -> None:
    """VAL-TEE-010: two successive hashes of fixed fixture are equal."""

    h1 = hash_compose_file(GOLDEN_COMPOSE)
    h2 = hash_compose_file(GOLDEN_COMPOSE)
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert len(h1) == len("sha256:") + 64


def test_compose_hash_matches_known_golden_file() -> None:
    """VAL-TEE-010: hash matches committed golden_compose.sha256."""

    got = hash_compose_file(GOLDEN_COMPOSE)
    expected = load_golden_hash_file(GOLDEN_HASH_FILE)
    assert got == expected


def test_compose_hash_cli_two_runs_equal_and_match_golden() -> None:
    """VAL-TEE-010: CLI ``attest compose-hash`` is stable across two invocations."""

    r1 = runner.invoke(
        cli_app,
        ["attest", "compose-hash", "--compose-file", str(GOLDEN_COMPOSE)],
    )
    r2 = runner.invoke(
        cli_app,
        ["attest", "compose-hash", "--compose-file", str(GOLDEN_COMPOSE)],
    )
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    h1 = r1.stdout.strip().splitlines()[-1].strip()
    h2 = r2.stdout.strip().splitlines()[-1].strip()
    assert h1 == h2
    assert h1 == load_golden_hash_file(GOLDEN_HASH_FILE)
    # JSON path also emits stable key for tooling.
    r_json = runner.invoke(
        cli_app,
        [
            "attest",
            "compose-hash",
            "--compose-file",
            str(GOLDEN_COMPOSE),
            "--json",
        ],
    )
    assert r_json.exit_code == 0, r_json.output
    assert h1 in r_json.stdout


def test_compose_hash_cli_missing_file_nonzero() -> None:
    """CLI exits non-zero for missing compose file (handlable, room for Typer)."""

    missing = REPO_ROOT / "tests" / "fixtures" / "tee" / "does-not-exist.yml"
    result = runner.invoke(
        cli_app,
        ["attest", "compose-hash", "--compose-file", str(missing)],
    )
    assert result.exit_code != 0


# ----- VAL-TEE-013 -----------------------------------------------------------


def test_run_scenario_tee_offline_dispatches_green() -> None:
    """VAL-TEE-013: module dispatch tee-offline passes without live hardware."""

    # Base URL unused for offline core; still accepted as API tip for future.
    result = run_scenario(TEE_OFFLINE, "http://127.0.0.1:3200")
    assert result.ok is True, result.message
    assert result.name == TEE_OFFLINE
    joined = " ".join(result.steps).lower() + " " + result.message.lower()
    assert "positive" in joined or "verify" in joined
    # No live network reason codes expected among offline path steps.
    assert "live_not_available" not in joined
    assert "networkerror" not in joined.replace(" ", "")


def test_cli_sim_run_scenario_tee_offline_exit_0() -> None:
    """VAL-TEE-013: ``sim run-scenario --name tee-offline`` exits 0 offline."""

    result = runner.invoke(
        cli_app,
        [
            "sim",
            "run-scenario",
            "--name",
            "tee-offline",
            "--url",
            "http://127.0.0.1:3200",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.stdout
    assert "tee-offline" in result.stdout


def test_tee_offline_scenario_uses_offline_fixtures_not_live_mode() -> None:
    """VAL-TEE-013: scenario result steps include offline_fixture / golden path."""

    result = run_scenario("tee-offline", "http://127.0.0.1:9")  # unroutable
    assert result.ok is True, result.message
    steps_blob = "\n".join(result.steps).lower()
    assert "offline" in steps_blob or "fixture" in steps_blob
    # Must not claim live verify was attempted as a required step.
    assert "live verify required" not in steps_blob


def test_positive_fixture_compose_allowlist_pin_still_present() -> None:
    """Regression: offline golden allowlist pin remains the DEFAULT golden."""

    # Positive TDX fixture uses DEFAULT_COMPOSE_HASH_GOLDEN (allowlist pin),
    # which is distinct from the *file* compose-hash of golden_compose.yml —
    # file hash is a separate golden for VAL-TEE-010 CLI stability.
    assert DEFAULT_COMPOSE_HASH_GOLDEN.startswith("sha256:")
    assert POSITIVE_FIXTURE.is_file()
    assert GOLDEN_COMPOSE.is_file()
    file_hash = hash_compose_file(GOLDEN_COMPOSE)
    assert file_hash.startswith("sha256:")
    # File content may or may not equal the allowlist pin depending on fixture
    # design — only require it match its own golden companion.
    assert file_hash == load_golden_hash_file(GOLDEN_HASH_FILE)
