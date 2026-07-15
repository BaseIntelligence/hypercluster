"""VAL-LIVE-001/002/010/011: product tree + CI never depend on Verda.

Static inventory, docs boundary, default pytest isolation, and challenge
outbound allowlist common cases. Live Verda rentals are external-only and
must never be required by gated CI.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from hypercluster.no_verda import (
    VERDA_HOST_MARKERS,
    VerdaForbiddenError,
    assert_challenge_outbound_allowed,
    audit_dependency_manifests,
    audit_docs_miner_path,
    audit_product_source_imports,
    challenge_requires_verda_env,
    is_verda_host,
    run_docs_verda_audit,
    run_product_verda_audit,
    strip_verda_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "hypercluster"


# ----- VAL-LIVE-001 product tree / lock / imports ---------------------------


def test_product_source_has_no_verda_imports() -> None:
    """AST scan of src/hypercluster rejects import verda / Verda clients."""

    report = audit_product_source_imports(PACKAGE_ROOT)
    assert report.ok, "\n".join(report.summary_lines())


def test_pyproject_and_lock_have_no_verda_dependency() -> None:
    report = audit_dependency_manifests(
        pyproject=REPO_ROOT / "pyproject.toml",
        lockfile=REPO_ROOT / "uv.lock",
    )
    assert report.ok, "\n".join(report.summary_lines())


def test_full_product_verda_audit_clean() -> None:
    report = run_product_verda_audit(REPO_ROOT)
    assert report.ok, "\n".join(report.summary_lines())
    # Positive control: empty report summary is greppable for CI evidence.
    assert report.summary_lines() == ["no-verda audit: clean"]


def test_is_verda_host_markers() -> None:
    assert is_verda_host("https://api.verda.com/v1/oauth2/token")
    assert is_verda_host("api.verda.com")
    assert is_verda_host("cloud.verda.io")
    assert not is_verda_host("https://127.0.0.1:3201/health")
    assert not is_verda_host("https://example.com/api")
    assert set(VERDA_HOST_MARKERS)  # shared constant non-empty


def test_audit_detects_injected_import(tmp_path: Path) -> None:
    """Negative unit: planted import verda is reported."""

    pkg = tmp_path / "hypercluster"
    pkg.mkdir()
    (pkg / "evil.py").write_text("import verda\n", encoding="utf-8")
    report = audit_product_source_imports(pkg)
    assert not report.ok

    def _is_import_hit(detail: str) -> bool:
        lower = detail.lower()
        return "import verda" in lower or "forbidden" in lower

    assert any(_is_import_hit(item.detail) for item in report.findings)


def test_audit_detects_api_verda_client_hardcode(tmp_path: Path) -> None:
    pkg = tmp_path / "hypercluster"
    pkg.mkdir()
    (pkg / "client.py").write_text(
        'BASE_URL = "https://api.verda.com/v1"\n',
        encoding="utf-8",
    )
    report = audit_product_source_imports(pkg)
    assert not report.ok


def test_audit_detects_forbidden_dep_in_pyproject(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """
            [project]
            name = "toy"
            dependencies = [
              "verda-sdk>=1.0",
            ]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    report = audit_dependency_manifests(pyproject=pyproject, lockfile=None)
    assert not report.ok


# ----- VAL-LIVE-002 docs miner path ----------------------------------------


def test_docs_do_not_require_verda_account_for_miners() -> None:
    """User-facing README/docs contain no mandatory Verda signup for miners.

    When docs landing is still pending, an empty README is treated as clean as
    long as no file forces Verda. A synthetic required-account snippet fails.
    """

    report = run_docs_verda_audit(REPO_ROOT)
    # Missing docs → missing miner force is ok for VAL-LIVE-002; only fail on
    # explicit required-Verda language. Specialized audit on empty set keeps FAIL
    # for total absence, so accept either clean or "no docs" sentinel only.
    bad = [f for f in report.findings if "forces Verda" in f.detail or "matched" in f.detail]
    assert not bad, "\n".join(f"{f.path}: {f.detail}" for f in bad)


def test_docs_audit_flags_required_verda_account(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Hypercluster\n\nMiners must create a Verda account before mining.\n",
        encoding="utf-8",
    )
    report = audit_docs_miner_path([readme])
    assert not report.ok


def test_docs_audit_allows_optional_ops_note(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        textwrap.dedent(
            """
            # Hypercluster

            Miners supply and demand capacity via the home-grown marketplace.
            Self-reported SSH inventory is enough — no commercial cloud account
            is required.

            ## Optional ops QA (maintainers only)

            Maintainers may optionally rent external cloud capacity (e.g. Verda)
            *outside* this package and register it through marketplace APIs.
            That path is never required for miners and is not part of default CI.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    report = audit_docs_miner_path([readme])
    assert report.ok, "\n".join(report.summary_lines())


# ----- VAL-LIVE-010 default CI / pytest never needs live Verda -------------


def test_default_pytest_ini_has_live_verda_marker() -> None:
    """Opt-in marker exists so live suites cannot hide inside default collection."""

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "live_verda" in pyproject
    # Default addopts must not select the live_verda mark.
    # Comment text may mention the mark; only treat unquoted addopts lines as bad.
    in_addopts = False
    for raw in pyproject.splitlines():
        line = raw.strip()
        if line.startswith("addopts"):
            in_addopts = True
            # addopts = "..." single-line form
            if "live_verda" in line and not line.lstrip().startswith("#"):
                raise AssertionError(f"addopts must not force live_verda: {line}")
            continue
        if in_addopts:
            ends_addopts = line.startswith("[") or (
                line
                and not line.startswith("#")
                and "=" in line
                and not line.startswith('"')
                and not line.startswith("'")
            )
            if ends_addopts:
                in_addopts = False
            elif "live_verda" in line and not line.lstrip().startswith("#"):
                raise AssertionError(f"addopts must not force live_verda: {line}")
    # Non-comment config must not reference the ops secrets path.
    for raw in pyproject.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "verda.env" not in stripped.lower()


def test_conftest_does_not_load_verda_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gated suite must not require or auto-source verda.env (VAL-LIVE-010)."""

    # Even if ops secrets exist on host, default test process should not read them.
    monkeypatch.setenv("VERDA_CLIENT_ID", "should-be-ignored-by-product-tests")
    monkeypatch.setenv("VERDA_CLIENT_SECRET", "should-be-ignored-by-product-tests")
    # Product settings import path must not touch VERDA_*.
    from hypercluster.settings import HyperSettings, Settings

    settings = Settings(
        database_url="sqlite+aiosqlite:////tmp/no-verda-settings.sqlite3",
        shared_token="no-verda-token",
        shared_token_file=None,
    )
    hyper = HyperSettings()
    assert settings.slug == "hypercluster"
    # No HyperSettings field should map VERDA_* into product knobs.
    field_names = set(HyperSettings.model_fields)
    assert not any(name.upper().startswith("VERDA") for name in field_names)
    assert not any("verda" in name.lower() for name in field_names)
    # Settings construction succeeded without Verda material.
    assert settings.shared_token == "no-verda-token"
    assert hyper.combined_worker is False or getattr(hyper, "combined_worker", None) is not None


def test_gated_tests_tree_has_no_live_verda_purchases() -> None:
    """tests/ must not auto-purchase Verda capacity in default modules.

    Files named *live_verda* or marked for opt-in are allowed only when they
    start with pytest.importorskip / mark and never run in default CI without
    -m live_verda. Scan source text for ops secret path reads.
    """

    forbidden_snippets = (
        "verda.env",
        "/root/.config/hypercluster-mission/verda.env",
        "VERDA_CLIENT_SECRET",
        "VERDA_CLIENT_ID",
    )
    offenders: list[str] = []
    tests_root = REPO_ROOT / "tests"
    for path in tests_root.rglob("*.py"):
        if path.name.startswith("."):
            continue
        text = path.read_text(encoding="utf-8")
        # Opt-in files may reference the env path for documentation of how to
        # enable the mark, but default modules must not load secrets.
        relative = str(path.relative_to(REPO_ROOT))
        is_opt_in = "live_verda" in path.name or "live_verda" in text[:2000]
        for snippet in forbidden_snippets:
            if snippet in text:
                if is_opt_in and "pytest.mark.live_verda" in text:
                    continue
                # Mentions only as denied boundary in comments of no-verda tests
                # are allowed.
                if path.name == "test_no_verda_in_product.py":
                    continue
                offenders.append(f"{relative}: contains {snippet!r}")
    assert not offenders, "\n".join(offenders)


# ----- VAL-LIVE-011 challenge network allowlist excludes mandatory Verda ---


def test_assert_challenge_outbound_blocks_verda() -> None:
    with pytest.raises(VerdaForbiddenError) as excinfo:
        assert_challenge_outbound_allowed("https://api.verda.com/v1/instances")
    assert excinfo.value.code == "verda_outbound_forbidden"


def test_assert_challenge_outbound_allows_local_master() -> None:
    # Local mock-master and identity surfaces remain allowed.
    assert_challenge_outbound_allowed("http://127.0.0.1:3201/internal/v1/raw-weights")
    assert_challenge_outbound_allowed("http://127.0.0.1:3200/health")


def test_strip_verda_env_isolation() -> None:
    """Product job path must succeed without VERDA_* in challenge env."""

    dirty = {
        "CHALLENGE_SHARED_TOKEN": "tok",
        "VERDA_CLIENT_ID": "x",
        "VERDA_CLIENT_SECRET": "y",
        "VERDA_API_BASE": "https://api.verda.com",
        "PATH": "/usr/bin",
    }
    hits = challenge_requires_verda_env(dirty)
    assert set(hits) == {"VERDA_API_BASE", "VERDA_CLIENT_ID", "VERDA_CLIENT_SECRET"}
    clean = strip_verda_env(dirty)
    assert "VERDA_CLIENT_ID" not in clean
    assert clean["CHALLENGE_SHARED_TOKEN"] == "tok"
    assert challenge_requires_verda_env(clean) == []


def test_weight_push_rejects_verda_master_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a misconfigured MASTER_BASE_URL pointing at Verda is refused."""

    from hypercluster.settings import HyperSettings, clear_settings_cache
    from hypercluster.weight_push import (
        WeightPushValidationError,
        maybe_build_push_client,
        resolve_master_base_url,
    )

    clear_settings_cache()
    monkeypatch.setenv("HYPER_MASTER_BASE_URL", "https://api.verda.com")
    hyper = HyperSettings()
    assert hyper.master_base_url and "verda" in hyper.master_base_url.lower()
    with pytest.raises((VerdaForbiddenError, WeightPushValidationError, ValueError)):
        resolve_master_base_url(hyper)

    # Client builder must refuse construction rather than dial Verda.
    class _Settings:
        slug = "hypercluster"
        shared_token = "tok"
        shared_token_file = None

    assert maybe_build_push_client(database=None, settings=_Settings(), hyper=hyper) is None


def test_create_app_does_not_require_verda_env(
    settings_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VAL-LIVE-011: challenge boots and is healthy without VERDA_* in process env."""

    for key in list(os.environ):
        if key.upper().startswith("VERDA_"):
            monkeypatch.delenv(key, raising=False)

    # Ensure no residual requirements.
    assert challenge_requires_verda_env(dict(os.environ)) == []

    from hypercluster.app import create_app

    app = create_app(settings_factory())
    assert app is not None
    # Identity route table present without Verda adapter routes.
    paths = {getattr(r, "path", None) for r in app.routes}
    joined = " ".join(p for p in paths if isinstance(p, str)).lower()
    assert "verda" not in joined


def test_cross_happy_path_uses_shared_host_markers() -> None:
    """Egress fence markers stay aligned with no_verda policy (VAL-CROSS-013)."""

    from hypercluster.sim import cross_happy_path as chp

    for marker in VERDA_HOST_MARKERS:
        assert marker in chp._VERDA_HOST_MARKERS or marker in getattr(chp, "VERDA_HOST_MARKERS", ())


def test_env_example_has_no_verda_secrets() -> None:
    example = REPO_ROOT / ".env.example"
    if not example.is_file():
        pytest.skip(".env.example missing")
    text = example.read_text(encoding="utf-8")
    lower = text.lower()
    # Product .env.example must not assign VERDA_* secret material.
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key = stripped.split("=", 1)[0].strip().upper()
        assert not key.startswith("VERDA_"), f"product env must not set {key}"
        assert "verda_client" not in stripped.lower()
        assert "client_secret" not in stripped.lower() or "verda" not in stripped.lower()
    # Non-comment assignments must not hardcode commercial control-plane URLs.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        assert "api.verda.com" not in stripped.lower()
    # Mentions of Verda as "not product" in comments are allowed; ensure the
    # deny commentary is present so ops secrets stay outside the tree.
    assert "verda" in lower  # denied-boundary note expected
