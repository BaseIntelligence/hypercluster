"""Hard product boundary: Verda is never part of the hypercluster package.

Mission M8 and architecture §11: Verda (api.verda.com / SDK / OAuth) is
**external ops QA only**. Product code, default CI, and challenge outbound
paths must never require Verda credentials or hosts.

Fulfills VAL-LIVE-001, VAL-LIVE-010, VAL-LIVE-011. Docs boundary lives in
user and tests (VAL-LIVE-002).
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

# Host substrings that identify Verda (or obvious misspellings used in tests).
# Shared with pure-sim egress traces so VAL-CROSS-013 and VAL-LIVE stay aligned.
VERDA_HOST_MARKERS: tuple[str, ...] = (
    "verda.com",
    "api.verda",
    "verda.cloud",
    "verda.io",
)

# Python top-level module names that must never be imported by product code.
FORBIDDEN_MODULE_PREFIXES: tuple[str, ...] = (
    "verda",
    "verda_sdk",
    "verda_client",
    "verdaapi",
)

# Dependency names that must not appear in product metadata / lockfiles.
FORBIDDEN_DEPENDENCY_NAMES: frozenset[str] = frozenset(
    {
        "verda",
        "verda-sdk",
        "verda_sdk",
        "verda-client",
        "verda_client",
        "pyverda",
    }
)

# Env vars that belong only to external ops (never product process requirements).
VERDA_ENV_PREFIXES: tuple[str, ...] = ("VERDA_",)

# User-facing phrases that would force miners through a Verda signup (VAL-LIVE-002).
REQUIRED_VERDA_DOC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brequire[sd]?\s+a?\s*verda\s+account\b", re.I),
    re.compile(r"\bmust\s+create\s+(a\s+)?verda\s+account\b", re.I),
    re.compile(r"\bcreate\s+(a\s+)?verda\s+account\b.*\b(required|must|mandatory)\b", re.I),
    re.compile(r"\b(mandatory|required)\b.*\bverda\s+account\b", re.I),
    re.compile(r"\bsign\s*up\s+(for|on)\s+verda\b.*\b(required|must|before\s+mining)\b", re.I),
    re.compile(r"\bminers?\s+must\s+.*verda\b", re.I),
)

_API_VERDA_LITERAL = re.compile(r"api\.verda\.com", re.I)
_IMPORT_VERDA_LITERAL = re.compile(
    r"^\s*(?:from|import)\s+verda(?:[.\s]|$)",
    re.I | re.M,
)


class VerdaForbiddenError(RuntimeError):
    """Raised when product code would contact or depend on Verda."""

    def __init__(self, message: str, *, code: str = "verda_forbidden") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class AuditFinding:
    """One static inventory hit (path + detail)."""

    path: str
    detail: str


@dataclass(slots=True)
class AuditReport:
    """Aggregated product-tree audit for VAL-LIVE-001 / docs / CI hygiene."""

    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def add(self, path: str | Path, detail: str) -> None:
        self.findings.append(AuditFinding(path=str(path), detail=detail))

    def summary_lines(self) -> list[str]:
        if self.ok:
            return ["no-verda audit: clean"]
        return [f"{f.path}: {f.detail}" for f in self.findings]


def is_verda_host(host_or_url: str) -> bool:
    """True when *host_or_url* matches a Verda control-plane host marker."""

    blob = (host_or_url or "").strip().lower()
    if not blob:
        return False
    # Prefer hostname when a full URL is given.
    parsed = urlparse(blob if "://" in blob else f"//{blob}", scheme="")
    host = (parsed.hostname or "").lower()
    combined = f"{host} {blob}"
    return any(marker in combined for marker in VERDA_HOST_MARKERS)


def assert_challenge_outbound_allowed(url: str) -> None:
    """Fail closed if the challenge process would dial a Verda endpoint.

    Product egress is limited to Base master / mock-master / optional TEE
    verifier / miner-provider SSH inventory. Verda cloud control plane is never
    a challenge dependency (VAL-LIVE-011).
    """

    if is_verda_host(url):
        raise VerdaForbiddenError(
            f"challenge outbound URL is forbidden Verda endpoint: {url!r}",
            code="verda_outbound_forbidden",
        )


def challenge_requires_verda_env(environ: dict[str, str] | None = None) -> list[str]:
    """Return VERDA_* keys present in *environ* that product code must ignore.

    Presence of VERDA_* in the host environment is allowed for ops shells, but
    the challenge process must never *require* them. Callers use this for
    isolation experiments (VAL-LIVE-011): strip list and confirm jobs still run.
    """

    env = environ if environ is not None else dict(os.environ)
    hits: list[str] = []
    for key in env:
        upper = key.upper()
        if any(upper.startswith(prefix) for prefix in VERDA_ENV_PREFIXES):
            hits.append(key)
    return sorted(hits)


def strip_verda_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of *environ* with all VERDA_* keys removed."""

    base = dict(os.environ if environ is None else environ)
    for key in challenge_requires_verda_env(base):
        base.pop(key, None)
    return base


def _iter_product_py_files(package_root: Path) -> Iterable[Path]:
    if not package_root.is_dir():
        return
    for path in sorted(package_root.rglob("*.py")):
        # Skip caches; only first-party product modules.
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        yield path


def _module_is_forbidden(name: str) -> bool:
    root = (name or "").split(".", 1)[0].strip().lower()
    if not root:
        return False
    return root in FORBIDDEN_MODULE_PREFIXES or any(
        root == prefix or root.startswith(f"{prefix}_") for prefix in FORBIDDEN_MODULE_PREFIXES
    )


def audit_product_source_imports(package_root: Path) -> AuditReport:
    """Static AST inventory: no Verda imports and no hard-coded api.verda.com client use.

    Sim egress *markers* matching Verda host substrings are allowed (they fence
    pure-sim tests). Hard-coded product HTTP clients to api.verda.com are not.
    """

    report = AuditReport()
    package_root = package_root.resolve()
    for path in _iter_product_py_files(package_root):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            report.add(path, f"unreadable: {exc}")
            continue
        # AST import scan
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            report.add(path, f"syntax error: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _module_is_forbidden(alias.name):
                        report.add(path, f"forbidden import: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if _module_is_forbidden(mod):
                    report.add(path, f"forbidden import: from {mod} import ...")
        # Literal product client hard-code (allow list/comment fence strings in no_verda
        # and sim egress marker tuples which do not construct I/O clients).
        rel = path.relative_to(package_root) if path.is_relative_to(package_root) else path
        rel_s = str(rel).replace("\\", "/")
        if _API_VERDA_LITERAL.search(source):
            # Allowed only as marker constants / comments for anti-egress fences.
            if not _api_verda_only_as_marker(source, rel_s):
                report.add(path, "hard-coded api.verda.com product client surface")
        if _IMPORT_VERDA_LITERAL.search(source):
            report.add(path, "source text contains import verda")
    return report


def _api_verda_only_as_marker(source: str, rel_s: str) -> bool:
    """True when api.verda appears only as anti-egress marker / docs string."""

    # Policy modules and pure-sim egress fences may mention the host as a deny list.
    allow_paths = {
        "no_verda.py",
        "sim/cross_happy_path.py",
    }
    if rel_s in allow_paths or rel_s.endswith("/no_verda.py"):
        return True
    # Every occurrence must sit inside a tuple/list of markers or a comment/docstring
    # denying Verda — reject assignment to base_url / client constructors.
    for line in source.splitlines():
        stripped = line.strip()
        if "api.verda" not in stripped.lower():
            continue
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        lower = stripped.lower()
        if any(
            token in lower
            for token in (
                "marker",
                "forbidden",
                "verda_host",
                "deny",
                "block",
                "not ",
                "never",
                "exclude",
            )
        ):
            continue
        # Suspicious: looks like a live client base URL.
        if any(
            token in lower
            for token in (
                "base_url",
                "baseurl",
                "client(",
                "httpx.",
                "requests.",
                "session(",
                "oauth",
                "token_url",
            )
        ):
            return False
        # Unclassified occurrence outside known fence modules → fail closed.
        return False
    return True


def _normalize_dep_name(raw: str) -> str:
    # PEP 508 / uv lock names: strip extras and version specs roughly.
    name = raw.strip().lower()
    name = name.split("[", 1)[0]
    name = re.split(r"[@<>=!~\s]", name, maxsplit=1)[0]
    return name.replace("_", "-")


def audit_dependency_manifests(
    *,
    pyproject: Path,
    lockfile: Path | None = None,
) -> AuditReport:
    """Ensure pyproject / lock have no Verda SDK runtime dependency (VAL-LIVE-001)."""

    report = AuditReport()
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        lower = text.lower()
        for forbidden in FORBIDDEN_DEPENDENCY_NAMES:
            # Match whole requirement tokens, not incidental comments about Verda.
            if re.search(
                rf'(?m)^[^#\n]*["\']{_normalize_dep_name(forbidden)}(?:["\'\[@<>=!~\s]|$)',
                lower,
            ) or re.search(
                rf"(?m)^\s*{re.escape(_normalize_dep_name(forbidden))}\s*[=<>!~]",
                lower,
            ):
                report.add(pyproject, f"forbidden dependency name: {forbidden}")
        if "api.verda.com" in lower and "no" not in lower:
            # Soft: only fail if it looks like a package URL index/dependency.
            if "verda.com" in lower and any(
                token in lower for token in ("dependencies", "requires", "index-url")
            ):
                # Comment-only mentions are fine; require a dependency-looking line.
                for line in text.splitlines():
                    lower_line = line.lower()
                    if "verda" in lower_line and not lower_line.strip().startswith("#"):
                        dep_tokens = ("http", "git+", "pypi", "==", "@")
                        if any(token in lower_line for token in dep_tokens):
                            detail = f"suspicious Verda dependency line: {line.strip()}"
                            report.add(pyproject, detail)
    else:
        report.add(pyproject, "pyproject.toml missing")

    if lockfile is not None and lockfile.is_file():
        lock_text = lockfile.read_text(encoding="utf-8").lower()
        for forbidden in FORBIDDEN_DEPENDENCY_NAMES:
            needle = _normalize_dep_name(forbidden)
            # uv.lock package entries look like: name = "foo"
            if re.search(rf'(?m)^\s*name\s*=\s*"{re.escape(needle)}"', lock_text):
                report.add(lockfile, f"forbidden locked package: {forbidden}")
            if "api.verda.com" in lock_text:
                report.add(lockfile, "lockfile references api.verda.com")
    return report


def audit_docs_miner_path(doc_paths: Sequence[Path]) -> AuditReport:
    """Docs must not require a Verda account for miners (VAL-LIVE-002)."""

    report = AuditReport()
    if not doc_paths:
        report.add("<docs>", "no user-facing docs found to audit")
        return report
    found_any = False
    for path in doc_paths:
        if not path.is_file():
            continue
        found_any = True
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            report.add(path, f"unreadable: {exc}")
            continue
        for pattern in REQUIRED_VERDA_DOC_PATTERNS:
            if pattern.search(text):
                report.add(path, f"miner path forces Verda: matched {pattern.pattern!r}")
    if not found_any:
        report.add("<docs>", "no readable user-facing docs")
    return report


def default_repo_root() -> Path:
    """Hypercluster package → repo root (src/hypercluster → ../..)."""

    return Path(__file__).resolve().parents[2]


def run_product_verda_audit(repo_root: Path | None = None) -> AuditReport:
    """Full product-tree dep + import audit (VAL-LIVE-001)."""

    root = (repo_root or default_repo_root()).resolve()
    package = root / "src" / "hypercluster"
    report = AuditReport()
    for sub in (
        audit_product_source_imports(package),
        audit_dependency_manifests(
            pyproject=root / "pyproject.toml",
            lockfile=root / "uv.lock",
        ),
    ):
        report.findings.extend(sub.findings)
    return report


def run_docs_verda_audit(repo_root: Path | None = None) -> AuditReport:
    """Audit README / docs for miner Verda requirements (VAL-LIVE-002).

    Missing user-facing docs is **not** a failure for this assertion: a miner
    path cannot require Verda when no docs force it. Explicit "required Verda
    account" language in present docs is a failure. Wiki-style notes that treat
    Verda as optional external ops remain allowed.
    """

    root = (repo_root or default_repo_root()).resolve()
    candidates: list[Path] = []
    for name in ("README.md", "README.rst", "README.txt"):
        p = root / name
        if p.is_file():
            candidates.append(p)
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        for path in sorted(docs_dir.rglob("*.md")):
            if any(part.startswith(".") for part in path.parts):
                continue
            candidates.append(path)
    if not candidates:
        # No user-facing docs yet → nothing forces a Verda miner signup.
        return AuditReport()
    return audit_docs_miner_path(candidates)


__all__ = [
    "FORBIDDEN_DEPENDENCY_NAMES",
    "FORBIDDEN_MODULE_PREFIXES",
    "VERDA_ENV_PREFIXES",
    "VERDA_HOST_MARKERS",
    "AuditFinding",
    "AuditReport",
    "VerdaForbiddenError",
    "assert_challenge_outbound_allowed",
    "audit_dependency_manifests",
    "audit_docs_miner_path",
    "audit_product_source_imports",
    "challenge_requires_verda_env",
    "default_repo_root",
    "is_verda_host",
    "run_docs_verda_audit",
    "run_product_verda_audit",
    "strip_verda_env",
]
