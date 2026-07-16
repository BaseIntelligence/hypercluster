#!/usr/bin/env python3
"""CI / ops guard: fail if product tree depends on Verda (VAL-LIVE-001/002).

Thin wrapper around ``hypercluster.no_verda`` so GitHub Actions can run a
dedicated no-verda fence job without pulling the full pytest suite.

Exit codes:
  0 — product + docs audit clean
  1 — findings or module load error
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    # Prefer installed package; fall back to src layout when run pre-install.
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    try:
        from hypercluster.no_verda import run_docs_verda_audit, run_product_verda_audit
    except ImportError as exc:  # pragma: no cover - environment misconfig
        print(f"check_no_verda: import failed: {exc}", file=sys.stderr)
        return 1

    product = run_product_verda_audit(REPO_ROOT)
    docs = run_docs_verda_audit(REPO_ROOT)

    # Docs: only fail on forced-Verda language (same policy as unit tests).
    docs_bad = [f for f in docs.findings if "forces Verda" in f.detail or "matched" in f.detail]

    ok = product.ok and not docs_bad
    if product.ok:
        print("check_no_verda: product audit clean")
    else:
        print("check_no_verda: product audit FAILED", file=sys.stderr)
        for line in product.summary_lines():
            print(f"  {line}", file=sys.stderr)

    if docs_bad:
        print("check_no_verda: docs audit FAILED", file=sys.stderr)
        for finding in docs_bad:
            print(f"  {finding.path}: {finding.detail}", file=sys.stderr)
    else:
        print("check_no_verda: docs audit clean")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
