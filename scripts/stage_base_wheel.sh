#!/usr/bin/env bash
# Stage the Base SDK wheel into docker/vendor/ for offline-friendly image builds.
#
# Prefer the release URL from pyproject (same pin Docker falls back to).
# Optional override: BASE_WHEEL_URL or BASE_WHEEL_FILE.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT}/docker/vendor"
PYPROJECT="${ROOT}/pyproject.toml"

mkdir -p "${VENDOR_DIR}"

if [[ -n "${BASE_WHEEL_FILE:-}" ]]; then
  if [[ ! -f "${BASE_WHEEL_FILE}" ]]; then
    echo "stage_base_wheel: BASE_WHEEL_FILE not found: ${BASE_WHEEL_FILE}" >&2
    exit 1
  fi
  cp -f "${BASE_WHEEL_FILE}" "${VENDOR_DIR}/"
  echo "stage_base_wheel: copied ${BASE_WHEEL_FILE} -> ${VENDOR_DIR}/"
  ls -la "${VENDOR_DIR}"/base-*.whl
  exit 0
fi

URL="${BASE_WHEEL_URL:-}"
if [[ -z "${URL}" ]]; then
  # Parse ``base @ https://...whl#sha256=...`` from pyproject dependencies.
  URL="$(
    python3 - <<'PY' "${PYPROJECT}"
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'base\s*@\s*(https://[^\s"\']+\.whl(?:#[^\s"\']+)?)',
    text,
)
if not match:
    raise SystemExit("stage_base_wheel: could not find base wheel URL in pyproject.toml")
print(match.group(1))
PY
  )"
fi

# Strip URL fragment for the download name; keep full URL (hash) for curl.
URL_NO_FRAG="${URL%%#*}"
BASENAME="$(basename "${URL_NO_FRAG}")"
DEST="${VENDOR_DIR}/${BASENAME}"

echo "stage_base_wheel: downloading ${URL_NO_FRAG}"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL --retry 3 --retry-delay 2 -o "${DEST}" "${URL_NO_FRAG}"
elif command -v wget >/dev/null 2>&1; then
  wget -q -O "${DEST}" "${URL_NO_FRAG}"
else
  echo "stage_base_wheel: need curl or wget" >&2
  exit 1
fi

# Optional integrity check when fragment carries sha256=
if [[ "${URL}" == *"#sha256="* ]]; then
  EXPECTED="${URL##*#sha256=}"
  if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL="$(sha256sum "${DEST}" | awk '{print $1}')"
    if [[ "${ACTUAL}" != "${EXPECTED}" ]]; then
      echo "stage_base_wheel: sha256 mismatch for ${BASENAME}" >&2
      echo "  expected: ${EXPECTED}" >&2
      echo "  actual:   ${ACTUAL}" >&2
      rm -f "${DEST}"
      exit 1
    fi
  fi
fi

echo "stage_base_wheel: staged ${DEST}"
ls -la "${DEST}"
