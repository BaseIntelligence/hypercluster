"""Secret redaction and output caps for probe evidence (VAL-GPU-031).

Private key PEM never appears in evidence JSON / logs. By design only
``key_fingerprint`` and key_ref kind+name surface publicly.
"""

from __future__ import annotations

import re
from typing import Any

# PEM / key block patterns (OpenSSH, PKCS#8, RSA/EC, etc.).
_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN[^-]*PRIVATE KEY-----.*?-----END[^-]*PRIVATE KEY-----",
    re.DOTALL | re.IGNORECASE,
)
_PEM_BLOCK_SSH_RE = re.compile(
    r"-----BEGIN OPENSSH PRIVATE KEY-----.*?-----END OPENSSH PRIVATE KEY-----",
    re.DOTALL | re.IGNORECASE,
)
_SINGLE_LINE_BEGIN = re.compile(r"-----BEGIN[^\n]*PRIVATE KEY-----", re.IGNORECASE)
_PASSWORD_ASSIGN = re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\s*[=:]\s*\S+")

# Keys that must never appear in evidence JSON (case-insensitive match on key names).
_FORBIDDEN_KEY_SUBSTR = (
    "private_key",
    "privatekey",
    "key_pem",
    "ssh_key_pem",
    "pem_data",
    "id_rsa",
    "id_ed25519_private",
)

OUTPUT_CAP_DEFAULT = 64 * 1024  # 64 KiB
_TRUNC_MARK = "\n…[truncated]…"


def contains_private_key_material(text: str | None) -> bool:
    if not text:
        return False
    if _PEM_BLOCK_RE.search(text) or _PEM_BLOCK_SSH_RE.search(text):
        return True
    if _SINGLE_LINE_BEGIN.search(text):
        return True
    upper = text.upper()
    return "BEGIN" in upper and "PRIVATE KEY" in upper


def redact_text(text: str | None) -> str:
    """Strip PEM blocks and password-like assignments from a string."""

    if not text:
        return ""
    out = _PEM_BLOCK_SSH_RE.sub("[REDACTED_SECRET]", text)
    out = _PEM_BLOCK_RE.sub("[REDACTED_SECRET]", out)
    out = _SINGLE_LINE_BEGIN.sub("[REDACTED_SECRET_HEADER]", out)
    out = _PASSWORD_ASSIGN.sub(r"\1=[REDACTED]", out)
    # Defense in depth if block regex missed quasi-PEM fragments.
    if "PRIVATE KEY" in out.upper() and "BEGIN" in out.upper():
        # Scrub residual lines containing BEGIN/END PRIVATE KEY
        lines = []
        for line in out.splitlines(keepends=True):
            u = line.upper()
            if "PRIVATE KEY" in u and ("BEGIN" in u or "END" in u):
                lines.append("[REDACTED_SECRET_LINE]\n")
            else:
                lines.append(line)
        out = "".join(lines)
    return out


def redact_secrets(text: str | None) -> str:
    """Alias for :func:`redact_text` (stability for callers/tests)."""

    return redact_text(text)


def truncate_output(text: str | None, max_bytes: int = OUTPUT_CAP_DEFAULT) -> str:
    """Cap stdout/stderr size (UTF-8 aware). Always appends a marker when cut."""

    if not text:
        return ""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text
    mark = _TRUNC_MARK.encode("utf-8")
    keep = max(0, max_bytes - len(mark))
    clipped = raw[:keep]
    # Avoid splitting mid-codepoint
    decoded = clipped.decode("utf-8", errors="ignore")
    return decoded + _TRUNC_MARK


def sanitize_output(
    text: str | None,
    *,
    max_bytes: int = OUTPUT_CAP_DEFAULT,
) -> str:
    return truncate_output(redact_text(text), max_bytes=max_bytes)


def _key_forbidden(name: str) -> bool:
    lower = name.lower().replace("-", "_")
    if "pem" in lower and "fingerprint" not in lower:
        return True
    return any(tok in lower for tok in _FORBIDDEN_KEY_SUBSTR)


def redact_mapping(obj: Any) -> Any:
    """Recursively scrub mappings/lists for evidence storage."""

    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if _key_forbidden(key):
                continue  # drop private-key-shaped keys entirely
            if isinstance(v, str):
                cleaned[key] = sanitize_output(v)
            else:
                cleaned[key] = redact_mapping(v)
        return cleaned
    if isinstance(obj, list):
        return [redact_mapping(item) for item in obj]
    if isinstance(obj, str):
        return sanitize_output(obj)
    return obj


def scan_for_private_key_leak(obj: Any) -> list[str]:
    """Return human-readable leak paths (empty = clean)."""

    hits: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{path}.{k}" if path else str(k)
                if _key_forbidden(str(k)):
                    hits.append(p)
                walk(v, p)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")
        elif isinstance(node, str):
            if contains_private_key_material(node):
                hits.append(path or "<root>")

    walk(obj, "")
    return hits


__all__ = [
    "OUTPUT_CAP_DEFAULT",
    "contains_private_key_material",
    "redact_mapping",
    "redact_secrets",
    "redact_text",
    "sanitize_output",
    "scan_for_private_key_leak",
    "truncate_output",
]
