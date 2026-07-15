"""GPU model normalize + VRAM honesty windows (VAL-GPU-014 / VAL-GPU-023).

Maps catalog aliases (e.g. ``1V100.6V``, merchant names) onto a stable family
key so claimed vs measured SKU windows match without exact-string equality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Order matters for contains-match: prefer more specific families first.
_FAMILY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("h100", re.compile(r"\bh100\b|hopper.?h100", re.I)),
    ("h200", re.compile(r"\bh200\b", re.I)),
    ("a100", re.compile(r"\ba100\b|ampere.?a100", re.I)),
    ("a10", re.compile(r"\ba10\b(?!\d)|nvidia a10\b", re.I)),
    ("a40", re.compile(r"\ba40\b", re.I)),
    ("a6000", re.compile(r"\ba6000\b|rtx.?a6000", re.I)),
    ("l40s", re.compile(r"\bl40s\b", re.I)),
    ("l40", re.compile(r"\bl40\b(?!s)", re.I)),
    ("l4", re.compile(r"\bl4\b(?!\d)|nvidia l4\b", re.I)),
    ("v100", re.compile(r"\bv100\b|tesla.?v100|1v100", re.I)),
    ("t4", re.compile(r"\bt4\b(?!\d)|tesla.?t4", re.I)),
    ("rtx4090", re.compile(r"4090|rtx.?4090", re.I)),
    ("rtx3090", re.compile(r"3090|rtx.?3090", re.I)),
]


@dataclass(frozen=True, slots=True)
class VramWindow:
    """Inclusive float memory.total window in MiB for a family."""

    family: str
    min_mb: int
    max_mb: int

    def contains(self, memory_mb: int | float | None) -> bool:
        if memory_mb is None:
            return False
        try:
            value = float(memory_mb)
        except (TypeError, ValueError):
            return False
        return float(self.min_mb) <= value <= float(self.max_mb)


@dataclass(frozen=True, slots=True)
class GpuFamilySpec:
    family: str
    vram: VramWindow
    aliases: frozenset[str]


# VRAM windows are wide enough for SXM/PCIe manufacturer variance (± ~8%).
_VRAM: dict[str, VramWindow] = {
    "v100": VramWindow("v100", min_mb=14_000, max_mb=34_000),  # 16/32 GB
    "a100": VramWindow("a100", min_mb=36_000, max_mb=86_000),  # 40/80 GB
    "h100": VramWindow("h100", min_mb=70_000, max_mb=100_000),
    "h200": VramWindow("h200", min_mb=130_000, max_mb=155_000),
    "a10": VramWindow("a10", min_mb=20_000, max_mb=26_000),
    "a40": VramWindow("a40", min_mb=42_000, max_mb=52_000),
    "a6000": VramWindow("a6000", min_mb=44_000, max_mb=52_000),
    "l40": VramWindow("l40", min_mb=42_000, max_mb=52_000),
    "l40s": VramWindow("l40s", min_mb=42_000, max_mb=52_000),
    "l4": VramWindow("l4", min_mb=20_000, max_mb=26_000),
    "t4": VramWindow("t4", min_mb=14_000, max_mb=18_000),
    "rtx4090": VramWindow("rtx4090", min_mb=22_000, max_mb=26_000),
    "rtx3090": VramWindow("rtx3090", min_mb=22_000, max_mb=26_000),
}

# Explicit catalog aliases (merchant / Verda / short-codes) → family.
_ALIAS_TABLE: dict[str, str] = {
    "1v100.6v": "v100",
    "1v100.16g": "v100",
    "1v100.32g": "v100",
    "tesla v100": "v100",
    "tesla-v100": "v100",
    "v100": "v100",
    "v100-sxm2-16gb": "v100",
    "tesla v100-sxm2-16gb": "v100",
    "tesla v100-sxm2-32gb": "v100",
    "1a100.40g": "a100",
    "1a100.80g": "a100",
    "a100": "a100",
    "a100-sxm4-40gb": "a100",
    "a100-sxm4-80gb": "a100",
    "nvidia a100-sxm4-40gb": "a100",
    "nvidia a100-sxm4-80gb": "a100",
    "1h100.80g": "h100",
    "h100": "h100",
    "h100-sxm": "h100",
    "nvidia h100": "h100",
    "h200": "h200",
    "t4": "t4",
    "tesla t4": "t4",
    "l4": "l4",
    "l40": "l40",
    "l40s": "l40s",
    "a10": "a10",
    "a40": "a40",
    "rtx a6000": "a6000",
    "rtx4090": "rtx4090",
    "rtx 4090": "rtx4090",
    "geforce rtx 4090": "rtx4090",
    "rtx3090": "rtx3090",
    "rtx 3090": "rtx3090",
}


def _clean(name: str) -> str:
    return " ".join((name or "").strip().lower().replace("_", " ").replace("|", " ").split())


def normalize_gpu_model(name: str | None) -> str | None:
    """Map free-form / catalog GPU name to a family key (or None if unknown)."""

    if name is None:
        return None
    cleaned = _clean(name)
    if not cleaned:
        return None
    if cleaned in _ALIAS_TABLE:
        return _ALIAS_TABLE[cleaned]
    # Direct alias table also tries compressions without spaces/punctuation.
    compact = re.sub(r"[^a-z0-9.]", "", cleaned)
    if compact in _ALIAS_TABLE:
        return _ALIAS_TABLE[compact]
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(cleaned):
            return family
    return None


def family_for_name(name: str | None) -> str | None:
    return normalize_gpu_model(name)


def models_match(claimed: str | None, measured: str | None) -> bool:
    """True when claimed catalog alias and measured smi name share a family."""

    c = normalize_gpu_model(claimed)
    m = normalize_gpu_model(measured)
    if c is None or m is None:
        return False
    return c == m


def lookup_vram_window(name_or_family: str | None) -> VramWindow | None:
    family = normalize_gpu_model(name_or_family) or (
        (name_or_family or "").strip().lower() or None
    )
    if family is None:
        return None
    return _VRAM.get(family)


def known_families() -> frozenset[str]:
    return frozenset(_VRAM.keys())


__all__ = [
    "GpuFamilySpec",
    "VramWindow",
    "family_for_name",
    "known_families",
    "lookup_vram_window",
    "models_match",
    "normalize_gpu_model",
]
