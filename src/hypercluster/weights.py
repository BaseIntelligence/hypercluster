"""Raw hotkey weights for Base master aggregation.

Hypercluster never calls `set_weights` and never opens master Postgres.
M6 will replace this stub with the four-factor scoring product.
"""

from __future__ import annotations


async def get_weights() -> dict[str, float]:
    """Return raw hotkey → finite non-negative floats (empty until scored)."""

    return {}


__all__ = ["get_weights"]
