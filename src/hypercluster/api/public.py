"""Public challenge routes (proxied by Base master when registered)."""

from __future__ import annotations

from fastapi import APIRouter

# Empty scaffold router — domain routes land in later milestones.
# Identity routes (`/health`, `/ready`, `/version`) are installed by
# `create_challenge_app` and are not registered here.
router = APIRouter()

__all__ = ["router"]
