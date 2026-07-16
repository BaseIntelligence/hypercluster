"""API routers for the hypercluster challenge."""

from __future__ import annotations

from hypercluster.api.prices import router as prices_router
from hypercluster.api.public import router

__all__ = ["prices_router", "router"]
