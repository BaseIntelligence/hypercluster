"""Mission local port band discipline (3200–3299).

Bare-metal smoke and local uvicorn defaults must stay inside this band
(VAL-SCAF-030). Docker maps host ports in the same band (e.g. 3250→8000).
"""

from __future__ import annotations

MIN_MISSION_PORT = 3200
MAX_MISSION_PORT = 3299
DEFAULT_BAREMETAL_PORT = 3200
DEFAULT_MOCK_MASTER_PORT = 3201
DEFAULT_DOCKER_HOST_PORT = 3250


def mission_port_band() -> tuple[int, int]:
    """Return inclusive (min, max) mission port band."""

    return (MIN_MISSION_PORT, MAX_MISSION_PORT)


def is_mission_port(port: int) -> bool:
    """True when *port* is inside the mission band [3200, 3299]."""

    return MIN_MISSION_PORT <= int(port) <= MAX_MISSION_PORT


def assert_mission_port(port: int) -> int:
    """Validate *port* is in band; raise ValueError otherwise.

    Returns the validated port for fluent use.
    """

    value = int(port)
    if not is_mission_port(value):
        raise ValueError(
            f"port {value} outside mission band {MIN_MISSION_PORT}–{MAX_MISSION_PORT} "
            f"(bare-metal smoke uses 3200; docker host map e.g. 3250; "
            f"do not bind master 3180 or foreign services)"
        )
    return value


def parse_port_from_url(url: str) -> int | None:
    """Extract explicit TCP port from an http(s) URL, or None if default-ish."""

    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.port is not None:
        return int(parsed.port)
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return None


__all__ = [
    "DEFAULT_BAREMETAL_PORT",
    "DEFAULT_DOCKER_HOST_PORT",
    "DEFAULT_MOCK_MASTER_PORT",
    "MAX_MISSION_PORT",
    "MIN_MISSION_PORT",
    "assert_mission_port",
    "is_mission_port",
    "mission_port_band",
    "parse_port_from_url",
]
