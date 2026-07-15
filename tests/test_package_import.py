"""Smoke imports for scaffold package layout."""

from __future__ import annotations


def test_package_version() -> None:
    import hypercluster

    assert hypercluster.__version__ == "0.1.0"


def test_create_app_importable() -> None:
    from hypercluster.app import create_app

    assert callable(create_app)
