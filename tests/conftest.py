"""Pytest configuration: add project root to sys.path and common fixtures."""

from __future__ import annotations

import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from src.world_generator import Omega  # noqa: E402


DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "default_world.yaml",
)


@pytest.fixture(scope="session")
def default_omega() -> Omega:
    """Omega loaded from config/default_world.yaml (base parameter values)."""
    return Omega.load(DEFAULT_CONFIG)
