"""Adapters for open two-arm bandit datasets used in transfer studies."""

from .adapters import ADAPTERS, build_dataset
from .schema import CANONICAL_REQUIRED_COLUMNS, validate_canonical_table
from .sources import SOURCES

__all__ = [
    "ADAPTERS",
    "CANONICAL_REQUIRED_COLUMNS",
    "SOURCES",
    "build_dataset",
    "validate_canonical_table",
]
