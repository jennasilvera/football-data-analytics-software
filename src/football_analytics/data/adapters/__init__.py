"""Adapters that translate source-specific data into canonical observations."""

from football_analytics.data.adapters.tabular import (
    TabularMatchColumns,
    TabularSourcePolicy,
    build_match_observations,
)

__all__ = [
    "TabularMatchColumns",
    "TabularSourcePolicy",
    "build_match_observations",
]
