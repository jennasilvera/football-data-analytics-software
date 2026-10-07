"""Adapters that translate source-specific data into canonical observations."""

from football_analytics.data.adapters.legacy import (
    LEGACY_RESULTS_COLUMNS,
    build_legacy_mens_results_observations,
)
from football_analytics.data.adapters.tabular import (
    TabularMatchColumns,
    TabularSourcePolicy,
    build_match_observations,
)

__all__ = [
    "LEGACY_RESULTS_COLUMNS",
    "TabularMatchColumns",
    "TabularSourcePolicy",
    "build_legacy_mens_results_observations",
    "build_match_observations",
]
