"""Model-agnostic temporal evaluation contracts."""

from football_analytics.evaluation.splits import (
    ExpandingWindowPolicy,
    RollingWindowPolicy,
    SkippedTemporalFold,
    TemporalBacktestFold,
    TemporalFoldBuildReport,
    TemporalFoldSkipReason,
    build_expanding_window_folds,
    build_rolling_window_folds,
)

__all__ = [
    "ExpandingWindowPolicy",
    "RollingWindowPolicy",
    "SkippedTemporalFold",
    "TemporalBacktestFold",
    "TemporalFoldBuildReport",
    "TemporalFoldSkipReason",
    "build_expanding_window_folds",
    "build_rolling_window_folds",
]
