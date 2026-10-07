"""Model-agnostic probability and evaluation contracts."""

from football_analytics.evaluation.metrics import (
    EvaluationMetrics,
    ScoredPrediction,
    accuracy,
    evaluate_predictions,
    log_loss,
    multiclass_brier_score,
    ranked_probability_score,
)
from football_analytics.evaluation.probabilities import OutcomeProbabilities
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
    "EvaluationMetrics",
    "ExpandingWindowPolicy",
    "OutcomeProbabilities",
    "RollingWindowPolicy",
    "ScoredPrediction",
    "SkippedTemporalFold",
    "TemporalBacktestFold",
    "TemporalFoldBuildReport",
    "TemporalFoldSkipReason",
    "accuracy",
    "build_expanding_window_folds",
    "build_rolling_window_folds",
    "evaluate_predictions",
    "log_loss",
    "multiclass_brier_score",
    "ranked_probability_score",
]
