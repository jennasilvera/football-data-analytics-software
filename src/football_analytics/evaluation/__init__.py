"""Model-agnostic probability and evaluation contracts."""

from football_analytics.evaluation.backtest import (
    BacktestFoldResult,
    BacktestPrediction,
    ModelTrainer,
    TemporalBacktestResult,
    run_temporal_backtest,
)
from football_analytics.evaluation.calibration import (
    CalibrationBin,
    CalibrationReport,
    OutcomeCalibrationReport,
    build_calibration_report,
)
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
    "BacktestFoldResult",
    "BacktestPrediction",
    "CalibrationBin",
    "CalibrationReport",
    "EvaluationMetrics",
    "ExpandingWindowPolicy",
    "ModelTrainer",
    "OutcomeCalibrationReport",
    "OutcomeProbabilities",
    "RollingWindowPolicy",
    "ScoredPrediction",
    "SkippedTemporalFold",
    "TemporalBacktestFold",
    "TemporalBacktestResult",
    "TemporalFoldBuildReport",
    "TemporalFoldSkipReason",
    "accuracy",
    "build_calibration_report",
    "build_expanding_window_folds",
    "build_rolling_window_folds",
    "evaluate_predictions",
    "log_loss",
    "multiclass_brier_score",
    "ranked_probability_score",
    "run_temporal_backtest",
]
