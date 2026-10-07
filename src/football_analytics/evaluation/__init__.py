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

__all__ = [
    "EvaluationMetrics",
    "OutcomeProbabilities",
    "ScoredPrediction",
    "accuracy",
    "evaluate_predictions",
    "log_loss",
    "multiclass_brier_score",
    "ranked_probability_score",
]
