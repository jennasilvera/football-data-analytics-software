from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from football_analytics.domain import MatchOutcome
from football_analytics.evaluation.probabilities import OutcomeProbabilities


LOG_LOSS_EPSILON = 1e-15


@dataclass(frozen=True, slots=True)
class ScoredPrediction:
    """Actual outcome paired with a validated probability forecast."""

    actual: MatchOutcome
    probabilities: OutcomeProbabilities


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Core probability and classification metrics for one evaluation sample."""

    n_predictions: int
    accuracy: float
    log_loss: float
    multiclass_brier_score: float
    ranked_probability_score: float


def accuracy(predictions: Sequence[ScoredPrediction]) -> float:
    """Return maximum-probability classification accuracy."""

    _require_predictions(predictions)
    correct = sum(
        prediction.probabilities.predicted_outcome is prediction.actual
        for prediction in predictions
    )
    return correct / len(predictions)


def log_loss(predictions: Sequence[ScoredPrediction]) -> float:
    """Return natural-log multiclass cross entropy."""

    _require_predictions(predictions)

    losses = []
    for prediction in predictions:
        probability = prediction.probabilities.probability_for(
            prediction.actual
        )
        clipped = min(max(probability, LOG_LOSS_EPSILON), 1.0)
        losses.append(-math.log(clipped))

    return sum(losses) / len(losses)


def multiclass_brier_score(
    predictions: Sequence[ScoredPrediction],
) -> float:
    """Return legacy-compatible unnormalized multiclass Brier score.

    The score is the per-match sum of squared errors across the three outcome
    classes, averaged across matches. This preserves the convention used by the
    legacy forecasting engine.
    """

    _require_predictions(predictions)

    scores = []
    for prediction in predictions:
        actual = _one_hot(prediction.actual)
        probabilities = prediction.probabilities.as_tuple()
        scores.append(
            sum(
                (forecast - observed) ** 2
                for forecast, observed in zip(
                    probabilities,
                    actual,
                    strict=True,
                )
            )
        )

    return sum(scores) / len(scores)


def ranked_probability_score(
    predictions: Sequence[ScoredPrediction],
) -> float:
    """Return normalized Ranked Probability Score for home/draw/away outcomes.

    The canonical ordinal order is home win, draw, away win. For three classes
    the two cumulative squared errors are averaged, producing a score in [0, 1].
    Lower is better.
    """

    _require_predictions(predictions)

    scores = []
    for prediction in predictions:
        probabilities = prediction.probabilities.as_tuple()
        actual = _one_hot(prediction.actual)

        probability_cumulative = (
            probabilities[0],
            probabilities[0] + probabilities[1],
        )
        actual_cumulative = (
            actual[0],
            actual[0] + actual[1],
        )

        score = sum(
            (forecast - observed) ** 2
            for forecast, observed in zip(
                probability_cumulative,
                actual_cumulative,
                strict=True,
            )
        ) / 2.0
        scores.append(score)

    return sum(scores) / len(scores)


def evaluate_predictions(
    predictions: Sequence[ScoredPrediction],
) -> EvaluationMetrics:
    """Evaluate one immutable collection of probability forecasts."""

    _require_predictions(predictions)

    return EvaluationMetrics(
        n_predictions=len(predictions),
        accuracy=accuracy(predictions),
        log_loss=log_loss(predictions),
        multiclass_brier_score=multiclass_brier_score(predictions),
        ranked_probability_score=ranked_probability_score(predictions),
    )


def _one_hot(outcome: MatchOutcome) -> tuple[float, float, float]:
    if outcome is MatchOutcome.HOME_WIN:
        return (1.0, 0.0, 0.0)
    if outcome is MatchOutcome.DRAW:
        return (0.0, 1.0, 0.0)
    if outcome is MatchOutcome.AWAY_WIN:
        return (0.0, 0.0, 1.0)

    raise ValueError(f"Unsupported match outcome: {outcome}")


def _require_predictions(
    predictions: Sequence[ScoredPrediction],
) -> None:
    if not predictions:
        raise ValueError("At least one prediction is required for evaluation.")
