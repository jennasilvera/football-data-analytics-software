from __future__ import annotations

import math

import pytest

from football_analytics.domain import MatchOutcome
from football_analytics.evaluation import (
    OutcomeProbabilities,
    ScoredPrediction,
    accuracy,
    evaluate_predictions,
    log_loss,
    multiclass_brier_score,
    ranked_probability_score,
)


def test_outcome_probabilities_require_valid_distribution() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        OutcomeProbabilities(
            home_win=0.5,
            draw=0.3,
            away_win=0.3,
        )

    with pytest.raises(ValueError, match="between 0 and 1"):
        OutcomeProbabilities(
            home_win=-0.1,
            draw=0.4,
            away_win=0.7,
        )

    with pytest.raises(ValueError, match="finite"):
        OutcomeProbabilities(
            home_win=float("nan"),
            draw=0.5,
            away_win=0.5,
        )


def test_predicted_outcome_uses_canonical_tie_order() -> None:
    probabilities = OutcomeProbabilities(
        home_win=0.4,
        draw=0.4,
        away_win=0.2,
    )

    assert probabilities.predicted_outcome is MatchOutcome.HOME_WIN


def test_perfect_prediction_has_zero_probability_error() -> None:
    predictions = [
        ScoredPrediction(
            actual=MatchOutcome.HOME_WIN,
            probabilities=OutcomeProbabilities(
                home_win=1.0,
                draw=0.0,
                away_win=0.0,
            ),
        )
    ]

    assert accuracy(predictions) == pytest.approx(1.0)
    assert log_loss(predictions) == pytest.approx(0.0)
    assert multiclass_brier_score(predictions) == pytest.approx(0.0)
    assert ranked_probability_score(predictions) == pytest.approx(0.0)


def test_uniform_draw_forecast_has_known_scores() -> None:
    predictions = [
        ScoredPrediction(
            actual=MatchOutcome.DRAW,
            probabilities=OutcomeProbabilities(
                home_win=1.0 / 3.0,
                draw=1.0 / 3.0,
                away_win=1.0 / 3.0,
            ),
        )
    ]

    assert log_loss(predictions) == pytest.approx(math.log(3.0))
    assert multiclass_brier_score(predictions) == pytest.approx(2.0 / 3.0)
    assert ranked_probability_score(predictions) == pytest.approx(1.0 / 9.0)


def test_confident_opposite_extreme_has_maximum_brier_and_rps() -> None:
    predictions = [
        ScoredPrediction(
            actual=MatchOutcome.AWAY_WIN,
            probabilities=OutcomeProbabilities(
                home_win=1.0,
                draw=0.0,
                away_win=0.0,
            ),
        )
    ]

    assert accuracy(predictions) == pytest.approx(0.0)
    assert multiclass_brier_score(predictions) == pytest.approx(2.0)
    assert ranked_probability_score(predictions) == pytest.approx(1.0)
    assert math.isfinite(log_loss(predictions))


def test_evaluation_metrics_aggregate_predictions() -> None:
    predictions = [
        ScoredPrediction(
            actual=MatchOutcome.HOME_WIN,
            probabilities=OutcomeProbabilities(
                home_win=0.7,
                draw=0.2,
                away_win=0.1,
            ),
        ),
        ScoredPrediction(
            actual=MatchOutcome.DRAW,
            probabilities=OutcomeProbabilities(
                home_win=0.2,
                draw=0.6,
                away_win=0.2,
            ),
        ),
    ]

    metrics = evaluate_predictions(predictions)

    assert metrics.n_predictions == 2
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.log_loss == pytest.approx(
        (-math.log(0.7) - math.log(0.6)) / 2.0
    )


@pytest.mark.parametrize(
    "metric",
    [
        accuracy,
        log_loss,
        multiclass_brier_score,
        ranked_probability_score,
        evaluate_predictions,
    ],
)
def test_evaluation_requires_non_empty_predictions(metric) -> None:
    with pytest.raises(ValueError, match="At least one prediction"):
        metric([])
