from __future__ import annotations

import pytest

from football_analytics.domain import MatchOutcome
from football_analytics.evaluation.calibration import build_calibration_report
from football_analytics.evaluation.metrics import ScoredPrediction
from football_analytics.evaluation.probabilities import OutcomeProbabilities


def _prediction(
    *,
    actual: MatchOutcome,
    home: float,
    draw: float,
    away: float,
) -> ScoredPrediction:
    return ScoredPrediction(
        actual=actual,
        probabilities=OutcomeProbabilities(
            home_win=home,
            draw=draw,
            away_win=away,
        ),
    )


def test_perfect_deterministic_forecasts_have_zero_calibration_error() -> None:
    predictions = [
        _prediction(
            actual=MatchOutcome.HOME_WIN,
            home=1.0,
            draw=0.0,
            away=0.0,
        ),
        _prediction(
            actual=MatchOutcome.DRAW,
            home=0.0,
            draw=1.0,
            away=0.0,
        ),
        _prediction(
            actual=MatchOutcome.AWAY_WIN,
            home=0.0,
            draw=0.0,
            away=1.0,
        ),
    ]

    report = build_calibration_report(predictions, n_bins=5)

    assert report.n_predictions == 3
    assert report.macro_expected_calibration_error == pytest.approx(0.0)
    for outcome_report in report.outcomes:
        assert outcome_report.expected_calibration_error == pytest.approx(0.0)
        assert outcome_report.maximum_calibration_error == pytest.approx(0.0)
        assert outcome_report.calibration_bias == pytest.approx(0.0)


def test_known_home_win_bin_reports_expected_error_and_bias() -> None:
    predictions = [
        _prediction(
            actual=MatchOutcome.HOME_WIN,
            home=0.8,
            draw=0.1,
            away=0.1,
        ),
        _prediction(
            actual=MatchOutcome.AWAY_WIN,
            home=0.8,
            draw=0.1,
            away=0.1,
        ),
    ]

    report = build_calibration_report(predictions, n_bins=5)
    home = report.for_outcome(MatchOutcome.HOME_WIN)
    populated = [item for item in home.bins if item.forecast_count]

    assert len(populated) == 1
    assert populated[0].bin_index == 4
    assert populated[0].average_forecast_probability == pytest.approx(0.8)
    assert populated[0].observed_frequency == pytest.approx(0.5)
    assert populated[0].calibration_error == pytest.approx(-0.3)
    assert home.expected_calibration_error == pytest.approx(0.3)
    assert home.maximum_calibration_error == pytest.approx(0.3)
    assert home.calibration_bias == pytest.approx(-0.3)


def test_zero_and_one_probabilities_map_to_boundary_bins() -> None:
    predictions = [
        _prediction(
            actual=MatchOutcome.HOME_WIN,
            home=1.0,
            draw=0.0,
            away=0.0,
        )
    ]

    report = build_calibration_report(predictions, n_bins=10)
    home = report.for_outcome(MatchOutcome.HOME_WIN)
    draw = report.for_outcome(MatchOutcome.DRAW)

    assert home.bins[-1].forecast_count == 1
    assert home.bins[-1].average_forecast_probability == pytest.approx(1.0)
    assert draw.bins[0].forecast_count == 1
    assert draw.bins[0].average_forecast_probability == pytest.approx(0.0)


def test_empty_bins_are_retained_for_stable_reliability_shape() -> None:
    report = build_calibration_report(
        [
            _prediction(
                actual=MatchOutcome.DRAW,
                home=0.2,
                draw=0.6,
                away=0.2,
            )
        ],
        n_bins=4,
    )

    assert all(len(outcome.bins) == 4 for outcome in report.outcomes)
    assert any(
        item.forecast_count == 0
        and item.average_forecast_probability is None
        and item.observed_frequency is None
        for item in report.for_outcome(MatchOutcome.DRAW).bins
    )


def test_calibration_report_identity_is_order_independent() -> None:
    first_prediction = _prediction(
        actual=MatchOutcome.HOME_WIN,
        home=0.7,
        draw=0.2,
        away=0.1,
    )
    second_prediction = _prediction(
        actual=MatchOutcome.DRAW,
        home=0.2,
        draw=0.6,
        away=0.2,
    )

    forward = build_calibration_report(
        [first_prediction, second_prediction],
        n_bins=5,
    )
    reverse = build_calibration_report(
        [second_prediction, first_prediction],
        n_bins=5,
    )

    assert forward.report_id == reverse.report_id


def test_calibration_validation_rejects_empty_inputs_and_invalid_bins() -> None:
    with pytest.raises(ValueError, match="At least one prediction"):
        build_calibration_report([])

    prediction = _prediction(
        actual=MatchOutcome.HOME_WIN,
        home=0.6,
        draw=0.2,
        away=0.2,
    )
    with pytest.raises(ValueError, match="n_bins must be positive"):
        build_calibration_report([prediction], n_bins=0)
