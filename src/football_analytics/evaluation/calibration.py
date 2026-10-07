from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from football_analytics.domain import MatchOutcome
from football_analytics.evaluation.metrics import ScoredPrediction


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One equal-width reliability bin for a single outcome class."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    forecast_count: int
    average_forecast_probability: float | None
    observed_frequency: float | None
    calibration_error: float | None


@dataclass(frozen=True, slots=True)
class OutcomeCalibrationReport:
    """Reliability diagnostics for one home/draw/away outcome class."""

    outcome: MatchOutcome
    n_predictions: int
    expected_calibration_error: float
    maximum_calibration_error: float
    calibration_bias: float
    bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Deterministic multiclass reliability report."""

    report_id: str
    n_predictions: int
    n_bins: int
    macro_expected_calibration_error: float
    outcomes: tuple[OutcomeCalibrationReport, ...]

    def for_outcome(self, outcome: MatchOutcome) -> OutcomeCalibrationReport:
        for report in self.outcomes:
            if report.outcome is outcome:
                return report
        raise KeyError(outcome)


def build_calibration_report(
    predictions: Sequence[ScoredPrediction],
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """Build equal-width one-vs-rest reliability diagnostics.

    Empty bins are retained so downstream tables and plots have a stable shape.
    Expected calibration error is frequency-weighted across non-empty bins.
    """

    if not predictions:
        raise ValueError("At least one prediction is required for calibration.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    outcome_reports = tuple(
        _outcome_report(predictions, outcome=outcome, n_bins=n_bins)
        for outcome in (
            MatchOutcome.HOME_WIN,
            MatchOutcome.DRAW,
            MatchOutcome.AWAY_WIN,
        )
    )
    macro_ece = sum(
        report.expected_calibration_error for report in outcome_reports
    ) / len(outcome_reports)

    return CalibrationReport(
        report_id=_report_id(predictions, n_bins=n_bins),
        n_predictions=len(predictions),
        n_bins=n_bins,
        macro_expected_calibration_error=macro_ece,
        outcomes=outcome_reports,
    )


def _outcome_report(
    predictions: Sequence[ScoredPrediction],
    *,
    outcome: MatchOutcome,
    n_bins: int,
) -> OutcomeCalibrationReport:
    counts = [0] * n_bins
    probability_sums = [0.0] * n_bins
    observed_sums = [0.0] * n_bins

    total_probability = 0.0
    total_observed = 0.0

    for prediction in predictions:
        probability = prediction.probabilities.probability_for(outcome)
        observed = 1.0 if prediction.actual is outcome else 0.0
        bin_index = min(int(probability * n_bins), n_bins - 1)

        counts[bin_index] += 1
        probability_sums[bin_index] += probability
        observed_sums[bin_index] += observed
        total_probability += probability
        total_observed += observed

    bins: list[CalibrationBin] = []
    weighted_absolute_error = 0.0
    maximum_error = 0.0
    total_count = len(predictions)

    for bin_index in range(n_bins):
        count = counts[bin_index]
        lower_bound = bin_index / n_bins
        upper_bound = (bin_index + 1) / n_bins

        if count == 0:
            bins.append(
                CalibrationBin(
                    bin_index=bin_index,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    forecast_count=0,
                    average_forecast_probability=None,
                    observed_frequency=None,
                    calibration_error=None,
                )
            )
            continue

        average_probability = probability_sums[bin_index] / count
        observed_frequency = observed_sums[bin_index] / count
        signed_error = observed_frequency - average_probability
        absolute_error = abs(signed_error)
        weighted_absolute_error += (count / total_count) * absolute_error
        maximum_error = max(maximum_error, absolute_error)

        bins.append(
            CalibrationBin(
                bin_index=bin_index,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                forecast_count=count,
                average_forecast_probability=average_probability,
                observed_frequency=observed_frequency,
                calibration_error=signed_error,
            )
        )

    calibration_bias = (total_observed - total_probability) / total_count

    return OutcomeCalibrationReport(
        outcome=outcome,
        n_predictions=total_count,
        expected_calibration_error=weighted_absolute_error,
        maximum_calibration_error=maximum_error,
        calibration_bias=calibration_bias,
        bins=tuple(bins),
    )


def _report_id(
    predictions: Sequence[ScoredPrediction],
    *,
    n_bins: int,
) -> str:
    serialized = [
        {
            "actual": prediction.actual.value,
            "probabilities": list(prediction.probabilities.as_tuple()),
        }
        for prediction in predictions
    ]
    serialized.sort(
        key=lambda item: (
            str(item["actual"]),
            tuple(float(value) for value in item["probabilities"]),
        )
    )
    payload = {
        "n_bins": n_bins,
        "predictions": serialized,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"calibration_report_{digest}"
