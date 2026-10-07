"""Paired model comparison rejects mismatched forecasts and duplicate matches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from football_analytics.domain.scores import REGULATION_TARGET_POLICY_ID
from football_analytics.evaluation.backtest import TemporalBacktestResult
from football_analytics.evaluation.metrics import (
    EvaluationMetrics,
    ScoredPrediction,
    evaluate_predictions,
)


@dataclass(frozen=True, slots=True)
class ModelComparisonRow:
    backtest_run_id: str
    model_family: str
    metrics: EvaluationMetrics
    log_loss_delta_vs_reference: float
    brier_delta_vs_reference: float
    rps_delta_vs_reference: float
    model_spec_id: str = ""


@dataclass(frozen=True, slots=True)
class ModelComparison:
    comparison_id: str
    reference_backtest_run_id: str
    paired_prediction_count: int
    rows: tuple[ModelComparisonRow, ...]
    target_policy_id: str = REGULATION_TARGET_POLICY_ID


def compare_backtests(
    backtests: Sequence[TemporalBacktestResult],
    *,
    reference_backtest_run_id: str,
) -> ModelComparison:
    if len(backtests) < 2 or len({run.backtest_run_id for run in backtests}) != len(backtests):
        raise ValueError("Comparison requires at least two distinct backtest runs.")
    reference = next(
        (run for run in backtests if run.backtest_run_id == reference_backtest_run_id), None
    )
    if reference is None:
        raise ValueError("Reference backtest is missing.")

    def sample(run: TemporalBacktestResult) -> tuple[tuple[object, ...], ...]:
        predictions = [prediction for fold in run.folds for prediction in fold.predictions]
        if len({p.match_id for p in predictions}) != len(predictions):
            raise ValueError("Comparison cannot include repeated evaluation matches.")
        return tuple(
            sorted(
                (p.match_id, p.prediction_time_iso, p.actual.value, fold.cutoff_iso)
                for fold in run.folds
                for p in fold.predictions
            )
        )

    def training_sample(run: TemporalBacktestResult) -> tuple[tuple[object, ...], ...]:
        if any(not fold.train_match_ids for fold in run.folds):
            raise ValueError("Comparison requires explicit training match identities.")
        return tuple(
            sorted((fold.cutoff_iso, tuple(sorted(fold.train_match_ids))) for fold in run.folds)
        )

    reference_sample = sample(reference)
    reference_training = training_sample(reference)
    metrics_by_run = {}
    for run in backtests:
        if run.target_policy_id != REGULATION_TARGET_POLICY_ID:
            raise ValueError("Models must use the regulation-time score target policy.")
        if sample(run) != reference_sample or training_sample(run) != reference_training:
            raise ValueError(
                "Models must use identical match, outcome, prediction-time and fold samples."
            )
        if (
            run.cutoff_policy_id != reference.cutoff_policy_id
            or run.result_eligibility_policy_id != reference.result_eligibility_policy_id
        ):
            raise ValueError("Models must use identical temporal policies.")
        # Recompute metrics; do not trust a report's summary when comparing runs.
        metrics_by_run[run.backtest_run_id] = evaluate_predictions(
            [
                ScoredPrediction(p.actual, p.probabilities)
                for fold in run.folds
                for p in fold.predictions
            ]
        )
    baseline = metrics_by_run[reference.backtest_run_id]
    rows = tuple(
        ModelComparisonRow(
            run.backtest_run_id,
            run.model_family,
            metrics_by_run[run.backtest_run_id],
            metrics_by_run[run.backtest_run_id].log_loss - baseline.log_loss,
            metrics_by_run[run.backtest_run_id].multiclass_brier_score
            - baseline.multiclass_brier_score,
            metrics_by_run[run.backtest_run_id].ranked_probability_score
            - baseline.ranked_probability_score,
            model_spec_id=run.model_spec_id,
        )
        for run in sorted(backtests, key=lambda item: (item.model_family, item.backtest_run_id))
    )
    payload = {
        "reference": reference_backtest_run_id,
        "sample": reference_sample,
        "rows": [asdict(row) for row in rows],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return ModelComparison(
        "comparison_" + digest, reference_backtest_run_id, len(reference_sample), rows
    )
