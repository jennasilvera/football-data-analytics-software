"""Temporal score-model evaluation sharing the outcome metrics/manifest contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from football_analytics.data import CanonicalMatchRecord
from football_analytics.domain import MatchOutcome
from football_analytics.evaluation.backtest import (
    BacktestFoldResult,
    BacktestPrediction,
    TemporalBacktestResult,
    _backtest_run_id,
)
from football_analytics.evaluation.metrics import ScoredPrediction, evaluate_predictions
from football_analytics.evaluation.splits import TemporalFoldBuildReport
from football_analytics.features.dataset import HistoricalFeatureExample
from football_analytics.features.history import DEFAULT_RESULT_ELIGIBILITY_POLICY
from football_analytics.models.base import ModelTrainingSpec
from football_analytics.models.poisson import (
    SCORE_CONTEXT_ID,
    SCORE_MISSING_POLICY_ID,
    PoissonModel,
    ScorelineForecast,
    config_from_spec,
    fit_poisson,
    goal_dataset_id,
)


@dataclass(frozen=True, slots=True)
class BacktestScoreForecast:
    fold_id: str
    match_id: str
    model_id: str
    prediction_time_iso: str
    forecast: ScorelineForecast


@dataclass(frozen=True, slots=True)
class ScoreBacktestResult:
    backtest: TemporalBacktestResult
    score_forecasts: tuple[BacktestScoreForecast, ...]
    models: tuple[PoissonModel, ...]


def _records_for(
    examples: Sequence[HistoricalFeatureExample],
    lookup: dict[str, CanonicalMatchRecord],
) -> tuple[CanonicalMatchRecord, ...]:
    records = []
    for example in examples:
        record = lookup[example.match_id]
        eligibility = DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(record)
        assert record.home_score is not None and record.away_score is not None
        actual = (
            MatchOutcome.HOME_WIN
            if record.home_score > record.away_score
            else MatchOutcome.AWAY_WIN
            if record.away_score > record.home_score
            else MatchOutcome.DRAW
        )
        if (
            example.target != actual
            or example.target_available_at != eligibility.eligible_at
            or example.target_availability_basis != eligibility.basis
            or example.source_match_id != record.source_match_id
        ):
            raise ValueError("Temporal example disagrees with its canonical result.")
        records.append(record)
    return tuple(records)


def run_poisson_backtest(
    *,
    fold_report: TemporalFoldBuildReport,
    records: Sequence[CanonicalMatchRecord],
    model_spec: ModelTrainingSpec,
) -> ScoreBacktestResult:
    config = config_from_spec(model_spec)
    if not fold_report.folds:
        raise ValueError("Poisson backtest requires at least one valid fold.")
    if (
        fold_report.source_result_eligibility_policy_id
        != DEFAULT_RESULT_ELIGIBILITY_POLICY.policy_id
    ):
        raise ValueError("Unsupported score-model result eligibility policy.")
    lookup = {record.match.match_id: record for record in records}
    if len(lookup) != len(records):
        raise ValueError("Canonical records contain duplicate match IDs.")
    fold_results = []
    all_scored = []
    forecasts = []
    models = []
    seen: set[str] = set()
    for fold in fold_report.folds:
        train = _records_for(fold.train, lookup)
        test = _records_for(fold.test, lookup)
        model = fit_poisson(train, training_cutoff=fold.cutoff, config=config)
        models.append(model)
        predictions = []
        scored = []
        for example, record in zip(fold.test, test, strict=True):
            if example.match_id in seen:
                raise ValueError("Overlapping evaluation samples would double-count a match.")
            seen.add(example.match_id)
            forecast = model.predict_match(record.match, prediction_time=example.prediction_time)
            predictions.append(
                BacktestPrediction(
                    fold.fold_id,
                    example.match_id,
                    example.prediction_time.isoformat(),
                    example.target,
                    forecast.probabilities,
                    model.model_id,
                )
            )
            forecasts.append(
                BacktestScoreForecast(
                    fold.fold_id,
                    example.match_id,
                    model.model_id,
                    example.prediction_time.isoformat(),
                    forecast,
                )
            )
            scored.append(ScoredPrediction(example.target, forecast.probabilities))
        all_scored.extend(scored)
        fold_results.append(
            BacktestFoldResult(
                fold.fold_id,
                fold.cutoff.isoformat(),
                model.training_dataset_id,
                goal_dataset_id(test),
                "training_run_" + model.model_id,
                model.model_id,
                len(train),
                len(test),
                evaluate_predictions(scored),
                tuple(predictions),
                tuple(example.match_id for example in fold.train),
            )
        )
    folds = tuple(fold_results)
    run_id = _backtest_run_id(
        split_policy_id=fold_report.policy_id,
        model_spec=model_spec,
        feature_set_id=SCORE_CONTEXT_ID,
        cutoff_policy_id=fold_report.source_cutoff_policy_id,
        result_eligibility_policy_id=fold_report.source_result_eligibility_policy_id,
        imputation_policy_id=SCORE_MISSING_POLICY_ID,
        folds=folds,
    )
    result = TemporalBacktestResult(
        run_id,
        fold_report.policy_id,
        model_spec.spec_id,
        model_spec.family.value,
        model_spec.model_version,
        model_spec.random_seed,
        model_spec.parameters,
        SCORE_CONTEXT_ID,
        fold_report.source_cutoff_policy_id,
        fold_report.source_result_eligibility_policy_id,
        SCORE_MISSING_POLICY_ID,
        folds,
        fold_report.skipped,
        evaluate_predictions(all_scored),
    )
    return ScoreBacktestResult(result, tuple(forecasts), tuple(models))
