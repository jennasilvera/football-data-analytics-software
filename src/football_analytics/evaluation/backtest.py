from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

from football_analytics.domain import MatchOutcome
from football_analytics.evaluation.metrics import (
    EvaluationMetrics,
    ScoredPrediction,
    evaluate_predictions,
)
from football_analytics.evaluation.probabilities import OutcomeProbabilities
from football_analytics.evaluation.splits import (
    SkippedTemporalFold,
    TemporalBacktestFold,
    TemporalFoldBuildReport,
)
from football_analytics.features import ImputationPolicy
from football_analytics.models import (
    ModelDataset,
    ModelTrainingResult,
    ModelTrainingSpec,
    build_model_dataset_from_examples,
)

ModelTrainer = Callable[[ModelDataset, ModelTrainingSpec], ModelTrainingResult]


@dataclass(frozen=True, slots=True)
class BacktestPrediction:
    """One immutable out-of-sample prediction generated inside a temporal fold."""

    fold_id: str
    match_id: str
    prediction_time_iso: str
    actual: MatchOutcome
    probabilities: OutcomeProbabilities
    model_id: str


@dataclass(frozen=True, slots=True)
class BacktestFoldResult:
    """Auditable model fit and evaluation result for one chronological fold."""

    fold_id: str
    cutoff_iso: str
    train_dataset_id: str
    evaluation_dataset_id: str
    training_run_id: str
    model_id: str
    train_count: int
    test_count: int
    metrics: EvaluationMetrics
    predictions: tuple[BacktestPrediction, ...]
    train_match_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TemporalBacktestResult:
    """Complete deterministic temporal evaluation result."""

    backtest_run_id: str
    split_policy_id: str
    model_spec_id: str
    model_family: str
    model_version: str
    model_random_seed: int
    model_parameters: tuple[tuple[str, bool | int | float | str], ...]
    feature_set_id: str
    cutoff_policy_id: str
    result_eligibility_policy_id: str
    imputation_policy_id: str
    folds: tuple[BacktestFoldResult, ...]
    skipped_folds: tuple[SkippedTemporalFold, ...]
    aggregate_metrics: EvaluationMetrics

    @property
    def prediction_count(self) -> int:
        return sum(len(fold.predictions) for fold in self.folds)


def run_temporal_backtest(
    *,
    fold_report: TemporalFoldBuildReport,
    imputation_policy: ImputationPolicy,
    model_spec: ModelTrainingSpec,
    trainer: ModelTrainer,
) -> TemporalBacktestResult:
    """Train and evaluate one model independently across declared temporal folds."""

    if not fold_report.folds:
        raise ValueError("Temporal backtest requires at least one valid fold.")

    fold_results: list[BacktestFoldResult] = []
    aggregate_scored: list[ScoredPrediction] = []

    for fold in fold_report.folds:
        fold_result, scored = _run_fold(
            fold=fold,
            feature_set_id=fold_report.source_feature_set_id,
            cutoff_policy_id=fold_report.source_cutoff_policy_id,
            result_eligibility_policy_id=(
                fold_report.source_result_eligibility_policy_id
            ),
            imputation_policy=imputation_policy,
            model_spec=model_spec,
            trainer=trainer,
        )
        fold_results.append(fold_result)
        aggregate_scored.extend(scored)

    aggregate_metrics = evaluate_predictions(aggregate_scored)
    backtest_run_id = _backtest_run_id(
        split_policy_id=fold_report.policy_id,
        model_spec=model_spec,
        feature_set_id=fold_report.source_feature_set_id,
        cutoff_policy_id=fold_report.source_cutoff_policy_id,
        result_eligibility_policy_id=(
            fold_report.source_result_eligibility_policy_id
        ),
        imputation_policy_id=imputation_policy.policy_id,
        folds=tuple(fold_results),
    )

    return TemporalBacktestResult(
        backtest_run_id=backtest_run_id,
        split_policy_id=fold_report.policy_id,
        model_spec_id=model_spec.spec_id,
        model_family=model_spec.family.value,
        model_version=model_spec.model_version,
        model_random_seed=model_spec.random_seed,
        model_parameters=model_spec.parameters,
        feature_set_id=fold_report.source_feature_set_id,
        cutoff_policy_id=fold_report.source_cutoff_policy_id,
        result_eligibility_policy_id=(
            fold_report.source_result_eligibility_policy_id
        ),
        imputation_policy_id=imputation_policy.policy_id,
        folds=tuple(fold_results),
        skipped_folds=fold_report.skipped,
        aggregate_metrics=aggregate_metrics,
    )


def _run_fold(
    *,
    fold: TemporalBacktestFold,
    feature_set_id: str,
    cutoff_policy_id: str,
    result_eligibility_policy_id: str,
    imputation_policy: ImputationPolicy,
    model_spec: ModelTrainingSpec,
    trainer: ModelTrainer,
) -> tuple[BacktestFoldResult, tuple[ScoredPrediction, ...]]:
    train_dataset = build_model_dataset_from_examples(
        feature_set_id=feature_set_id,
        cutoff_policy_id=cutoff_policy_id,
        result_eligibility_policy_id=result_eligibility_policy_id,
        examples=fold.train,
        imputation_policy=imputation_policy,
    )
    evaluation_dataset = build_model_dataset_from_examples(
        feature_set_id=feature_set_id,
        cutoff_policy_id=cutoff_policy_id,
        result_eligibility_policy_id=result_eligibility_policy_id,
        examples=fold.test,
        imputation_policy=imputation_policy,
    )

    training = trainer(train_dataset, model_spec)
    predictions: list[BacktestPrediction] = []
    scored: list[ScoredPrediction] = []

    for example in evaluation_dataset.examples:
        probabilities = training.model.predict_row(example.row)
        scored_prediction = ScoredPrediction(
            actual=example.target,
            probabilities=probabilities,
        )
        scored.append(scored_prediction)
        predictions.append(
            BacktestPrediction(
                fold_id=fold.fold_id,
                match_id=example.match_id,
                prediction_time_iso=example.row.prediction_time_iso,
                actual=example.target,
                probabilities=probabilities,
                model_id=training.metadata.model_id,
            )
        )

    metrics = evaluate_predictions(scored)

    return (
        BacktestFoldResult(
            fold_id=fold.fold_id,
            cutoff_iso=fold.cutoff.isoformat(),
            train_dataset_id=train_dataset.dataset_id,
            evaluation_dataset_id=evaluation_dataset.dataset_id,
            training_run_id=training.metadata.training_run_id,
            model_id=training.metadata.model_id,
            train_count=len(train_dataset.examples),
            test_count=len(evaluation_dataset.examples),
            metrics=metrics,
            predictions=tuple(predictions),
            train_match_ids=tuple(example.match_id for example in train_dataset.examples),
        ),
        tuple(scored),
    )


def _backtest_run_id(
    *,
    split_policy_id: str,
    model_spec: ModelTrainingSpec,
    feature_set_id: str,
    cutoff_policy_id: str,
    result_eligibility_policy_id: str,
    imputation_policy_id: str,
    folds: tuple[BacktestFoldResult, ...],
) -> str:
    payload = {
        "split_policy_id": split_policy_id,
        "model_spec": {
            "spec_id": model_spec.spec_id,
            "family": model_spec.family.value,
            "model_version": model_spec.model_version,
            "random_seed": model_spec.random_seed,
            "parameters": list(model_spec.parameters),
        },
        "feature_set_id": feature_set_id,
        "cutoff_policy_id": cutoff_policy_id,
        "result_eligibility_policy_id": result_eligibility_policy_id,
        "imputation_policy_id": imputation_policy_id,
        "folds": [
            {
                "fold_id": fold.fold_id,
                "train_dataset_id": fold.train_dataset_id,
                "evaluation_dataset_id": fold.evaluation_dataset_id,
                "training_run_id": fold.training_run_id,
                "model_id": fold.model_id,
            }
            for fold in folds
        ],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"backtest_run_{digest}"
