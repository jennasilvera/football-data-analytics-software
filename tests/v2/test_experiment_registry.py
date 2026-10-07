from __future__ import annotations

import json
from dataclasses import replace

import pytest

from football_analytics.domain import MatchOutcome
from football_analytics.evaluation import (
    BacktestFoldResult,
    BacktestPrediction,
    EvaluationMetrics,
    OutcomeProbabilities,
    ScoredPrediction,
    TemporalBacktestResult,
    build_calibration_report,
)
from football_analytics.experiments import (
    ExperimentManifest,
    JsonExperimentRegistry,
    build_experiment_manifest,
)


def _metrics(n_predictions: int = 2) -> EvaluationMetrics:
    return EvaluationMetrics(
        n_predictions=n_predictions,
        accuracy=0.5,
        log_loss=0.8,
        multiclass_brier_score=0.55,
        ranked_probability_score=0.21,
    )


def _predictions() -> tuple[BacktestPrediction, ...]:
    return (
        BacktestPrediction(
            fold_id="fold-1",
            match_id="match-1",
            prediction_time_iso="2026-01-01T12:00:00+00:00",
            actual=MatchOutcome.HOME_WIN,
            probabilities=OutcomeProbabilities(
                home_win=0.60,
                draw=0.25,
                away_win=0.15,
            ),
            model_id="model-1",
        ),
        BacktestPrediction(
            fold_id="fold-1",
            match_id="match-2",
            prediction_time_iso="2026-01-02T12:00:00+00:00",
            actual=MatchOutcome.AWAY_WIN,
            probabilities=OutcomeProbabilities(
                home_win=0.40,
                draw=0.25,
                away_win=0.35,
            ),
            model_id="model-1",
        ),
    )


def _backtest() -> TemporalBacktestResult:
    predictions = _predictions()
    fold = BacktestFoldResult(
        fold_id="fold-1",
        cutoff_iso="2026-01-01T00:00:00+00:00",
        train_dataset_id="model_dataset_train",
        evaluation_dataset_id="model_dataset_eval",
        training_run_id="training_run_1",
        model_id="model-1",
        train_count=100,
        test_count=2,
        metrics=_metrics(),
        predictions=predictions,
    )
    return TemporalBacktestResult(
        backtest_run_id="backtest_run_1",
        split_policy_id="expanding_v1",
        model_spec_id="logistic_v1",
        model_family="logistic_regression",
        model_version="v1",
        model_random_seed=42,
        model_parameters=(("C", 1.0), ("max_iter", 1000)),
        feature_set_id="feature_set_1",
        cutoff_policy_id="pre_match_v1",
        result_eligibility_policy_id="completed_result_eligibility_v1",
        imputation_policy_id="imputation_v1",
        folds=(fold,),
        skipped_folds=(),
        aggregate_metrics=_metrics(),
    )


def _calibration():
    scored = [
        ScoredPrediction(
            actual=prediction.actual,
            probabilities=prediction.probabilities,
        )
        for prediction in _predictions()
    ]
    return build_calibration_report(scored, n_bins=5)


def test_experiment_manifest_is_deterministic_and_self_describing() -> None:
    first = build_experiment_manifest(
        _backtest(),
        calibration=_calibration(),
        code_revision="abc123",
    )
    second = build_experiment_manifest(
        _backtest(),
        calibration=_calibration(),
        code_revision="abc123",
    )

    assert first == second
    assert first.experiment_id.startswith("experiment_")
    assert first.model_family == "logistic_regression"
    assert first.model_random_seed == 42
    assert first.model_parameters == (("C", 1.0), ("max_iter", 1000))
    assert first.calibration_report_id == _calibration().report_id
    assert first.code_revision == "abc123"


def test_experiment_identity_changes_with_code_revision() -> None:
    first = build_experiment_manifest(_backtest(), code_revision="abc123")
    second = build_experiment_manifest(_backtest(), code_revision="def456")

    assert first.experiment_id != second.experiment_id



def test_experiment_identity_changes_with_reported_metrics() -> None:
    baseline = _backtest()
    changed = replace(
        baseline,
        aggregate_metrics=EvaluationMetrics(
            n_predictions=2,
            accuracy=0.75,
            log_loss=0.7,
            multiclass_brier_score=0.45,
            ranked_probability_score=0.18,
        ),
    )

    first = build_experiment_manifest(baseline)
    second = build_experiment_manifest(changed)

    assert first.experiment_id != second.experiment_id


def test_manifest_round_trip_preserves_deterministic_identity() -> None:
    manifest = build_experiment_manifest(
        _backtest(),
        calibration=_calibration(),
        code_revision="abc123",
    )

    restored = ExperimentManifest.from_dict(manifest.to_dict())

    assert restored == manifest


def test_manifest_rejects_calibration_from_different_prediction_sample() -> None:
    other = build_calibration_report(
        [
            ScoredPrediction(
                actual=MatchOutcome.DRAW,
                probabilities=OutcomeProbabilities(
                    home_win=0.2,
                    draw=0.6,
                    away_win=0.2,
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
        ],
        n_bins=5,
    )

    with pytest.raises(ValueError, match="does not match"):
        build_experiment_manifest(_backtest(), calibration=other)


def test_manifest_detects_identity_tampering() -> None:
    manifest = build_experiment_manifest(_backtest(), code_revision="abc123")
    payload = manifest.to_dict()
    payload["model_version"] = "tampered"

    with pytest.raises(ValueError, match="deterministic identity"):
        ExperimentManifest.from_dict(payload)


def test_json_registry_is_idempotent_and_round_trips(tmp_path) -> None:
    registry = JsonExperimentRegistry(tmp_path / "experiments")
    manifest = build_experiment_manifest(
        _backtest(),
        calibration=_calibration(),
        code_revision="abc123",
    )

    first_path = registry.put(manifest)
    second_path = registry.put(manifest)

    assert first_path == second_path
    assert registry.get(manifest.experiment_id) == manifest
    assert registry.list_ids() == (manifest.experiment_id,)


def test_registry_detects_tampered_manifest_content(tmp_path) -> None:
    registry = JsonExperimentRegistry(tmp_path / "experiments")
    manifest = build_experiment_manifest(_backtest(), code_revision="abc123")
    path = registry.put(manifest)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["aggregate_metrics"]["accuracy"] = 0.99
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic identity"):
        registry.get(manifest.experiment_id)


def test_registry_rejects_path_identity_mismatch(tmp_path) -> None:
    registry = JsonExperimentRegistry(tmp_path / "experiments")
    first = build_experiment_manifest(_backtest(), code_revision="abc123")
    second = build_experiment_manifest(_backtest(), code_revision="def456")
    first_path = registry.put(first)

    first_path.write_text(
        json.dumps(second.to_dict()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="registry path"):
        registry.get(first.experiment_id)


def test_registry_rejects_invalid_experiment_ids(tmp_path) -> None:
    registry = JsonExperimentRegistry(tmp_path / "experiments")

    with pytest.raises(ValueError, match="Invalid experiment_id"):
        registry.get("../escape")
