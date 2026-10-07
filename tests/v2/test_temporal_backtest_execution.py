from __future__ import annotations

from datetime import UTC, datetime, timedelta

from football_analytics.domain import MatchOutcome
from football_analytics.evaluation import (
    ExpandingWindowPolicy,
    TemporalFoldBuildReport,
    build_expanding_window_folds,
    run_temporal_backtest,
)
from football_analytics.features import (
    FeatureDefinition,
    FeatureStatus,
    FeatureValue,
    FeatureVector,
    HistoricalFeatureDataset,
    HistoricalFeatureExample,
    ImputationPolicy,
    feature_set_id_for_definitions,
)
from football_analytics.models import (
    logistic_regression_spec,
    train_sklearn_model,
)

DEFINITION = FeatureDefinition(
    name="synthetic.strength",
    version="v1",
    group="test",
    description="Synthetic feature for temporal backtest execution tests.",
)
FEATURE_SET_ID = feature_set_id_for_definitions([DEFINITION])
IMPUTATION_POLICY = ImputationPolicy(
    policy_id="backtest_no_missing_v1",
    rules=(),
    include_status_indicators=False,
)


def _target_for(index: int) -> MatchOutcome:
    return (
        MatchOutcome.HOME_WIN,
        MatchOutcome.DRAW,
        MatchOutcome.AWAY_WIN,
    )[index % 3]


def _feature_for(target: MatchOutcome, index: int) -> float:
    base = {
        MatchOutcome.HOME_WIN: 1.0,
        MatchOutcome.DRAW: 0.0,
        MatchOutcome.AWAY_WIN: -1.0,
    }[target]
    return base + (index * 0.01)


def _example(index: int) -> HistoricalFeatureExample:
    prediction_time = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index)
    target = _target_for(index)
    match_id = f"backtest-{index:02d}"
    vector = FeatureVector(
        match_id=match_id,
        prediction_time=prediction_time,
        feature_set_id=FEATURE_SET_ID,
        values=(
            FeatureValue(
                definition=DEFINITION,
                status=FeatureStatus.OBSERVED,
                as_of=prediction_time,
                value=_feature_for(target, index),
            ),
        ),
    )
    return HistoricalFeatureExample(
        match_id=match_id,
        source_match_id=f"source-{match_id}",
        prediction_time=prediction_time,
        target=target,
        vector=vector,
    )


def _dataset() -> HistoricalFeatureDataset:
    return HistoricalFeatureDataset(
        examples=tuple(_example(index) for index in range(12)),
    )


def _fold_report() -> TemporalFoldBuildReport:
    policy = ExpandingWindowPolicy(
        policy_id="expanding_model_test_v1",
        cutoffs=(
            datetime(2020, 1, 7, tzinfo=UTC),
            datetime(2020, 1, 10, tzinfo=UTC),
        ),
        evaluation_window=timedelta(days=3),
        min_train_examples=6,
    )
    return build_expanding_window_folds(_dataset(), policy)


def test_temporal_backtest_fits_each_fold_and_aggregates_predictions() -> None:
    result = run_temporal_backtest(
        fold_report=_fold_report(),
        imputation_policy=IMPUTATION_POLICY,
        model_spec=logistic_regression_spec(max_iter=500),
        trainer=train_sklearn_model,
    )

    assert len(result.folds) == 2
    assert result.prediction_count == 6
    assert result.aggregate_metrics.n_predictions == 6
    assert result.skipped_folds == ()

    assert [fold.train_count for fold in result.folds] == [6, 9]
    assert [fold.test_count for fold in result.folds] == [3, 3]
    assert result.folds[0].train_dataset_id != result.folds[1].train_dataset_id
    assert result.folds[0].model_id != result.folds[1].model_id

    predicted_match_ids = [
        prediction.match_id
        for fold in result.folds
        for prediction in fold.predictions
    ]
    assert predicted_match_ids == [
        "backtest-06",
        "backtest-07",
        "backtest-08",
        "backtest-09",
        "backtest-10",
        "backtest-11",
    ]


def test_temporal_backtest_identity_is_reproducible() -> None:
    kwargs = {
        "fold_report": _fold_report(),
        "imputation_policy": IMPUTATION_POLICY,
        "model_spec": logistic_regression_spec(max_iter=500),
        "trainer": train_sklearn_model,
    }

    first = run_temporal_backtest(**kwargs)
    second = run_temporal_backtest(**kwargs)

    assert first.backtest_run_id == second.backtest_run_id
    assert [fold.training_run_id for fold in first.folds] == [
        fold.training_run_id for fold in second.folds
    ]
    assert [fold.metrics for fold in first.folds] == [
        fold.metrics for fold in second.folds
    ]
