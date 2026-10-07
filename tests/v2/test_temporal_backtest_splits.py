from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from football_analytics.domain import MatchOutcome
from football_analytics.evaluation import (
    ExpandingWindowPolicy,
    RollingWindowPolicy,
    TemporalFoldSkipReason,
    build_expanding_window_folds,
    build_rolling_window_folds,
)
from football_analytics.features import (
    FeatureDefinition,
    FeatureStatus,
    FeatureValue,
    FeatureVector,
    HistoricalFeatureDataset,
    HistoricalFeatureExample,
    feature_set_id_for_definitions,
)

DEFINITION = FeatureDefinition(
    name="test.feature",
    version="v1",
    group="test",
    description="Feature used to test chronological fold construction.",
)
FEATURE_SET_ID = feature_set_id_for_definitions([DEFINITION])


def _example(match_id: str, prediction_time: datetime) -> HistoricalFeatureExample:
    vector = FeatureVector(
        match_id=match_id,
        prediction_time=prediction_time,
        feature_set_id=FEATURE_SET_ID,
        values=(
            FeatureValue(
                definition=DEFINITION,
                status=FeatureStatus.OBSERVED,
                as_of=prediction_time,
                value=1.0,
            ),
        ),
    )
    return HistoricalFeatureExample(
        match_id=match_id,
        source_match_id=f"source-{match_id}",
        prediction_time=prediction_time,
        target=MatchOutcome.HOME_WIN,
        vector=vector,
    )


def _dataset() -> HistoricalFeatureDataset:
    return HistoricalFeatureDataset(
        feature_set_id=FEATURE_SET_ID,
        cutoff_policy_id="test-cutoffs",
        examples=(
            _example("m2020", datetime(2020, 1, 1, tzinfo=UTC)),
            _example("m2021", datetime(2021, 1, 1, tzinfo=UTC)),
            _example("m2022", datetime(2022, 1, 10, tzinfo=UTC)),
            _example("m2023", datetime(2023, 1, 10, tzinfo=UTC)),
        ),
    )


def test_expanding_window_uses_all_history_before_cutoff() -> None:
    policy = ExpandingWindowPolicy(
        policy_id="expanding-v1",
        cutoffs=(datetime(2022, 1, 1, tzinfo=UTC),),
        evaluation_window=timedelta(days=365),
        min_train_examples=2,
    )

    report = build_expanding_window_folds(_dataset(), policy)

    assert len(report.folds) == 1
    fold = report.folds[0]
    assert [example.match_id for example in fold.train] == ["m2020", "m2021"]
    assert [example.match_id for example in fold.test] == ["m2022"]
    assert fold.training_start is None
    assert report.skipped == ()


def test_rolling_window_discards_old_training_history() -> None:
    policy = RollingWindowPolicy(
        policy_id="rolling-v1",
        cutoffs=(datetime(2022, 1, 1, tzinfo=UTC),),
        evaluation_window=timedelta(days=365),
        training_window=timedelta(days=400),
    )

    report = build_rolling_window_folds(_dataset(), policy)

    fold = report.folds[0]
    assert [example.match_id for example in fold.train] == ["m2021"]
    assert [example.match_id for example in fold.test] == ["m2022"]
    assert fold.training_start == datetime(2020, 11, 27, tzinfo=UTC)


def test_requested_cutoffs_record_explicit_skip_reasons() -> None:
    policy = ExpandingWindowPolicy(
        policy_id="audit-v1",
        cutoffs=(
            datetime(2019, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        ),
        evaluation_window=timedelta(days=365),
        min_train_examples=2,
    )

    report = build_expanding_window_folds(_dataset(), policy)

    assert report.folds == ()
    assert report.requested_fold_count == 2
    assert [skip.reason for skip in report.skipped] == [
        TemporalFoldSkipReason.INSUFFICIENT_TRAINING_EXAMPLES,
        TemporalFoldSkipReason.EMPTY_EVALUATION_WINDOW,
    ]


def test_policy_normalizes_and_sorts_aware_cutoffs() -> None:
    policy = ExpandingWindowPolicy(
        policy_id="sorted-v1",
        cutoffs=(
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2022, 1, 1, tzinfo=UTC),
        ),
        evaluation_window=timedelta(days=30),
    )

    assert policy.cutoffs == (
        datetime(2022, 1, 1, tzinfo=UTC),
        datetime(2023, 1, 1, tzinfo=UTC),
    )


def test_policy_rejects_naive_or_duplicate_cutoffs() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ExpandingWindowPolicy(
            policy_id="naive",
            cutoffs=(datetime(2022, 1, 1),),
            evaluation_window=timedelta(days=30),
        )

    with pytest.raises(ValueError, match="unique"):
        ExpandingWindowPolicy(
            policy_id="duplicate",
            cutoffs=(
                datetime(2022, 1, 1, tzinfo=UTC),
                datetime(2022, 1, 1, tzinfo=UTC),
            ),
            evaluation_window=timedelta(days=30),
        )


def test_policy_rejects_nonpositive_windows_and_training_minimums() -> None:
    cutoff = (datetime(2022, 1, 1, tzinfo=UTC),)

    with pytest.raises(ValueError, match="evaluation_window"):
        ExpandingWindowPolicy(
            policy_id="bad-eval",
            cutoffs=cutoff,
            evaluation_window=timedelta(0),
        )

    with pytest.raises(ValueError, match="training_window"):
        RollingWindowPolicy(
            policy_id="bad-train",
            cutoffs=cutoff,
            evaluation_window=timedelta(days=30),
            training_window=timedelta(0),
        )

    with pytest.raises(ValueError, match="min_train_examples"):
        ExpandingWindowPolicy(
            policy_id="bad-min",
            cutoffs=cutoff,
            evaluation_window=timedelta(days=30),
            min_train_examples=0,
        )
