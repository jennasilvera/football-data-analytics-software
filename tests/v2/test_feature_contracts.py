from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from football_analytics.data import TemporalIntegrityError
from football_analytics.domain import Match
from football_analytics.features import (
    FeatureDefinition,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    PredictionContext,
)


DEFINITION = FeatureDefinition(
    name="example.feature",
    version="v1",
    group="example",
    description="Example feature used by contract tests.",
)


def test_exact_kickoff_context_rejects_post_kickoff_prediction_time() -> None:
    match = Match(
        match_id="m1",
        match_date=date(2026, 11, 14),
        kickoff_at=datetime(2026, 11, 14, 20, 0, tzinfo=UTC),
        home_team_id="ARG",
        away_team_id="BRA",
        competition_id="friendly",
        neutral=False,
    )

    with pytest.raises(TemporalIntegrityError, match="after exact kickoff"):
        PredictionContext(
            match=match,
            prediction_time=datetime(2026, 11, 14, 20, 0, 1, tzinfo=UTC),
        )


def test_date_only_context_requires_prior_calendar_day_cutoff() -> None:
    match = Match(
        match_id="m1",
        match_date=date(2001, 9, 5),
        home_team_id="ARG",
        away_team_id="BRA",
        competition_id="qualifier",
        neutral=False,
    )

    with pytest.raises(TemporalIntegrityError, match="before match_date"):
        PredictionContext(
            match=match,
            prediction_time=datetime(2001, 9, 5, 0, 0, tzinfo=UTC),
        )

    context = PredictionContext(
        match=match,
        prediction_time=datetime(2001, 9, 4, 23, 59, tzinfo=UTC),
    )
    assert context.prediction_time.date() == date(2001, 9, 4)


def test_observed_feature_requires_value_without_missing_metadata() -> None:
    value = FeatureValue(
        definition=DEFINITION,
        status=FeatureStatus.OBSERVED,
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        value=1.25,
    )

    assert value.value == pytest.approx(1.25)


def test_missing_feature_requires_reason_and_no_value() -> None:
    value = FeatureValue(
        definition=DEFINITION,
        status=FeatureStatus.MISSING,
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        missing_reason=FeatureMissingReason.NO_HISTORY,
    )

    assert value.value is None

    with pytest.raises(ValueError, match="require missing_reason"):
        FeatureValue(
            definition=DEFINITION,
            status=FeatureStatus.MISSING,
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_imputed_feature_requires_method_and_original_missing_reason() -> None:
    with pytest.raises(ValueError, match="require imputation_method"):
        FeatureValue(
            definition=DEFINITION,
            status=FeatureStatus.IMPUTED,
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
            value=0.0,
            missing_reason=FeatureMissingReason.NO_HISTORY,
        )


def test_feature_values_reject_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="finite"):
        FeatureValue(
            definition=DEFINITION,
            status=FeatureStatus.OBSERVED,
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
            value=float("nan"),
        )
