from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from football_analytics.data import CanonicalMatchRecord, LeakageRisk, SourceMetadata
from football_analytics.domain import Match, MatchStatus
from football_analytics.domain.scores import ScoreBasis
from football_analytics.features import (
    FeatureMissingReason,
    FeatureStatus,
    PredictionContext,
    ScheduleRestFeatureProvider,
    build_feature_vector,
)


def _record(
    *,
    match_id: str,
    match_date: date,
    home: str,
    away: str,
    kickoff_at: datetime | None = None,
) -> CanonicalMatchRecord:
    return CanonicalMatchRecord(
        score_basis=ScoreBasis.REGULATION_TIME,
        match=Match(
            match_id=match_id,
            match_date=match_date,
            kickoff_at=kickoff_at,
            home_team_id=home,
            away_team_id=away,
            competition_id="friendly",
            neutral=True,
            status=MatchStatus.COMPLETED,
        ),
        metadata=SourceMetadata(
            source="history",
            ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
            leakage_risk=LeakageRisk.POST_MATCH_ONLY,
        ),
        source_match_id=f"source-{match_id}",
        source_home_team_name=home,
        source_away_team_name=away,
        source_competition_name="International Friendly",
        home_score=1,
        away_score=0,
    )


def test_rest_days_are_observed_when_both_match_times_are_exact() -> None:
    target = Match(
        match_id="target",
        match_date=date(2026, 6, 10),
        kickoff_at=datetime(2026, 6, 10, 20, 0, tzinfo=UTC),
        home_team_id="ARG",
        away_team_id="FRA",
        competition_id="friendly",
        neutral=True,
        status=MatchStatus.SCHEDULED,
    )
    history = [
        _record(
            match_id="arg-prior",
            match_date=date(2026, 6, 1),
            kickoff_at=datetime(2026, 6, 1, 20, 0, tzinfo=UTC),
            home="ARG",
            away="BRA",
        ),
        _record(
            match_id="fra-prior",
            match_date=date(2026, 6, 3),
            kickoff_at=datetime(2026, 6, 3, 20, 0, tzinfo=UTC),
            home="FRA",
            away="GER",
        ),
    ]
    context = PredictionContext(
        match=target,
        prediction_time=datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
    )

    vector = build_feature_vector(
        context,
        [ScheduleRestFeatureProvider(history)],
    )
    values = {value.definition.name: value for value in vector.values}

    assert values["schedule.home.rest_days"].value == pytest.approx(9.0)
    assert values["schedule.away.rest_days"].value == pytest.approx(7.0)
    assert values["schedule.rest_days_diff_home_minus_away"].value == pytest.approx(2.0)
    assert values["schedule.home.rest_days"].status is FeatureStatus.OBSERVED


def test_date_only_prior_match_makes_rest_explicitly_imputed() -> None:
    target = Match(
        match_id="target",
        match_date=date(2026, 6, 10),
        kickoff_at=datetime(2026, 6, 10, 20, 0, tzinfo=UTC),
        home_team_id="ARG",
        away_team_id="FRA",
        competition_id="friendly",
        neutral=True,
        status=MatchStatus.SCHEDULED,
    )
    history = [
        _record(
            match_id="arg-prior",
            match_date=date(2026, 6, 1),
            home="ARG",
            away="BRA",
        )
    ]
    context = PredictionContext(
        match=target,
        prediction_time=datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
    )

    vector = build_feature_vector(
        context,
        [ScheduleRestFeatureProvider(history)],
    )
    values = {value.definition.name: value for value in vector.values}

    home = values["schedule.home.rest_days"]
    assert home.value == pytest.approx(9.0)
    assert home.status is FeatureStatus.IMPUTED
    assert home.missing_reason is FeatureMissingReason.TEMPORAL_PRECISION
    assert home.imputation_method is not None


def test_rest_days_are_missing_when_team_has_no_prior_history() -> None:
    target = Match(
        match_id="target",
        match_date=date(2026, 6, 10),
        home_team_id="ARG",
        away_team_id="FRA",
        competition_id="friendly",
        neutral=True,
        status=MatchStatus.SCHEDULED,
    )
    context = PredictionContext(
        match=target,
        prediction_time=datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
    )

    vector = build_feature_vector(
        context,
        [ScheduleRestFeatureProvider([])],
    )
    values = {value.definition.name: value for value in vector.values}

    assert values["schedule.home.rest_days"].status is FeatureStatus.MISSING
    assert (
        values["schedule.home.rest_days"].missing_reason
        is FeatureMissingReason.NO_HISTORY
    )
    assert values["schedule.rest_days_diff_home_minus_away"].value is None
