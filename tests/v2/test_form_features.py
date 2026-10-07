from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from football_analytics.data import CanonicalMatchRecord, LeakageRisk, SourceMetadata
from football_analytics.domain import Match, MatchStatus
from football_analytics.domain.scores import ScoreBasis
from football_analytics.features import (
    FeatureStatus,
    PredictionContext,
    RollingFormFeatureProvider,
    build_feature_vector,
)


def _record(
    *,
    match_id: str,
    match_date: date,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
) -> CanonicalMatchRecord:
    return CanonicalMatchRecord(
        score_basis=ScoreBasis.REGULATION_TIME,
        match=Match(
            match_id=match_id,
            match_date=match_date,
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
        home_score=home_score,
        away_score=away_score,
    )


def _target() -> Match:
    return Match(
        match_id="target",
        match_date=date(2026, 6, 1),
        home_team_id="ARG",
        away_team_id="FRA",
        competition_id="friendly",
        neutral=True,
        status=MatchStatus.SCHEDULED,
    )


def test_rolling_form_uses_only_results_before_prediction_cutoff() -> None:
    history = [
        _record(
            match_id="arg-bra",
            match_date=date(2026, 5, 1),
            home="ARG",
            away="BRA",
            home_score=2,
            away_score=0,
        ),
        _record(
            match_id="fra-arg",
            match_date=date(2026, 5, 10),
            home="FRA",
            away="ARG",
            home_score=1,
            away_score=1,
        ),
        _record(
            match_id="arg-uru-future",
            match_date=date(2026, 5, 20),
            home="ARG",
            away="URU",
            home_score=0,
            away_score=4,
        ),
    ]
    context = PredictionContext(
        match=_target(),
        prediction_time=datetime(2026, 5, 15, 12, 0, tzinfo=UTC),
    )

    vector = build_feature_vector(
        context,
        [RollingFormFeatureProvider(history)],
    )
    values = {value.definition.name: value for value in vector.values}

    assert values["form.home.matches_used_5"].value == pytest.approx(2.0)
    assert values["form.home.points_per_match_5"].value == pytest.approx(2.0)
    assert values["form.home.goal_diff_per_match_5"].value == pytest.approx(1.0)
    assert values["form.home.goals_for_per_match_5"].value == pytest.approx(1.5)
    assert values["form.home.goals_against_per_match_5"].value == pytest.approx(0.5)
    assert values["form.home.points_per_match_5"].lineage.source_record_ids == (
        "source-arg-bra",
        "source-fra-arg",
    )


def test_rolling_form_reports_missing_instead_of_neutral_default() -> None:
    context = PredictionContext(
        match=_target(),
        prediction_time=datetime(2026, 5, 15, 12, 0, tzinfo=UTC),
    )

    vector = build_feature_vector(
        context,
        [RollingFormFeatureProvider([])],
    )
    values = {value.definition.name: value for value in vector.values}

    assert values["form.home.matches_used_5"].value == pytest.approx(0.0)
    assert values["form.home.matches_used_5"].status is FeatureStatus.OBSERVED
    assert values["form.home.points_per_match_5"].value is None
    assert values["form.home.points_per_match_5"].status is FeatureStatus.MISSING
    assert values["form.away.goals_for_per_match_10"].status is FeatureStatus.MISSING


def test_rolling_form_window_size_is_visible_in_matches_used() -> None:
    history = [
        _record(
            match_id=f"arg-opponent-{index}",
            match_date=date(2026, 4, index + 1),
            home="ARG",
            away=f"T{index}",
            home_score=1,
            away_score=0,
        )
        for index in range(7)
    ]
    context = PredictionContext(
        match=_target(),
        prediction_time=datetime(2026, 5, 15, 12, 0, tzinfo=UTC),
    )

    vector = build_feature_vector(
        context,
        [RollingFormFeatureProvider(history)],
    )
    values = {value.definition.name: value for value in vector.values}

    assert values["form.home.matches_used_5"].value == pytest.approx(5.0)
    assert values["form.home.matches_used_10"].value == pytest.approx(7.0)
