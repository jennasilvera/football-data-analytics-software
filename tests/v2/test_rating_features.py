from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from football_analytics.data import CanonicalMatchRecord, LeakageRisk, SourceMetadata
from football_analytics.domain import Match, MatchStatus
from football_analytics.features import (
    FeatureStatus,
    PredictionContext,
    RatingFeatureProvider,
    build_feature_vector,
)
from football_analytics.ratings import LegacyEloRatingEngine, replay_completed_matches


def _completed_record() -> CanonicalMatchRecord:
    return CanonicalMatchRecord(
        match=Match(
            match_id="arg-fra",
            match_date=date(2022, 12, 18),
            home_team_id="ARG",
            away_team_id="FRA",
            competition_id="world-cup",
            neutral=True,
            status=MatchStatus.COMPLETED,
        ),
        metadata=SourceMetadata(
            source="legacy",
            ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
            leakage_risk=LeakageRisk.POST_MATCH_ONLY,
        ),
        source_match_id="legacy-arg-fra",
        source_home_team_name="Argentina",
        source_away_team_name="France",
        source_competition_name="FIFA World Cup",
        home_score=3,
        away_score=3,
    )


def test_replay_captures_pre_match_rating_prediction() -> None:
    replay = replay_completed_matches(
        [_completed_record()],
        engine=LegacyEloRatingEngine(),
        competition_names={"world-cup": "FIFA World Cup"},
    )

    assert len(replay.predictions) == 1
    prediction = replay.predictions[0]
    assert prediction.match_id == "arg-fra"
    assert prediction.prediction.home_rating == pytest.approx(1500.0)
    assert prediction.prediction.away_rating == pytest.approx(1500.0)


def test_rating_provider_builds_versioned_point_in_time_features() -> None:
    record = _completed_record()
    replay = replay_completed_matches(
        [record],
        engine=LegacyEloRatingEngine(),
        competition_names={"world-cup": "FIFA World Cup"},
    )
    context = PredictionContext(
        match=record.match,
        prediction_time=datetime(2022, 12, 17, 23, 59, tzinfo=UTC),
    )

    vector = build_feature_vector(
        context,
        [RatingFeatureProvider(replay.predictions)],
    )

    values = {value.definition.name: value for value in vector.values}
    assert vector.feature_set_id.startswith("features_")
    assert values["elo.home_rating"].value == pytest.approx(1500.0)
    assert values["elo.away_rating"].value == pytest.approx(1500.0)
    assert values["elo.rating_diff_home_minus_away"].value == pytest.approx(0.0)
    assert all(value.status is FeatureStatus.OBSERVED for value in vector.values)
    assert all(
        value.definition.version == "legacy_elo_v1"
        for value in vector.values
    )


def test_rating_provider_marks_absent_upstream_prediction_as_missing() -> None:
    record = _completed_record()
    context = PredictionContext(
        match=record.match,
        prediction_time=datetime(2022, 12, 17, 23, 59, tzinfo=UTC),
    )

    vector = build_feature_vector(
        context,
        [RatingFeatureProvider([])],
    )

    assert all(value.status is FeatureStatus.MISSING for value in vector.values)
    assert all(value.value is None for value in vector.values)


def test_feature_set_id_is_deterministic() -> None:
    record = _completed_record()
    replay = replay_completed_matches(
        [record],
        engine=LegacyEloRatingEngine(),
        competition_names={"world-cup": "FIFA World Cup"},
    )
    context = PredictionContext(
        match=record.match,
        prediction_time=datetime(2022, 12, 17, 23, 59, tzinfo=UTC),
    )
    provider = RatingFeatureProvider(replay.predictions)

    first = build_feature_vector(context, [provider])
    second = build_feature_vector(context, [provider])

    assert first.feature_set_id == second.feature_set_id
