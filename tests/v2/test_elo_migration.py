from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from football_analytics.data import CanonicalMatchRecord, LeakageRisk, SourceMetadata
from football_analytics.domain import Match, MatchStatus
from football_analytics.ratings import (
    LEGACY_ELO_MODEL_ID,
    LegacyEloRatingEngine,
    rating_input_from_record,
)
from wc_forecast.models.elo import EloModel


def _rating_input_record() -> CanonicalMatchRecord:
    return CanonicalMatchRecord(
        match=Match(
            match_id="match-arg-fra",
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
        source_match_id="legacy-1",
        source_home_team_name="Argentina",
        source_away_team_name="France",
        source_competition_name="FIFA World Cup",
        home_score=3,
        away_score=3,
    )


def test_v2_elo_adapter_matches_legacy_update_math() -> None:
    legacy = EloModel()
    migrated = LegacyEloRatingEngine()

    legacy_update = legacy.update_match(
        home_team="ARG",
        away_team="FRA",
        home_score=3,
        away_score=3,
        tournament="FIFA World Cup",
        neutral=True,
    )
    migrated_update = migrated.update(
        rating_input_from_record(
            _rating_input_record(),
            competition_name="FIFA World Cup",
        )
    )

    assert migrated_update.model_id == LEGACY_ELO_MODEL_ID
    assert migrated_update.home_rating_before == pytest.approx(
        legacy_update.home_rating_before
    )
    assert migrated_update.away_rating_before == pytest.approx(
        legacy_update.away_rating_before
    )
    assert migrated_update.home_rating_after == pytest.approx(
        legacy_update.home_rating_after
    )
    assert migrated_update.away_rating_after == pytest.approx(
        legacy_update.away_rating_after
    )
    assert migrated_update.rating_change == pytest.approx(legacy_update.rating_change)


def test_v2_elo_prediction_matches_legacy_prediction_math() -> None:
    legacy = EloModel()
    migrated = LegacyEloRatingEngine()

    legacy_prediction = legacy.predict_match(
        home_team="ARG",
        away_team="BRA",
        neutral=False,
    )
    migrated_prediction = migrated.predict(
        home_team_id="ARG",
        away_team_id="BRA",
        neutral=False,
    )

    assert migrated_prediction.home_rating == pytest.approx(
        legacy_prediction.home_rating
    )
    assert migrated_prediction.away_rating == pytest.approx(
        legacy_prediction.away_rating
    )
    assert migrated_prediction.expected_home_score == pytest.approx(
        legacy_prediction.expected_home_score
    )


def test_v2_elo_emits_immutable_canonical_snapshots() -> None:
    engine = LegacyEloRatingEngine()
    engine.update(
        rating_input_from_record(
            _rating_input_record(),
            competition_name="FIFA World Cup",
        )
    )

    snapshots = engine.snapshots()

    assert len(snapshots) == 2
    assert {snapshot.team_id for snapshot in snapshots} == {"ARG", "FRA"}
    assert {snapshot.source_match_id for snapshot in snapshots} == {
        "match-arg-fra"
    }
