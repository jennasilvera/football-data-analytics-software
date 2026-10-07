from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from football_analytics.data import CanonicalMatchRecord, LeakageRisk, SourceMetadata
from football_analytics.domain import Match, MatchStatus
from football_analytics.ratings import (
    AmbiguousRatingOrderError,
    LegacyEloRatingEngine,
    replay_completed_matches,
)
from wc_forecast.models.elo import EloModel


def _record(
    *,
    match_id: str,
    match_date: date,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    kickoff_at: datetime | None = None,
    status: MatchStatus = MatchStatus.COMPLETED,
) -> CanonicalMatchRecord:
    return CanonicalMatchRecord(
        match=Match(
            match_id=match_id,
            match_date=match_date,
            kickoff_at=kickoff_at,
            home_team_id=home_team,
            away_team_id=away_team,
            competition_id="world-cup",
            neutral=True,
            status=status,
        ),
        metadata=SourceMetadata(
            source="legacy",
            ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
            leakage_risk=LeakageRisk.POST_MATCH_ONLY,
        ),
        source_match_id=f"source-{match_id}",
        source_home_team_name=home_team,
        source_away_team_name=away_team,
        source_competition_name="FIFA World Cup",
        home_score=home_score,
        away_score=away_score,
    )


def test_rating_replay_matches_legacy_sequential_elo_updates() -> None:
    records = [
        _record(
            match_id="cro-mar",
            match_date=date(2022, 12, 17),
            home_team="Croatia",
            away_team="Morocco",
            home_score=2,
            away_score=1,
        ),
        _record(
            match_id="arg-fra",
            match_date=date(2022, 12, 18),
            home_team="Argentina",
            away_team="France",
            home_score=3,
            away_score=3,
        ),
    ]

    legacy = EloModel()
    expected_updates = [
        legacy.update_match(
            home_team="Croatia",
            away_team="Morocco",
            home_score=2,
            away_score=1,
            tournament="FIFA World Cup",
            neutral=True,
        ),
        legacy.update_match(
            home_team="Argentina",
            away_team="France",
            home_score=3,
            away_score=3,
            tournament="FIFA World Cup",
            neutral=True,
        ),
    ]

    replay = replay_completed_matches(
        list(reversed(records)),
        engine=LegacyEloRatingEngine(),
        competition_names={"world-cup": "FIFA World Cup"},
    )

    assert len(replay.updates) == 2
    assert len(replay.snapshots) == 4

    for migrated, expected in zip(replay.updates, expected_updates, strict=True):
        assert migrated.home_rating_before == pytest.approx(
            expected.home_rating_before
        )
        assert migrated.away_rating_before == pytest.approx(
            expected.away_rating_before
        )
        assert migrated.home_rating_after == pytest.approx(
            expected.home_rating_after
        )
        assert migrated.away_rating_after == pytest.approx(
            expected.away_rating_after
        )
        assert migrated.rating_change == pytest.approx(expected.rating_change)


def test_rating_replay_rejects_ambiguous_same_day_date_only_sequence() -> None:
    records = [
        _record(
            match_id="arg-bra-1",
            match_date=date(2026, 6, 1),
            home_team="Argentina",
            away_team="Brazil",
            home_score=1,
            away_score=0,
        ),
        _record(
            match_id="arg-uru-2",
            match_date=date(2026, 6, 1),
            home_team="Argentina",
            away_team="Uruguay",
            home_score=2,
            away_score=0,
        ),
    ]

    with pytest.raises(AmbiguousRatingOrderError, match="kickoff time is unknown"):
        replay_completed_matches(
            records,
            engine=LegacyEloRatingEngine(),
            competition_names={"world-cup": "FIFA World Cup"},
        )


def test_rating_replay_accepts_same_day_sequence_with_exact_kickoffs() -> None:
    records = [
        _record(
            match_id="arg-bra-1",
            match_date=date(2026, 6, 1),
            kickoff_at=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
            home_team="Argentina",
            away_team="Brazil",
            home_score=1,
            away_score=0,
        ),
        _record(
            match_id="arg-uru-2",
            match_date=date(2026, 6, 1),
            kickoff_at=datetime(2026, 6, 1, 20, 0, tzinfo=UTC),
            home_team="Argentina",
            away_team="Uruguay",
            home_score=2,
            away_score=0,
        ),
    ]

    replay = replay_completed_matches(
        list(reversed(records)),
        engine=LegacyEloRatingEngine(),
        competition_names={"world-cup": "FIFA World Cup"},
    )

    assert [update.match_id for update in replay.updates] == [
        "arg-bra-1",
        "arg-uru-2",
    ]


def test_rating_replay_requires_canonical_competition_name() -> None:
    with pytest.raises(KeyError, match="Missing competition name"):
        replay_completed_matches(
            [
                _record(
                    match_id="arg-fra",
                    match_date=date(2022, 12, 18),
                    home_team="Argentina",
                    away_team="France",
                    home_score=3,
                    away_score=3,
                )
            ],
            engine=LegacyEloRatingEngine(),
            competition_names={},
        )


def test_rating_replay_rejects_unplayed_match() -> None:
    record = _record(
        match_id="future",
        match_date=date(2027, 1, 1),
        home_team="Argentina",
        away_team="France",
        home_score=0,
        away_score=0,
        status=MatchStatus.SCHEDULED,
    )

    with pytest.raises(ValueError, match="requires completed matches"):
        replay_completed_matches(
            [record],
            engine=LegacyEloRatingEngine(),
            competition_names={"world-cup": "FIFA World Cup"},
        )
