from __future__ import annotations

from datetime import UTC, date, datetime

from football_analytics.data import (
    CompetitionEntityResolver,
    GenderCategory,
    LeakageRisk,
    MatchObservation,
    NormalizationDecision,
    SourceMetadata,
    TeamEntityResolver,
    TeamLevel,
    normalize_match_batch,
)
from football_analytics.domain import Competition, CompetitionKind, MatchStatus, Team


def _metadata(source_record_id: str) -> SourceMetadata:
    return SourceMetadata(
        source="test",
        source_record_id=source_record_id,
        ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
        available_at=None,
        leakage_risk=LeakageRisk.POST_MATCH_ONLY,
    )


def _observation(
    source_match_id: str,
    home: str,
    away: str,
) -> MatchObservation:
    return MatchObservation(
        source_match_id=source_match_id,
        match_date=date(2022, 12, 18),
        home_team_name=home,
        away_team_name=away,
        competition_name="FIFA World Cup",
        neutral=True,
        status=MatchStatus.COMPLETED,
        gender=GenderCategory.MEN,
        team_level=TeamLevel.SENIOR_A,
        official=True,
        metadata=_metadata(source_match_id),
        home_score=1,
        away_score=0,
    )


def test_batch_partitions_normalized_quarantined_and_excluded_rows() -> None:
    teams = TeamEntityResolver(
        [
            Team(team_id="ARG", name="Argentina"),
            Team(team_id="FRA", name="France"),
        ]
    )
    competitions = CompetitionEntityResolver(
        [
            Competition(
                competition_id="world-cup",
                name="FIFA World Cup",
                kind=CompetitionKind.WORLD_CUP,
            )
        ]
    )

    report = normalize_match_batch(
        [
            _observation("1", "Argentina", "France"),
            _observation("2", "France", "Argentina"),
            _observation("3", "Argentina", "Atlantis"),
        ],
        team_resolver=teams,
        competition_resolver=competitions,
    )

    assert report.total_rows == 3
    assert len(report.normalized) == 1
    assert len(report.quarantined) == 2
    assert len(report.excluded) == 0
    assert report.quarantined[0].decision is NormalizationDecision.QUARANTINED
    assert report.quarantined[0].reasons[0].startswith(
        "duplicate_or_reversed_fixture:"
    )
    assert report.quarantined[1].reasons == ("unresolved_away_team:Atlantis",)
