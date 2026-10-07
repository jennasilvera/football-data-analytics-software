from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from football_analytics.data import (
    CompetitionEntityResolver,
    GenderCategory,
    LeakageRisk,
    MatchObservation,
    SourceMetadata,
    TeamEntityResolver,
    TeamLevel,
    dataframe_snapshot,
    normalization_frames,
    normalize_match_batch,
)
from football_analytics.domain import Competition, CompetitionKind, MatchStatus, Team


def test_dataframe_snapshot_is_deterministic_for_same_content() -> None:
    first = pd.DataFrame({"b": [2, 3], "a": [1, 4]})
    second = pd.DataFrame({"a": [1, 4], "b": [2, 3]})

    assert dataframe_snapshot(first) == dataframe_snapshot(second)


def test_normalization_artifacts_preserve_lineage_and_quarantine_identity() -> None:
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
    metadata = SourceMetadata(
        source="legacy",
        ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
        leakage_risk=LeakageRisk.POST_MATCH_ONLY,
    )

    observations = [
        MatchObservation(
            source_match_id="known-1",
            match_date=date(2022, 12, 18),
            home_team_name="Argentina",
            away_team_name="France",
            competition_name="FIFA World Cup",
            neutral=True,
            status=MatchStatus.COMPLETED,
            gender=GenderCategory.MEN,
            team_level=TeamLevel.SENIOR_A,
            official=True,
            metadata=metadata,
            home_score=3,
            away_score=3,
        ),
        MatchObservation(
            source_match_id="unknown-1",
            match_date=date(2022, 12, 19),
            home_team_name="Argentina",
            away_team_name="Atlantis",
            competition_name="FIFA World Cup",
            neutral=True,
            status=MatchStatus.COMPLETED,
            gender=GenderCategory.MEN,
            team_level=TeamLevel.SENIOR_A,
            official=True,
            metadata=metadata,
            home_score=2,
            away_score=0,
        ),
    ]

    report = normalize_match_batch(
        observations,
        team_resolver=teams,
        competition_resolver=competitions,
    )
    frames = normalization_frames(report)

    assert frames["normalized"].loc[0, "source_match_id"] == "known-1"
    assert frames["normalized"].loc[0, "leakage_risk"] == "post_match_only"
    assert frames["quarantined"].loc[0, "source_match_id"] == "unknown-1"
    assert "unresolved_away_team:Atlantis" in frames["quarantined"].loc[0, "reasons"]
