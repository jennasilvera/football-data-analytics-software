from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from football_analytics.data import GenderCategory, LeakageRisk, TeamLevel
from football_analytics.data.adapters import (
    TabularMatchColumns,
    TabularSourcePolicy,
    build_match_observations,
)
from football_analytics.domain import MatchStatus


def _historical_policy() -> TabularSourcePolicy:
    return TabularSourcePolicy(
        source_id="historical_results",
        gender=GenderCategory.MEN,
        team_level=TeamLevel.SENIOR_A,
        official=True,
        default_status=MatchStatus.COMPLETED,
        leakage_risk=LeakageRisk.POST_MATCH_ONLY,
        source_version="archive-2026-10-07",
        legal_use_notes="Test source.",
    )


def test_date_only_historical_results_do_not_invent_kickoff_or_availability() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2001-09-05"],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "home_score": [2],
            "away_score": [1],
            "tournament": ["Friendly"],
            "neutral": [False],
        }
    )

    observations = build_match_observations(
        frame,
        columns=TabularMatchColumns(
            match_date="date",
            home_team="home_team",
            away_team="away_team",
            competition="tournament",
            neutral="neutral",
            home_score="home_score",
            away_score="away_score",
        ),
        policy=_historical_policy(),
        ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
    )

    observation = observations[0]
    assert observation.kickoff_at is None
    assert observation.metadata.available_at is None
    assert observation.metadata.leakage_risk is LeakageRisk.POST_MATCH_ONLY
    assert observation.home_score == 2
    assert observation.status is MatchStatus.COMPLETED


def test_fixture_feed_preserves_exact_timezone_aware_kickoff() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-11-14"],
            "kickoff_at": ["2026-11-14T20:00:00Z"],
            "available_at": ["2026-11-14T16:00:00Z"],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "tournament": ["Friendly"],
            "neutral": [True],
        }
    )

    policy = TabularSourcePolicy(
        source_id="fixture_feed",
        gender=GenderCategory.MEN,
        team_level=TeamLevel.SENIOR_A,
        official=True,
        default_status=MatchStatus.SCHEDULED,
        leakage_risk=LeakageRisk.SAFE,
    )

    observations = build_match_observations(
        frame,
        columns=TabularMatchColumns(
            match_date="date",
            kickoff_at="kickoff_at",
            available_at="available_at",
            home_team="home_team",
            away_team="away_team",
            competition="tournament",
            neutral="neutral",
        ),
        policy=policy,
        ingested_at=datetime(2026, 11, 14, 16, 5, tzinfo=UTC),
    )

    assert observations[0].kickoff_at == datetime(2026, 11, 14, 20, 0, tzinfo=UTC)
    assert observations[0].metadata.available_at == datetime(
        2026, 11, 14, 16, 0, tzinfo=UTC
    )


def test_naive_source_timestamp_is_rejected_instead_of_assumed_utc() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-11-14"],
            "kickoff_at": ["2026-11-14 20:00:00"],
            "home_team": ["Argentina"],
            "away_team": ["Brazil"],
            "tournament": ["Friendly"],
            "neutral": [True],
        }
    )

    with pytest.raises(ValueError, match="timezone information"):
        build_match_observations(
            frame,
            columns=TabularMatchColumns(
                match_date="date",
                kickoff_at="kickoff_at",
                home_team="home_team",
                away_team="away_team",
                competition="tournament",
                neutral="neutral",
            ),
            policy=TabularSourcePolicy(
                source_id="fixture_feed",
                gender=GenderCategory.MEN,
                team_level=TeamLevel.SENIOR_A,
                official=True,
                default_status=MatchStatus.SCHEDULED,
                leakage_risk=LeakageRisk.SAFE,
            ),
            ingested_at=datetime(2026, 11, 14, 16, 5, tzinfo=UTC),
        )
