from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from football_analytics.domain import (
    Competition,
    CompetitionKind,
    Confederation,
    Match,
    Team,
)


def test_team_normalizes_fifa_code() -> None:
    team = Team(
        team_id="arg",
        name="Argentina",
        fifa_code="arg",
        confederation=Confederation.CONMEBOL,
    )

    assert team.fifa_code == "ARG"


def test_match_normalizes_kickoff_to_utc() -> None:
    kickoff = datetime(
        2026,
        11,
        14,
        20,
        0,
        tzinfo=UTC,
    ).astimezone(timezone(timedelta(hours=-5)))

    match = Match(
        match_id="arg-bra-2026-11-14",
        kickoff_at=kickoff,
        home_team_id="ARG",
        away_team_id="BRA",
        competition_id="friendly",
        neutral=False,
    )

    assert match.kickoff_at.tzinfo is UTC


def test_match_rejects_naive_kickoff() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Match(
            match_id="arg-bra",
            kickoff_at=datetime(2026, 11, 14, 20, 0),
            home_team_id="ARG",
            away_team_id="BRA",
            competition_id="friendly",
            neutral=False,
        )


def test_match_rejects_identical_teams() -> None:
    with pytest.raises(ValueError, match="cannot play itself"):
        Match(
            match_id="arg-arg",
            kickoff_at=datetime(2026, 11, 14, 20, 0, tzinfo=UTC),
            home_team_id="ARG",
            away_team_id="ARG",
            competition_id="friendly",
            neutral=False,
        )


def test_competition_expresses_formal_scope() -> None:
    competition = Competition(
        competition_id="uefa-nations-league",
        name="UEFA Nations League",
        kind=CompetitionKind.NATIONS_LEAGUE,
        confederation=Confederation.UEFA,
    )

    assert competition.senior_mens_a_international is True
