from __future__ import annotations

import pytest

from football_analytics.data import (
    CompetitionEntityResolver,
    ResolutionStatus,
    TeamEntityResolver,
)
from football_analytics.domain import Competition, CompetitionKind, Team


def test_team_resolver_uses_explicit_alias() -> None:
    resolver = TeamEntityResolver(
        [Team(team_id="USA", name="United States", fifa_code="USA")],
        aliases={"USA": "USA", "United States of America": "USA"},
    )

    resolution = resolver.resolve("United States of America")

    assert resolution.status is ResolutionStatus.RESOLVED
    assert resolution.team is not None
    assert resolution.team.team_id == "USA"
    assert resolution.matched_by == "alias"


def test_team_resolver_does_not_guess_unknown_team() -> None:
    resolver = TeamEntityResolver(
        [Team(team_id="ARG", name="Argentina", fifa_code="ARG")]
    )

    resolution = resolver.resolve("Atlantis")

    assert resolution.status is ResolutionStatus.UNRESOLVED
    assert resolution.team is None


def test_team_resolver_rejects_alias_for_unknown_id() -> None:
    with pytest.raises(ValueError, match="unknown team_id"):
        TeamEntityResolver(
            [Team(team_id="ARG", name="Argentina", fifa_code="ARG")],
            aliases={"Argentine": "MISSING"},
        )


def test_competition_resolver_uses_canonical_name() -> None:
    resolver = CompetitionEntityResolver(
        [
            Competition(
                competition_id="fifa-world-cup",
                name="FIFA World Cup",
                kind=CompetitionKind.WORLD_CUP,
            )
        ]
    )

    resolution = resolver.resolve(" FIFA World Cup ")

    assert resolution.status is ResolutionStatus.RESOLVED
    assert resolution.competition is not None
    assert resolution.competition.competition_id == "fifa-world-cup"
    assert resolution.matched_by == "canonical_name"
