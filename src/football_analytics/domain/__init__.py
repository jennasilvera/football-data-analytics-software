"""Canonical football domain types."""

from football_analytics.domain.competitions import (
    Competition,
    CompetitionKind,
    CompetitionStage,
)
from football_analytics.domain.locations import GeographicPoint, Venue
from football_analytics.domain.matches import (
    Match,
    MatchOutcome,
    MatchStatus,
    MatchTimePrecision,
)
from football_analytics.domain.teams import Confederation, Team

__all__ = [
    "Competition",
    "CompetitionKind",
    "CompetitionStage",
    "Confederation",
    "GeographicPoint",
    "Match",
    "MatchOutcome",
    "MatchStatus",
    "MatchTimePrecision",
    "Team",
    "Venue",
]
