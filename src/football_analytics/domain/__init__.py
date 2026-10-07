"""Canonical football domain types."""

from football_analytics.domain.competitions import Competition, CompetitionKind
from football_analytics.domain.matches import (
    Match,
    MatchStatus,
    MatchTimePrecision,
)
from football_analytics.domain.teams import Confederation, Team

__all__ = [
    "Competition",
    "CompetitionKind",
    "Confederation",
    "Match",
    "MatchStatus",
    "MatchTimePrecision",
    "Team",
]
