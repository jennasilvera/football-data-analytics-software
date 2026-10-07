from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum


class MatchStatus(StrEnum):
    """Lifecycle status for a canonical match."""

    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class MatchTimePrecision(StrEnum):
    """Temporal precision available for a canonical match."""

    DATE_ONLY = "date_only"
    EXACT_KICKOFF = "exact_kickoff"


def _utc_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")

    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class Match:
    """Canonical senior men's A-international fixture or result.

    Historical sources often provide only a match date.  Exact kickoff time is
    therefore optional and its absence is represented explicitly rather than by
    inventing a midnight timestamp.
    """

    match_id: str
    match_date: date
    home_team_id: str
    away_team_id: str
    competition_id: str
    neutral: bool
    kickoff_at: datetime | None = None
    status: MatchStatus = MatchStatus.SCHEDULED
    venue_id: str | None = None

    def __post_init__(self) -> None:
        match_id = self.match_id.strip()
        home_team_id = self.home_team_id.strip()
        away_team_id = self.away_team_id.strip()
        competition_id = self.competition_id.strip()

        if not match_id:
            raise ValueError("match_id must not be blank.")

        if not home_team_id or not away_team_id:
            raise ValueError("home_team_id and away_team_id must not be blank.")

        if home_team_id == away_team_id:
            raise ValueError("A team cannot play itself.")

        if not competition_id:
            raise ValueError("competition_id must not be blank.")

        if not isinstance(self.match_date, date):
            raise TypeError("match_date must be a datetime.date.")

        object.__setattr__(self, "match_id", match_id)
        object.__setattr__(self, "home_team_id", home_team_id)
        object.__setattr__(self, "away_team_id", away_team_id)
        object.__setattr__(self, "competition_id", competition_id)

        if self.kickoff_at is not None:
            object.__setattr__(
                self,
                "kickoff_at",
                _utc_datetime(self.kickoff_at, "kickoff_at"),
            )

        if self.venue_id is not None:
            venue_id = self.venue_id.strip()
            object.__setattr__(self, "venue_id", venue_id or None)

    @property
    def time_precision(self) -> MatchTimePrecision:
        """Return the temporal precision actually supported by source data."""

        if self.kickoff_at is None:
            return MatchTimePrecision.DATE_ONLY

        return MatchTimePrecision.EXACT_KICKOFF
