from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from football_analytics.data.contracts import SourceMetadata, ensure_utc
from football_analytics.data.scope import GenderCategory, TeamLevel
from football_analytics.domain import MatchStatus


@dataclass(frozen=True, slots=True)
class MatchObservation:
    """Source-level fixture/result observation before canonical entity resolution."""

    source_match_id: str
    kickoff_at: datetime
    home_team_name: str
    away_team_name: str
    competition_name: str
    neutral: bool
    status: MatchStatus
    gender: GenderCategory
    team_level: TeamLevel
    official: bool | None
    metadata: SourceMetadata
    home_score: int | None = None
    away_score: int | None = None
    venue_name: str | None = None

    def __post_init__(self) -> None:
        source_match_id = self.source_match_id.strip()
        home_team_name = self.home_team_name.strip()
        away_team_name = self.away_team_name.strip()
        competition_name = self.competition_name.strip()

        if not source_match_id:
            raise ValueError("source_match_id must not be blank.")

        if not home_team_name or not away_team_name:
            raise ValueError("Source team names must not be blank.")

        if normalize_source_name(home_team_name) == normalize_source_name(away_team_name):
            raise ValueError("A team cannot play itself.")

        if not competition_name:
            raise ValueError("competition_name must not be blank.")

        object.__setattr__(self, "source_match_id", source_match_id)
        object.__setattr__(self, "home_team_name", home_team_name)
        object.__setattr__(self, "away_team_name", away_team_name)
        object.__setattr__(self, "competition_name", competition_name)
        object.__setattr__(
            self,
            "kickoff_at",
            ensure_utc(self.kickoff_at, "kickoff_at"),
        )

        scores = (self.home_score, self.away_score)
        if (scores[0] is None) != (scores[1] is None):
            raise ValueError("home_score and away_score must be supplied together.")

        if scores[0] is not None and (scores[0] < 0 or scores[1] < 0):
            raise ValueError("Match scores cannot be negative.")

        if self.status is MatchStatus.COMPLETED and scores[0] is None:
            raise ValueError("Completed matches require final scores.")

        if self.status is not MatchStatus.COMPLETED and scores[0] is not None:
            raise ValueError("Only completed matches may contain final scores.")

        if self.venue_name is not None:
            value = self.venue_name.strip()
            object.__setattr__(self, "venue_name", value or None)


def normalize_source_name(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())
