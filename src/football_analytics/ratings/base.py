from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CompletedMatchRatingInput:
    """Canonical completed-match input consumed by rating engines."""

    match_id: str
    match_date: date
    home_team_id: str
    away_team_id: str
    home_score: int
    away_score: int
    competition_name: str
    neutral: bool

    def __post_init__(self) -> None:
        if not self.match_id.strip():
            raise ValueError("match_id must not be blank.")

        if not self.home_team_id.strip() or not self.away_team_id.strip():
            raise ValueError("team IDs must not be blank.")

        if self.home_team_id == self.away_team_id:
            raise ValueError("A team cannot play itself.")

        if self.home_score < 0 or self.away_score < 0:
            raise ValueError("Scores cannot be negative.")

        if not self.competition_name.strip():
            raise ValueError("competition_name must not be blank.")


@dataclass(frozen=True, slots=True)
class RatingPrediction:
    """Pre-match rating view for two canonical teams."""

    model_id: str
    home_team_id: str
    away_team_id: str
    home_rating: float
    away_rating: float
    expected_home_score: float
    expected_away_score: float


@dataclass(frozen=True, slots=True)
class RatingUpdate:
    """Immutable result of applying one completed match to a rating engine."""

    model_id: str
    match_id: str
    home_team_id: str
    away_team_id: str
    home_rating_before: float
    away_rating_before: float
    home_rating_after: float
    away_rating_after: float
    expected_home_score: float
    actual_home_score: float
    rating_change: float


@dataclass(frozen=True, slots=True)
class RatingSnapshot:
    """Immutable team rating state after a specific completed match."""

    model_id: str
    team_id: str
    rating: float
    effective_date: date
    source_match_id: str


class RatingEngine(Protocol):
    """Common contract for replaceable national-team rating systems."""

    model_id: str

    def predict(
        self,
        *,
        home_team_id: str,
        away_team_id: str,
        neutral: bool,
    ) -> RatingPrediction:
        """Return a pre-match rating prediction."""

    def update(self, match: CompletedMatchRatingInput) -> RatingUpdate:
        """Apply one completed match."""

    def snapshots(self) -> tuple[RatingSnapshot, ...]:
        """Return immutable snapshots emitted by the engine."""
