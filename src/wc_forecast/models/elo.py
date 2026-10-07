from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

DEFAULT_RATING = 1500.0
DEFAULT_K_FACTOR = 20.0
DEFAULT_HOME_ADVANTAGE = 75.0


@dataclass(frozen=True)
class EloPrediction:
    """Pre-match Elo prediction for one match."""

    home_team: str
    away_team: str
    home_rating: float
    away_rating: float
    expected_home_score: float
    expected_away_score: float


@dataclass(frozen=True)
class EloUpdate:
    """Post-match Elo update details."""

    home_team: str
    away_team: str
    home_rating_before: float
    away_rating_before: float
    home_rating_after: float
    away_rating_after: float
    expected_home_score: float
    actual_home_score: float
    rating_change: float


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return expected score for team A against team B."""

    return float(1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)))


def actual_score(goals_for: int, goals_against: int) -> float:
    """Convert a match result into Elo score: win=1, draw=0.5, loss=0."""

    if goals_for > goals_against:
        return 1.0

    if goals_for == goals_against:
        return 0.5

    return 0.0


def match_importance_weight(tournament: str) -> float:
    """Assign a practical importance weight based on tournament type."""

    name = tournament.strip().lower()

    if name == "fifa world cup":
        return 1.60

    if "world cup" in name and "qualification" in name:
        return 1.25

    if any(
        token in name
        for token in [
            "euro",
            "copa america",
            "african cup",
            "asian cup",
            "gold cup",
            "nations league",
        ]
    ):
        return 1.15

    if "friendly" in name:
        return 0.75

    return 1.00


def margin_of_victory_multiplier(
    goals_for: int,
    goals_against: int,
    rating_difference: float,
) -> float:
    """Scale Elo movement by goal margin while damping favorite blowouts."""

    goal_difference = abs(goals_for - goals_against)

    if goal_difference <= 1:
        return 1.0

    margin_component = math.log(goal_difference + 1.0)
    rating_component = 2.2 / ((0.001 * abs(rating_difference)) + 2.2)

    return margin_component * rating_component


class EloModel:
    """Custom Elo model for international football matches."""

    def __init__(
        self,
        default_rating: float = DEFAULT_RATING,
        k_factor: float = DEFAULT_K_FACTOR,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    ) -> None:
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Return current team rating, initializing unseen teams."""

        if team not in self.ratings:
            self.ratings[team] = self.default_rating

        return self.ratings[team]

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool,
    ) -> EloPrediction:
        """Generate pre-match Elo expected scores for both teams."""

        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        effective_home_rating = home_rating if neutral else home_rating + self.home_advantage
        expected_home = expected_score(effective_home_rating, away_rating)

        return EloPrediction(
            home_team=home_team,
            away_team=away_team,
            home_rating=home_rating,
            away_rating=away_rating,
            expected_home_score=expected_home,
            expected_away_score=1.0 - expected_home,
        )

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str,
        neutral: bool,
    ) -> EloUpdate:
        """Update Elo ratings after one completed match."""

        prediction = self.predict_match(
            home_team=home_team,
            away_team=away_team,
            neutral=neutral,
        )

        actual_home = actual_score(home_score, away_score)

        effective_home_rating = (
            prediction.home_rating if neutral else prediction.home_rating + self.home_advantage
        )
        rating_difference = effective_home_rating - prediction.away_rating

        importance = match_importance_weight(tournament)
        margin_multiplier = margin_of_victory_multiplier(
            goals_for=home_score,
            goals_against=away_score,
            rating_difference=rating_difference,
        )

        rating_change = (
            self.k_factor
            * importance
            * margin_multiplier
            * (actual_home - prediction.expected_home_score)
        )

        home_rating_after = prediction.home_rating + rating_change
        away_rating_after = prediction.away_rating - rating_change

        self.ratings[home_team] = home_rating_after
        self.ratings[away_team] = away_rating_after

        return EloUpdate(
            home_team=home_team,
            away_team=away_team,
            home_rating_before=prediction.home_rating,
            away_rating_before=prediction.away_rating,
            home_rating_after=home_rating_after,
            away_rating_after=away_rating_after,
            expected_home_score=prediction.expected_home_score,
            actual_home_score=actual_home,
            rating_change=rating_change,
        )

    def fit(self, results: pd.DataFrame) -> pd.DataFrame:
        """Replay matches chronologically and return match-level Elo history."""

        updates: list[dict[str, object]] = []

        sorted_results = results.sort_values(["date", "home_team", "away_team"]).reset_index(
            drop=True
        )

        for row in sorted_results.itertuples(index=False):
            update = self.update_match(
                home_team=row.home_team,
                away_team=row.away_team,
                home_score=int(row.home_score),
                away_score=int(row.away_score),
                tournament=row.tournament,
                neutral=bool(row.neutral),
            )

            updates.append(
                {
                    "date": row.date,
                    "home_team": update.home_team,
                    "away_team": update.away_team,
                    "home_score": int(row.home_score),
                    "away_score": int(row.away_score),
                    "tournament": row.tournament,
                    "neutral": bool(row.neutral),
                    "home_rating_before": update.home_rating_before,
                    "away_rating_before": update.away_rating_before,
                    "home_rating_after": update.home_rating_after,
                    "away_rating_after": update.away_rating_after,
                    "expected_home_score": update.expected_home_score,
                    "expected_away_score": 1.0 - update.expected_home_score,
                    "actual_home_score": update.actual_home_score,
                    "rating_change": update.rating_change,
                }
            )

        return pd.DataFrame(updates)

    def ratings_table(self) -> pd.DataFrame:
        """Return current ratings as a sorted dataframe."""

        rows = [
            {"team": team, "elo_rating": rating}
            for team, rating in sorted(
                self.ratings.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        return pd.DataFrame(rows)
