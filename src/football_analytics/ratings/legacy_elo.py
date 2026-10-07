from __future__ import annotations

from football_analytics.data.normalization import CanonicalMatchRecord
from football_analytics.domain import MatchStatus
from football_analytics.domain.scores import require_regulation_score
from football_analytics.ratings.base import (
    CompletedMatchRatingInput,
    RatingPrediction,
    RatingSnapshot,
    RatingUpdate,
)
from wc_forecast.models.elo import (
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_K_FACTOR,
    DEFAULT_RATING,
    EloModel,
)

LEGACY_ELO_MODEL_ID = "legacy_elo_v1"


class LegacyEloRatingEngine:
    """V2 adapter around the tested legacy Elo mathematics.

    This migration adapter intentionally preserves the existing Elo behavior.
    Algorithm changes belong in later research commits after parity is proven.
    """

    model_id = LEGACY_ELO_MODEL_ID

    def __init__(
        self,
        *,
        default_rating: float = DEFAULT_RATING,
        k_factor: float = DEFAULT_K_FACTOR,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    ) -> None:
        self._model = EloModel(
            default_rating=default_rating,
            k_factor=k_factor,
            home_advantage=home_advantage,
        )
        self._snapshots: list[RatingSnapshot] = []

    def predict(
        self,
        *,
        home_team_id: str,
        away_team_id: str,
        neutral: bool,
    ) -> RatingPrediction:
        prediction = self._model.predict_match(
            home_team=home_team_id,
            away_team=away_team_id,
            neutral=neutral,
        )

        return RatingPrediction(
            model_id=self.model_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_rating=prediction.home_rating,
            away_rating=prediction.away_rating,
            expected_home_score=prediction.expected_home_score,
            expected_away_score=prediction.expected_away_score,
        )

    def update(self, match: CompletedMatchRatingInput) -> RatingUpdate:
        update = self._model.update_match(
            home_team=match.home_team_id,
            away_team=match.away_team_id,
            home_score=match.home_score,
            away_score=match.away_score,
            tournament=match.competition_name,
            neutral=match.neutral,
        )

        self._snapshots.extend(
            [
                RatingSnapshot(
                    model_id=self.model_id,
                    team_id=match.home_team_id,
                    rating=update.home_rating_after,
                    effective_date=match.match_date,
                    source_match_id=match.match_id,
                ),
                RatingSnapshot(
                    model_id=self.model_id,
                    team_id=match.away_team_id,
                    rating=update.away_rating_after,
                    effective_date=match.match_date,
                    source_match_id=match.match_id,
                ),
            ]
        )

        return RatingUpdate(
            model_id=self.model_id,
            match_id=match.match_id,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            home_rating_before=update.home_rating_before,
            away_rating_before=update.away_rating_before,
            home_rating_after=update.home_rating_after,
            away_rating_after=update.away_rating_after,
            expected_home_score=update.expected_home_score,
            actual_home_score=update.actual_home_score,
            rating_change=update.rating_change,
        )

    def snapshots(self) -> tuple[RatingSnapshot, ...]:
        return tuple(self._snapshots)


def rating_input_from_record(
    record: CanonicalMatchRecord,
    *,
    competition_name: str,
) -> CompletedMatchRatingInput:
    """Convert a canonical completed match into rating-engine input."""

    if record.match.status is not MatchStatus.COMPLETED:
        raise ValueError("Rating updates require a completed match.")

    if record.home_score is None or record.away_score is None:
        raise ValueError("Rating updates require final scores.")

    require_regulation_score(record.score_basis, match_id=record.match.match_id)
    return CompletedMatchRatingInput(
        match_id=record.match.match_id,
        match_date=record.match.match_date,
        home_team_id=record.match.home_team_id,
        away_team_id=record.match.away_team_id,
        home_score=record.home_score,
        away_score=record.away_score,
        competition_name=competition_name,
        neutral=record.match.neutral,
    )
