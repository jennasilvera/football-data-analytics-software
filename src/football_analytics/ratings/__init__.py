"""Rating-engine contracts and migration adapters."""

from football_analytics.ratings.base import (
    CompletedMatchRatingInput,
    RatingEngine,
    RatingPrediction,
    RatingSnapshot,
    RatingUpdate,
)
from football_analytics.ratings.legacy_elo import (
    LEGACY_ELO_MODEL_ID,
    LegacyEloRatingEngine,
    rating_input_from_record,
)

__all__ = [
    "CompletedMatchRatingInput",
    "LEGACY_ELO_MODEL_ID",
    "LegacyEloRatingEngine",
    "RatingEngine",
    "RatingPrediction",
    "RatingSnapshot",
    "RatingUpdate",
    "rating_input_from_record",
]
