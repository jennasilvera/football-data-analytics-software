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
from football_analytics.ratings.replay import (
    AmbiguousRatingOrderError,
    RatingReplayPrediction,
    RatingReplayResult,
    replay_completed_matches,
)

__all__ = [
    "AmbiguousRatingOrderError",
    "CompletedMatchRatingInput",
    "LEGACY_ELO_MODEL_ID",
    "LegacyEloRatingEngine",
    "RatingEngine",
    "RatingPrediction",
    "RatingReplayPrediction",
    "RatingReplayResult",
    "RatingSnapshot",
    "RatingUpdate",
    "rating_input_from_record",
    "replay_completed_matches",
]
