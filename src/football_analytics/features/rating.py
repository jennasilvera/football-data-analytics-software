from __future__ import annotations

from collections.abc import Sequence

from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    PredictionContext,
)
from football_analytics.ratings.replay import RatingReplayPrediction


class RatingFeatureProvider:
    """Expose replayed pre-match rating state as versioned model features."""

    provider_id = "rating_features"

    def __init__(
        self,
        replay_predictions: Sequence[RatingReplayPrediction],
        *,
        namespace: str = "elo",
    ) -> None:
        namespace = namespace.strip()
        if not namespace:
            raise ValueError("namespace must not be blank.")

        self._namespace = namespace
        self._by_match_id: dict[str, RatingReplayPrediction] = {}

        for replay_prediction in replay_predictions:
            match_id = replay_prediction.match_id
            if match_id in self._by_match_id:
                raise ValueError(
                    f"Duplicate rating prediction for match_id: {match_id}"
                )
            self._by_match_id[match_id] = replay_prediction

        model_ids = {
            replay_prediction.prediction.model_id
            for replay_prediction in replay_predictions
        }
        if len(model_ids) > 1:
            raise ValueError("RatingFeatureProvider requires one rating model_id.")

        self._model_id = next(iter(model_ids), "unknown_rating_model")
        self._definitions = _definitions(
            namespace=self._namespace,
            version=self._model_id,
        )

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        replay_prediction = self._by_match_id.get(context.match.match_id)

        if replay_prediction is None:
            return tuple(
                FeatureValue(
                    definition=definition,
                    status=FeatureStatus.MISSING,
                    as_of=context.prediction_time,
                    missing_reason=FeatureMissingReason.UPSTREAM_UNAVAILABLE,
                    lineage=FeatureLineage(model_ids=(self._model_id,)),
                )
                for definition in self._definitions
            )

        prediction = replay_prediction.prediction

        if (
            prediction.home_team_id != context.match.home_team_id
            or prediction.away_team_id != context.match.away_team_id
        ):
            raise ValueError(
                "Rating prediction team identities do not match feature context."
            )

        values = {
            f"{self._namespace}.home_rating": prediction.home_rating,
            f"{self._namespace}.away_rating": prediction.away_rating,
            f"{self._namespace}.rating_diff_home_minus_away": (
                prediction.home_rating - prediction.away_rating
            ),
            f"{self._namespace}.expected_home_score": prediction.expected_home_score,
            f"{self._namespace}.expected_away_score": prediction.expected_away_score,
        }
        lineage = FeatureLineage(
            artifact_ids=(context.match.match_id,),
            model_ids=(prediction.model_id,),
        )

        return tuple(
            FeatureValue(
                definition=definition,
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=float(values[definition.name]),
                lineage=lineage,
            )
            for definition in self._definitions
        )


def _definitions(
    *,
    namespace: str,
    version: str,
) -> tuple[FeatureDefinition, ...]:
    specs = [
        ("home_rating", "Home-team pre-match rating."),
        ("away_rating", "Away-team pre-match rating."),
        (
            "rating_diff_home_minus_away",
            "Home rating minus away rating before the match.",
        ),
        ("expected_home_score", "Rating-model expected home score."),
        ("expected_away_score", "Rating-model expected away score."),
    ]

    return tuple(
        FeatureDefinition(
            name=f"{namespace}.{name}",
            version=version,
            group="team_strength",
            description=description,
        )
        for name, description in specs
    )
