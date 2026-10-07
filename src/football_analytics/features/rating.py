from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    PredictionContext,
)
from football_analytics.ratings.base import RatingSnapshot
from football_analytics.ratings.legacy_elo import LEGACY_ELO_MODEL_ID
from wc_forecast.models.elo import (
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_RATING,
    expected_score,
)


@dataclass(frozen=True, slots=True)
class _ResolvedRating:
    value: float
    status: FeatureStatus
    missing_reason: FeatureMissingReason | None
    imputation_method: str | None
    lineage: FeatureLineage


class LegacyEloFeatureProvider:
    """Build cutoff-safe Elo features from immutable rating snapshots.

    Rating snapshots are currently date-resolution artifacts. To avoid inventing
    within-day result availability, a snapshot is eligible only when its
    effective date is strictly earlier than the prediction cutoff's UTC date.
    """

    provider_id = "legacy_elo_rating_features"

    def __init__(
        self,
        snapshots: Sequence[RatingSnapshot],
        *,
        model_id: str = LEGACY_ELO_MODEL_ID,
        default_rating: float = DEFAULT_RATING,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
        namespace: str = "elo",
    ) -> None:
        namespace = namespace.strip()
        model_id = model_id.strip()

        if not namespace:
            raise ValueError("namespace must not be blank.")
        if not model_id:
            raise ValueError("model_id must not be blank.")

        self._namespace = namespace
        self._model_id = model_id
        self._default_rating = float(default_rating)
        self._home_advantage = float(home_advantage)
        self._snapshots_by_team: dict[str, list[RatingSnapshot]] = {}

        for snapshot in snapshots:
            if snapshot.model_id != self._model_id:
                raise ValueError(
                    "Rating snapshot model_id does not match provider model_id: "
                    f"{snapshot.model_id} != {self._model_id}"
                )
            self._snapshots_by_team.setdefault(snapshot.team_id, []).append(snapshot)

        for team_snapshots in self._snapshots_by_team.values():
            team_snapshots.sort(
                key=lambda snapshot: (
                    snapshot.effective_date,
                    snapshot.source_match_id,
                )
            )

        self._definitions = _definitions(
            namespace=self._namespace,
            version=self._model_id,
        )

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        cutoff_date = context.prediction_time.date()
        home = self._rating_as_of(context.match.home_team_id, cutoff_date)
        away = self._rating_as_of(context.match.away_team_id, cutoff_date)

        effective_home_rating = (
            home.value
            if context.match.neutral
            else home.value + self._home_advantage
        )
        expected_home = expected_score(effective_home_rating, away.value)
        derived_status = (
            FeatureStatus.OBSERVED
            if home.status is FeatureStatus.OBSERVED
            and away.status is FeatureStatus.OBSERVED
            else FeatureStatus.IMPUTED
        )
        derived_reason = (
            None
            if derived_status is FeatureStatus.OBSERVED
            else FeatureMissingReason.NO_HISTORY
        )
        derived_method = (
            None
            if derived_status is FeatureStatus.OBSERVED
            else "derived_from_imputed_legacy_elo_rating"
        )
        derived_lineage = FeatureLineage(
            artifact_ids=tuple(
                sorted(
                    set(
                        home.lineage.artifact_ids
                        + away.lineage.artifact_ids
                    )
                )
            ),
            model_ids=(self._model_id,),
        )

        values_by_name = {
            f"{self._namespace}.home_rating": _feature_value(
                definition=self._definition("home_rating"),
                resolved=home,
                context=context,
            ),
            f"{self._namespace}.away_rating": _feature_value(
                definition=self._definition("away_rating"),
                resolved=away,
                context=context,
            ),
            f"{self._namespace}.rating_diff_home_minus_away": FeatureValue(
                definition=self._definition("rating_diff_home_minus_away"),
                status=derived_status,
                as_of=context.prediction_time,
                value=home.value - away.value,
                missing_reason=derived_reason,
                imputation_method=derived_method,
                lineage=derived_lineage,
            ),
            f"{self._namespace}.expected_home_score": FeatureValue(
                definition=self._definition("expected_home_score"),
                status=derived_status,
                as_of=context.prediction_time,
                value=expected_home,
                missing_reason=derived_reason,
                imputation_method=derived_method,
                lineage=derived_lineage,
            ),
            f"{self._namespace}.expected_away_score": FeatureValue(
                definition=self._definition("expected_away_score"),
                status=derived_status,
                as_of=context.prediction_time,
                value=1.0 - expected_home,
                missing_reason=derived_reason,
                imputation_method=derived_method,
                lineage=derived_lineage,
            ),
        }

        return tuple(
            values_by_name[definition.name]
            for definition in self._definitions
        )

    def _definition(self, suffix: str) -> FeatureDefinition:
        name = f"{self._namespace}.{suffix}"
        for definition in self._definitions:
            if definition.name == name:
                return definition
        raise KeyError(name)

    def _rating_as_of(self, team_id: str, cutoff_date: date) -> _ResolvedRating:
        eligible = [
            snapshot
            for snapshot in self._snapshots_by_team.get(team_id, [])
            if snapshot.effective_date < cutoff_date
        ]

        if eligible:
            snapshot = eligible[-1]
            return _ResolvedRating(
                value=snapshot.rating,
                status=FeatureStatus.OBSERVED,
                missing_reason=None,
                imputation_method=None,
                lineage=FeatureLineage(
                    artifact_ids=(snapshot.source_match_id,),
                    model_ids=(snapshot.model_id,),
                ),
            )

        return _ResolvedRating(
            value=self._default_rating,
            status=FeatureStatus.IMPUTED,
            missing_reason=FeatureMissingReason.NO_HISTORY,
            imputation_method=(
                f"{self._model_id}:default_rating={self._default_rating:g}"
            ),
            lineage=FeatureLineage(model_ids=(self._model_id,)),
        )


def _feature_value(
    *,
    definition: FeatureDefinition,
    resolved: _ResolvedRating,
    context: PredictionContext,
) -> FeatureValue:
    return FeatureValue(
        definition=definition,
        status=resolved.status,
        as_of=context.prediction_time,
        value=resolved.value,
        missing_reason=resolved.missing_reason,
        imputation_method=resolved.imputation_method,
        lineage=resolved.lineage,
    )


def _definitions(
    *,
    namespace: str,
    version: str,
) -> tuple[FeatureDefinition, ...]:
    specs = [
        ("home_rating", "Home-team rating available at the prediction cutoff."),
        ("away_rating", "Away-team rating available at the prediction cutoff."),
        (
            "rating_diff_home_minus_away",
            "Home rating minus away rating at the prediction cutoff.",
        ),
        (
            "expected_home_score",
            "Legacy Elo expected home score at the prediction cutoff.",
        ),
        (
            "expected_away_score",
            "Legacy Elo expected away score at the prediction cutoff.",
        ),
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
