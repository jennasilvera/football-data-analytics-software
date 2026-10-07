from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from football_analytics.data.normalization import CanonicalMatchRecord
from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    PredictionContext,
)
from football_analytics.features.history import team_history_before_cutoff


SCHEDULE_FEATURE_VERSION = "schedule_rest_v1"


@dataclass(frozen=True, slots=True)
class _ResolvedRest:
    value: float | None
    status: FeatureStatus
    missing_reason: FeatureMissingReason | None
    imputation_method: str | None
    lineage: FeatureLineage


class ScheduleRestFeatureProvider:
    """Compute rest-day features from canonical prior match history."""

    provider_id = "schedule_rest_features"

    def __init__(
        self,
        records: Sequence[CanonicalMatchRecord],
        *,
        version: str = SCHEDULE_FEATURE_VERSION,
    ) -> None:
        self._records = tuple(records)
        self._version = version.strip()

        if not self._version:
            raise ValueError("version must not be blank.")

        self._definitions = _definitions(version=self._version)

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        home = self._rest_for_team(
            context.match.home_team_id,
            context=context,
        )
        away = self._rest_for_team(
            context.match.away_team_id,
            context=context,
        )

        home_value = _rest_feature_value(
            definition=self._definition("schedule.home.rest_days"),
            resolved=home,
            context=context,
        )
        away_value = _rest_feature_value(
            definition=self._definition("schedule.away.rest_days"),
            resolved=away,
            context=context,
        )

        return (
            home_value,
            away_value,
            _rest_difference(
                definition=self._definition(
                    "schedule.rest_days_diff_home_minus_away"
                ),
                home=home,
                away=away,
                context=context,
            ),
        )

    def _rest_for_team(
        self,
        team_id: str,
        *,
        context: PredictionContext,
    ) -> _ResolvedRest:
        history = team_history_before_cutoff(
            self._records,
            team_id=team_id,
            cutoff=context.prediction_time,
        )

        if not history:
            return _ResolvedRest(
                value=None,
                status=FeatureStatus.MISSING,
                missing_reason=FeatureMissingReason.NO_HISTORY,
                imputation_method=None,
                lineage=FeatureLineage(),
            )

        prior = history[-1]
        lineage = FeatureLineage(
            source_record_ids=(prior.source_match_id,)
        )

        if (
            context.match.kickoff_at is not None
            and prior.match.kickoff_at is not None
        ):
            seconds = (
                context.match.kickoff_at - prior.match.kickoff_at
            ).total_seconds()
            return _ResolvedRest(
                value=seconds / 86_400.0,
                status=FeatureStatus.OBSERVED,
                missing_reason=None,
                imputation_method=None,
                lineage=lineage,
            )

        calendar_days = float(
            (context.match.match_date - prior.match.match_date).days
        )
        return _ResolvedRest(
            value=calendar_days,
            status=FeatureStatus.IMPUTED,
            missing_reason=FeatureMissingReason.TEMPORAL_PRECISION,
            imputation_method="calendar_day_difference_from_date_only_match_time",
            lineage=lineage,
        )

    def _definition(self, name: str) -> FeatureDefinition:
        for definition in self._definitions:
            if definition.name == name:
                return definition
        raise KeyError(name)


def _rest_feature_value(
    *,
    definition: FeatureDefinition,
    resolved: _ResolvedRest,
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


def _rest_difference(
    *,
    definition: FeatureDefinition,
    home: _ResolvedRest,
    away: _ResolvedRest,
    context: PredictionContext,
) -> FeatureValue:
    lineage = FeatureLineage(
        source_record_ids=tuple(
            sorted(
                set(
                    home.lineage.source_record_ids
                    + away.lineage.source_record_ids
                )
            )
        )
    )

    if home.value is None or away.value is None:
        return FeatureValue(
            definition=definition,
            status=FeatureStatus.MISSING,
            as_of=context.prediction_time,
            missing_reason=(
                home.missing_reason
                or away.missing_reason
                or FeatureMissingReason.NO_HISTORY
            ),
            lineage=lineage,
        )

    status = (
        FeatureStatus.OBSERVED
        if home.status is FeatureStatus.OBSERVED
        and away.status is FeatureStatus.OBSERVED
        else FeatureStatus.IMPUTED
    )

    return FeatureValue(
        definition=definition,
        status=status,
        as_of=context.prediction_time,
        value=home.value - away.value,
        missing_reason=(
            None
            if status is FeatureStatus.OBSERVED
            else FeatureMissingReason.TEMPORAL_PRECISION
        ),
        imputation_method=(
            None
            if status is FeatureStatus.OBSERVED
            else "derived_from_calendar_day_rest_approximation"
        ),
        lineage=lineage,
    )


def _definitions(
    *,
    version: str,
) -> tuple[FeatureDefinition, ...]:
    specs = [
        (
            "schedule.home.rest_days",
            "Days between the home team's prior match and target match.",
        ),
        (
            "schedule.away.rest_days",
            "Days between the away team's prior match and target match.",
        ),
        (
            "schedule.rest_days_diff_home_minus_away",
            "Home rest days minus away rest days.",
        ),
    ]

    return tuple(
        FeatureDefinition(
            name=name,
            version=version,
            group="schedule_context",
            description=description,
        )
        for name, description in specs
    )
