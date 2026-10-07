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


DEFAULT_FORM_WINDOWS = (5, 10)
ROLLING_FORM_VERSION = "rolling_form_v1"


@dataclass(frozen=True, slots=True)
class _TeamMatchForm:
    points: float
    goal_diff: float
    goals_for: float
    goals_against: float
    source_record_id: str


class RollingFormFeatureProvider:
    """Compute leakage-safe rolling form from canonical completed matches."""

    provider_id = "rolling_form_features"

    def __init__(
        self,
        records: Sequence[CanonicalMatchRecord],
        *,
        windows: tuple[int, ...] = DEFAULT_FORM_WINDOWS,
        version: str = ROLLING_FORM_VERSION,
    ) -> None:
        if not windows:
            raise ValueError("At least one rolling-form window is required.")

        if any(window <= 0 for window in windows):
            raise ValueError("Rolling-form windows must be positive.")

        if len(set(windows)) != len(windows):
            raise ValueError("Rolling-form windows must be unique.")

        self._records = tuple(records)
        self._windows = tuple(sorted(windows))
        self._version = version.strip()

        if not self._version:
            raise ValueError("version must not be blank.")

        self._definitions = _definitions(
            windows=self._windows,
            version=self._version,
        )

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        values: list[FeatureValue] = []

        for side, team_id in (
            ("home", context.match.home_team_id),
            ("away", context.match.away_team_id),
        ):
            history = team_history_before_cutoff(
                self._records,
                team_id=team_id,
                cutoff=context.prediction_time,
            )
            perspective = tuple(
                _team_match_form(record, team_id)
                for record in history
            )

            for window in self._windows:
                recent = perspective[-window:]
                values.extend(
                    self._window_features(
                        side=side,
                        window=window,
                        recent=recent,
                        context=context,
                    )
                )

        return tuple(values)

    def _window_features(
        self,
        *,
        side: str,
        window: int,
        recent: tuple[_TeamMatchForm, ...],
        context: PredictionContext,
    ) -> tuple[FeatureValue, ...]:
        count_definition = self._definition(
            f"form.{side}.matches_used_{window}"
        )

        if not recent:
            missing = FeatureValue(
                definition=count_definition,
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=0.0,
            )
            metric_values = tuple(
                FeatureValue(
                    definition=self._definition(
                        f"form.{side}.{metric_name}_{window}"
                    ),
                    status=FeatureStatus.MISSING,
                    as_of=context.prediction_time,
                    missing_reason=FeatureMissingReason.NO_HISTORY,
                )
                for metric_name in (
                    "points_per_match",
                    "goal_diff_per_match",
                    "goals_for_per_match",
                    "goals_against_per_match",
                )
            )
            return (missing, *metric_values)

        lineage = FeatureLineage(
            source_record_ids=tuple(
                item.source_record_id for item in recent
            )
        )
        count = float(len(recent))

        metrics = {
            "points_per_match": sum(item.points for item in recent) / count,
            "goal_diff_per_match": sum(item.goal_diff for item in recent) / count,
            "goals_for_per_match": sum(item.goals_for for item in recent) / count,
            "goals_against_per_match": (
                sum(item.goals_against for item in recent) / count
            ),
        }

        return (
            FeatureValue(
                definition=count_definition,
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=count,
                lineage=lineage,
            ),
            *(
                FeatureValue(
                    definition=self._definition(
                        f"form.{side}.{metric_name}_{window}"
                    ),
                    status=FeatureStatus.OBSERVED,
                    as_of=context.prediction_time,
                    value=float(metric_value),
                    lineage=lineage,
                )
                for metric_name, metric_value in metrics.items()
            ),
        )

    def _definition(self, name: str) -> FeatureDefinition:
        for definition in self._definitions:
            if definition.name == name:
                return definition
        raise KeyError(name)


def _team_match_form(
    record: CanonicalMatchRecord,
    team_id: str,
) -> _TeamMatchForm:
    assert record.home_score is not None
    assert record.away_score is not None

    if team_id == record.match.home_team_id:
        goals_for = record.home_score
        goals_against = record.away_score
    elif team_id == record.match.away_team_id:
        goals_for = record.away_score
        goals_against = record.home_score
    else:
        raise ValueError(
            f"Team {team_id} does not participate in {record.match.match_id}."
        )

    if goals_for > goals_against:
        points = 3.0
    elif goals_for < goals_against:
        points = 0.0
    else:
        points = 1.0

    return _TeamMatchForm(
        points=points,
        goal_diff=float(goals_for - goals_against),
        goals_for=float(goals_for),
        goals_against=float(goals_against),
        source_record_id=record.source_match_id,
    )


def _definitions(
    *,
    windows: tuple[int, ...],
    version: str,
) -> tuple[FeatureDefinition, ...]:
    definitions: list[FeatureDefinition] = []

    metric_descriptions = {
        "matches_used": "Completed matches contributing to the rolling window.",
        "points_per_match": "Average points per match in the rolling window.",
        "goal_diff_per_match": "Average goal difference in the rolling window.",
        "goals_for_per_match": "Average goals scored in the rolling window.",
        "goals_against_per_match": "Average goals conceded in the rolling window.",
    }

    for side in ("home", "away"):
        for window in windows:
            for metric_name, description in metric_descriptions.items():
                definitions.append(
                    FeatureDefinition(
                        name=f"form.{side}.{metric_name}_{window}",
                        version=version,
                        group="recent_form",
                        description=description,
                    )
                )

    return tuple(definitions)
