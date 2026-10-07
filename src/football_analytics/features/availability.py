from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from football_analytics.data.contracts import LeakageRisk, SourceMetadata
from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    PredictionContext,
)

SQUAD_AVAILABILITY_FEATURE_VERSION = "squad_availability_v1"


class PlayerAvailabilityStatus(StrEnum):
    """Descriptive player status retained from a governed availability source."""

    AVAILABLE = "available"
    PROBABLE = "probable"
    QUESTIONABLE = "questionable"
    DOUBTFUL = "doubtful"
    LIMITED = "limited"
    OUT = "out"
    INJURED = "injured"
    SUSPENDED = "suspended"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PlayerAvailabilityObservation:
    """One point-in-time availability assessment for a player and target match.

    The probability is source-provided. V2 does not derive a probability from
    the descriptive status label.
    """

    match_id: str
    team_id: str
    player_id: str
    status: PlayerAvailabilityStatus
    availability_probability: float
    metadata: SourceMetadata
    baseline_expected_minutes: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        match_id = self.match_id.strip()
        team_id = self.team_id.strip()
        player_id = self.player_id.strip()
        probability = float(self.availability_probability)

        if not match_id:
            raise ValueError("match_id must not be blank.")
        if not team_id:
            raise ValueError("team_id must not be blank.")
        if not player_id:
            raise ValueError("player_id must not be blank.")
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError("availability_probability must be between 0 and 1.")

        object.__setattr__(self, "match_id", match_id)
        object.__setattr__(self, "team_id", team_id)
        object.__setattr__(self, "player_id", player_id)
        object.__setattr__(self, "availability_probability", probability)

        if self.baseline_expected_minutes is not None:
            minutes = float(self.baseline_expected_minutes)
            if not math.isfinite(minutes) or minutes < 0.0 or minutes > 120.0:
                raise ValueError(
                    "baseline_expected_minutes must be between 0 and 120."
                )
            object.__setattr__(self, "baseline_expected_minutes", minutes)

        if self.reason is not None:
            reason = self.reason.strip()
            object.__setattr__(self, "reason", reason or None)


@dataclass(frozen=True, slots=True)
class _TeamAvailabilitySummary:
    player_count: int
    mean_probability: float | None
    expected_unavailable_count: float | None
    minutes_coverage_ratio: float | None
    minutes_weighted_availability: float | None
    missing_reason: FeatureMissingReason | None
    lineage: FeatureLineage


class SquadAvailabilityFeatureProvider:
    """Aggregate cutoff-safe player availability into team-level features."""

    provider_id = "squad_availability_features"

    def __init__(
        self,
        observations: Sequence[PlayerAvailabilityObservation],
        *,
        version: str = SQUAD_AVAILABILITY_FEATURE_VERSION,
    ) -> None:
        version = version.strip()
        if not version:
            raise ValueError("version must not be blank.")

        self._version = version
        self._by_match_team: dict[
            tuple[str, str],
            list[PlayerAvailabilityObservation],
        ] = {}
        seen_timestamp: set[tuple[str, str, str, datetime]] = set()

        for observation in observations:
            available_at = observation.metadata.available_at
            if available_at is not None:
                key = (
                    observation.match_id,
                    observation.team_id,
                    observation.player_id,
                    available_at,
                )
                if key in seen_timestamp:
                    raise ValueError(
                        "Duplicate player availability timestamp for "
                        f"{observation.match_id}/{observation.team_id}/"
                        f"{observation.player_id}: {available_at.isoformat()}"
                    )
                seen_timestamp.add(key)

            self._by_match_team.setdefault(
                (observation.match_id, observation.team_id),
                [],
            ).append(observation)

        for team_observations in self._by_match_team.values():
            team_observations.sort(key=_observation_sort_key)

        self._definitions = _definitions(version=version)

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        home = self._summary(
            match_id=context.match.match_id,
            team_id=context.match.home_team_id,
            cutoff=context.prediction_time,
        )
        away = self._summary(
            match_id=context.match.match_id,
            team_id=context.match.away_team_id,
            cutoff=context.prediction_time,
        )

        home_values = self._team_values(
            side="home",
            summary=home,
            context=context,
        )
        away_values = self._team_values(
            side="away",
            summary=away,
            context=context,
        )

        home_by_name = {
            value.definition.name.rsplit(".", 1)[-1]: value
            for value in home_values
        }
        away_by_name = {
            value.definition.name.rsplit(".", 1)[-1]: value
            for value in away_values
        }

        return (
            *home_values,
            *away_values,
            _difference(
                definition=self._definition(
                    "squad.mean_availability_probability_diff_home_minus_away"
                ),
                home=home_by_name["mean_availability_probability"],
                away=away_by_name["mean_availability_probability"],
                context=context,
            ),
            _difference(
                definition=self._definition(
                    "squad.minutes_weighted_availability_diff_home_minus_away"
                ),
                home=home_by_name["minutes_weighted_availability"],
                away=away_by_name["minutes_weighted_availability"],
                context=context,
            ),
        )

    def _summary(
        self,
        *,
        match_id: str,
        team_id: str,
        cutoff: datetime,
    ) -> _TeamAvailabilitySummary:
        all_observations = self._by_match_team.get((match_id, team_id), [])
        eligible = [
            observation
            for observation in all_observations
            if observation.metadata.available_at is not None
            and observation.metadata.available_at <= cutoff
            and observation.metadata.leakage_risk is LeakageRisk.SAFE
        ]

        if not eligible:
            return _TeamAvailabilitySummary(
                player_count=0,
                mean_probability=None,
                expected_unavailable_count=None,
                minutes_coverage_ratio=None,
                minutes_weighted_availability=None,
                missing_reason=(
                    FeatureMissingReason.TEMPORAL_INTEGRITY
                    if all_observations
                    else FeatureMissingReason.UPSTREAM_UNAVAILABLE
                ),
                lineage=FeatureLineage(),
            )

        latest_by_player: dict[str, PlayerAvailabilityObservation] = {}
        for observation in eligible:
            latest_by_player[observation.player_id] = observation

        latest = tuple(latest_by_player.values())
        probabilities = [
            observation.availability_probability
            for observation in latest
        ]
        minute_observations = [
            observation
            for observation in latest
            if observation.baseline_expected_minutes is not None
        ]
        source_record_ids = tuple(
            sorted(
                {
                    observation.metadata.source_record_id
                    for observation in latest
                    if observation.metadata.source_record_id is not None
                }
            )
        )

        player_count = len(latest)
        mean_probability = sum(probabilities) / player_count
        expected_unavailable_count = sum(
            1.0 - probability
            for probability in probabilities
        )
        minutes_coverage_ratio = len(minute_observations) / player_count

        denominator = sum(
            observation.baseline_expected_minutes or 0.0
            for observation in minute_observations
        )
        if denominator > 0.0:
            minutes_weighted_availability = (
                sum(
                    (observation.baseline_expected_minutes or 0.0)
                    * observation.availability_probability
                    for observation in minute_observations
                )
                / denominator
            )
        else:
            minutes_weighted_availability = None

        return _TeamAvailabilitySummary(
            player_count=player_count,
            mean_probability=mean_probability,
            expected_unavailable_count=expected_unavailable_count,
            minutes_coverage_ratio=minutes_coverage_ratio,
            minutes_weighted_availability=minutes_weighted_availability,
            missing_reason=None,
            lineage=FeatureLineage(source_record_ids=source_record_ids),
        )

    def _team_values(
        self,
        *,
        side: str,
        summary: _TeamAvailabilitySummary,
        context: PredictionContext,
    ) -> tuple[FeatureValue, ...]:
        count = FeatureValue(
            definition=self._definition(
                f"squad.{side}.player_observation_count"
            ),
            status=FeatureStatus.OBSERVED,
            as_of=context.prediction_time,
            value=float(summary.player_count),
            lineage=summary.lineage,
        )

        if summary.mean_probability is None:
            reason = (
                summary.missing_reason
                or FeatureMissingReason.UPSTREAM_UNAVAILABLE
            )
            return (
                count,
                *(
                    FeatureValue(
                        definition=self._definition(f"squad.{side}.{name}"),
                        status=FeatureStatus.MISSING,
                        as_of=context.prediction_time,
                        missing_reason=reason,
                        lineage=summary.lineage,
                    )
                    for name in (
                        "mean_availability_probability",
                        "expected_unavailable_player_count",
                        "minutes_coverage_ratio",
                        "minutes_weighted_availability",
                    )
                ),
            )

        assert summary.expected_unavailable_count is not None
        assert summary.minutes_coverage_ratio is not None

        weighted = (
            FeatureValue(
                definition=self._definition(
                    f"squad.{side}.minutes_weighted_availability"
                ),
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=summary.minutes_weighted_availability,
                lineage=summary.lineage,
            )
            if summary.minutes_weighted_availability is not None
            else FeatureValue(
                definition=self._definition(
                    f"squad.{side}.minutes_weighted_availability"
                ),
                status=FeatureStatus.MISSING,
                as_of=context.prediction_time,
                missing_reason=FeatureMissingReason.UPSTREAM_UNAVAILABLE,
                lineage=summary.lineage,
            )
        )

        return (
            count,
            FeatureValue(
                definition=self._definition(
                    f"squad.{side}.mean_availability_probability"
                ),
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=summary.mean_probability,
                lineage=summary.lineage,
            ),
            FeatureValue(
                definition=self._definition(
                    f"squad.{side}.expected_unavailable_player_count"
                ),
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=summary.expected_unavailable_count,
                lineage=summary.lineage,
            ),
            FeatureValue(
                definition=self._definition(
                    f"squad.{side}.minutes_coverage_ratio"
                ),
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=summary.minutes_coverage_ratio,
                lineage=summary.lineage,
            ),
            weighted,
        )

    def _definition(self, name: str) -> FeatureDefinition:
        for definition in self._definitions:
            if definition.name == name:
                return definition
        raise KeyError(name)


def _observation_sort_key(
    observation: PlayerAvailabilityObservation,
) -> tuple[object, ...]:
    available_at = observation.metadata.available_at
    return (
        available_at is None,
        available_at.isoformat() if available_at is not None else "",
        observation.player_id,
        observation.metadata.source_record_id or "",
    )


def _combined_missing_reason(
    home: FeatureValue,
    away: FeatureValue,
) -> FeatureMissingReason:
    reasons = tuple(
        reason
        for reason in (home.missing_reason, away.missing_reason)
        if reason is not None
    )

    if FeatureMissingReason.TEMPORAL_INTEGRITY in reasons:
        return FeatureMissingReason.TEMPORAL_INTEGRITY
    if FeatureMissingReason.STALE in reasons:
        return FeatureMissingReason.STALE
    if reasons:
        return reasons[0]

    return FeatureMissingReason.UPSTREAM_UNAVAILABLE


def _merge_lineage(
    home: FeatureLineage,
    away: FeatureLineage,
) -> FeatureLineage:
    return FeatureLineage(
        source_record_ids=tuple(
            sorted(set(home.source_record_ids + away.source_record_ids))
        ),
        artifact_ids=tuple(
            sorted(set(home.artifact_ids + away.artifact_ids))
        ),
        model_ids=tuple(
            sorted(set(home.model_ids + away.model_ids))
        ),
    )


def _difference(
    *,
    definition: FeatureDefinition,
    home: FeatureValue,
    away: FeatureValue,
    context: PredictionContext,
) -> FeatureValue:
    lineage = _merge_lineage(home.lineage, away.lineage)

    if home.value is None or away.value is None:
        return FeatureValue(
            definition=definition,
            status=FeatureStatus.MISSING,
            as_of=context.prediction_time,
            missing_reason=_combined_missing_reason(home, away),
            lineage=lineage,
        )

    return FeatureValue(
        definition=definition,
        status=FeatureStatus.OBSERVED,
        as_of=context.prediction_time,
        value=home.value - away.value,
        lineage=lineage,
    )


def _definitions(
    *,
    version: str,
) -> tuple[FeatureDefinition, ...]:
    definitions: list[FeatureDefinition] = []
    team_specs = [
        (
            "player_observation_count",
            "Players with cutoff-eligible availability observations.",
        ),
        (
            "mean_availability_probability",
            "Mean source-provided player availability probability.",
        ),
        (
            "expected_unavailable_player_count",
            "Sum of one minus availability probability across observed players.",
        ),
        (
            "minutes_coverage_ratio",
            "Share of observed players with baseline expected minutes.",
        ),
        (
            "minutes_weighted_availability",
            "Availability probability weighted by supplied baseline minutes.",
        ),
    ]

    for side in ("home", "away"):
        for name, description in team_specs:
            definitions.append(
                FeatureDefinition(
                    name=f"squad.{side}.{name}",
                    version=version,
                    group="squad_availability",
                    description=description,
                )
            )

    definitions.extend(
        [
            FeatureDefinition(
                name=(
                    "squad.mean_availability_probability_diff_home_minus_away"
                ),
                version=version,
                group="squad_availability",
                description=(
                    "Home mean availability probability minus away mean "
                    "availability probability."
                ),
            ),
            FeatureDefinition(
                name=(
                    "squad.minutes_weighted_availability_diff_home_minus_away"
                ),
                version=version,
                group="squad_availability",
                description=(
                    "Home minutes-weighted availability minus away "
                    "minutes-weighted availability."
                ),
            ),
        ]
    )

    return tuple(definitions)
