from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from football_analytics.data.contracts import LeakageRisk, SourceMetadata
from football_analytics.domain import GeographicPoint
from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    PredictionContext,
)

TRAVEL_CONTEXT_FEATURE_VERSION = "travel_reference_v1"
EARTH_MEAN_RADIUS_KM = 6371.0088


@dataclass(frozen=True, slots=True)
class VenueLocationObservation:
    """Point-in-time geospatial observation for one canonical venue."""

    venue_id: str
    location: GeographicPoint
    metadata: SourceMetadata

    def __post_init__(self) -> None:
        venue_id = self.venue_id.strip()
        if not venue_id:
            raise ValueError("venue_id must not be blank.")
        object.__setattr__(self, "venue_id", venue_id)


@dataclass(frozen=True, slots=True)
class TeamReferenceLocationObservation:
    """Point-in-time geographic reference for a national team.

    The reference point is an analytical baseline, not an assertion about the
    team's actual departure airport, camp, or itinerary.
    """

    team_id: str
    location: GeographicPoint
    metadata: SourceMetadata
    reference_type: str = "national_reference"

    def __post_init__(self) -> None:
        team_id = self.team_id.strip()
        reference_type = self.reference_type.strip()

        if not team_id:
            raise ValueError("team_id must not be blank.")
        if not reference_type:
            raise ValueError("reference_type must not be blank.")

        object.__setattr__(self, "team_id", team_id)
        object.__setattr__(self, "reference_type", reference_type)


@dataclass(frozen=True, slots=True)
class _ResolvedLocation:
    location: GeographicPoint | None
    missing_reason: FeatureMissingReason | None
    lineage: FeatureLineage


class TravelContextFeatureProvider:
    """Build cutoff-safe venue and reference-distance context features.

    Distances are great-circle distances from governed team reference points to
    the governed match venue. They must not be interpreted as actual itinerary
    or flight distance.
    """

    provider_id = "travel_context_features"

    def __init__(
        self,
        *,
        venue_locations: Sequence[VenueLocationObservation],
        team_reference_locations: Sequence[TeamReferenceLocationObservation],
        version: str = TRAVEL_CONTEXT_FEATURE_VERSION,
    ) -> None:
        version = version.strip()
        if not version:
            raise ValueError("version must not be blank.")

        self._version = version
        self._venue_locations = _index_venue_observations(venue_locations)
        self._team_locations = _index_team_observations(team_reference_locations)
        self._definitions = _definitions(version=version)

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        neutral = FeatureValue(
            definition=self._definition("context.match.neutral_site"),
            status=FeatureStatus.OBSERVED,
            as_of=context.prediction_time,
            value=1.0 if context.match.neutral else 0.0,
        )

        venue = self._resolve_venue(context)
        home = self._resolve_team(
            context.match.home_team_id,
            context=context,
        )
        away = self._resolve_team(
            context.match.away_team_id,
            context=context,
        )

        home_distance = self._distance_feature(
            name="travel.home.reference_distance_km",
            team=home,
            venue=venue,
            context=context,
        )
        away_distance = self._distance_feature(
            name="travel.away.reference_distance_km",
            team=away,
            venue=venue,
            context=context,
        )

        return (
            neutral,
            home_distance,
            away_distance,
            _difference(
                definition=self._definition(
                    "travel.reference_distance_diff_home_minus_away_km"
                ),
                home=home_distance,
                away=away_distance,
                context=context,
            ),
        )

    def _resolve_venue(self, context: PredictionContext) -> _ResolvedLocation:
        venue_id = context.match.venue_id
        if venue_id is None:
            return _ResolvedLocation(
                location=None,
                missing_reason=FeatureMissingReason.UPSTREAM_UNAVAILABLE,
                lineage=FeatureLineage(),
            )

        return _resolve_location(
            self._venue_locations.get(venue_id, ()),
            cutoff=context.prediction_time,
        )

    def _resolve_team(
        self,
        team_id: str,
        *,
        context: PredictionContext,
    ) -> _ResolvedLocation:
        return _resolve_location(
            self._team_locations.get(team_id, ()),
            cutoff=context.prediction_time,
        )

    def _distance_feature(
        self,
        *,
        name: str,
        team: _ResolvedLocation,
        venue: _ResolvedLocation,
        context: PredictionContext,
    ) -> FeatureValue:
        lineage = _merge_lineage(team.lineage, venue.lineage)

        if team.location is None or venue.location is None:
            return FeatureValue(
                definition=self._definition(name),
                status=FeatureStatus.MISSING,
                as_of=context.prediction_time,
                missing_reason=_combined_missing_reason(
                    team.missing_reason,
                    venue.missing_reason,
                ),
                lineage=lineage,
            )

        return FeatureValue(
            definition=self._definition(name),
            status=FeatureStatus.OBSERVED,
            as_of=context.prediction_time,
            value=great_circle_distance_km(team.location, venue.location),
            lineage=lineage,
        )

    def _definition(self, name: str) -> FeatureDefinition:
        for definition in self._definitions:
            if definition.name == name:
                return definition
        raise KeyError(name)


def great_circle_distance_km(
    origin: GeographicPoint,
    destination: GeographicPoint,
) -> float:
    """Return haversine great-circle distance using the IUGG mean Earth radius."""

    lat1 = math.radians(origin.latitude)
    lat2 = math.radians(destination.latitude)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(destination.longitude - origin.longitude)

    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2.0 * math.asin(min(1.0, math.sqrt(haversine)))
    return EARTH_MEAN_RADIUS_KM * central_angle


def _resolve_location(
    observations: Sequence[
        VenueLocationObservation | TeamReferenceLocationObservation
    ],
    *,
    cutoff: datetime,
) -> _ResolvedLocation:
    eligible = [
        observation
        for observation in observations
        if observation.metadata.available_at is not None
        and observation.metadata.available_at <= cutoff
        and observation.metadata.leakage_risk is LeakageRisk.SAFE
    ]

    if not eligible:
        return _ResolvedLocation(
            location=None,
            missing_reason=(
                FeatureMissingReason.TEMPORAL_INTEGRITY
                if observations
                else FeatureMissingReason.UPSTREAM_UNAVAILABLE
            ),
            lineage=FeatureLineage(),
        )

    observation = eligible[-1]
    source_record_id = observation.metadata.source_record_id

    return _ResolvedLocation(
        location=observation.location,
        missing_reason=None,
        lineage=FeatureLineage(
            source_record_ids=(source_record_id,)
            if source_record_id is not None
            else ()
        ),
    )


def _index_venue_observations(
    observations: Sequence[VenueLocationObservation],
) -> dict[str, tuple[VenueLocationObservation, ...]]:
    indexed: dict[str, list[VenueLocationObservation]] = {}
    for observation in observations:
        indexed.setdefault(observation.venue_id, []).append(observation)

    return {
        venue_id: tuple(sorted(items, key=_observation_sort_key))
        for venue_id, items in indexed.items()
    }


def _index_team_observations(
    observations: Sequence[TeamReferenceLocationObservation],
) -> dict[str, tuple[TeamReferenceLocationObservation, ...]]:
    indexed: dict[str, list[TeamReferenceLocationObservation]] = {}
    for observation in observations:
        indexed.setdefault(observation.team_id, []).append(observation)

    return {
        team_id: tuple(sorted(items, key=_observation_sort_key))
        for team_id, items in indexed.items()
    }


def _observation_sort_key(
    observation: VenueLocationObservation | TeamReferenceLocationObservation,
) -> tuple[object, ...]:
    available_at = observation.metadata.available_at
    return (
        available_at is None,
        available_at.isoformat() if available_at is not None else "",
        observation.metadata.source_record_id or "",
    )


def _merge_lineage(
    first: FeatureLineage,
    second: FeatureLineage,
) -> FeatureLineage:
    return FeatureLineage(
        source_record_ids=tuple(
            sorted(set(first.source_record_ids + second.source_record_ids))
        ),
        artifact_ids=tuple(
            sorted(set(first.artifact_ids + second.artifact_ids))
        ),
        model_ids=tuple(
            sorted(set(first.model_ids + second.model_ids))
        ),
    )


def _combined_missing_reason(
    *reasons: FeatureMissingReason | None,
) -> FeatureMissingReason:
    """Return the most safety-relevant reason when several inputs are absent."""

    present = tuple(reason for reason in reasons if reason is not None)

    if FeatureMissingReason.TEMPORAL_INTEGRITY in present:
        return FeatureMissingReason.TEMPORAL_INTEGRITY
    if FeatureMissingReason.STALE in present:
        return FeatureMissingReason.STALE
    if FeatureMissingReason.TEMPORAL_PRECISION in present:
        return FeatureMissingReason.TEMPORAL_PRECISION
    if present:
        return present[0]

    return FeatureMissingReason.UPSTREAM_UNAVAILABLE


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
            missing_reason=_combined_missing_reason(
                home.missing_reason,
                away.missing_reason,
            ),
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
    specs = [
        (
            "context.match.neutral_site",
            "Whether the canonical match is explicitly marked as neutral.",
            "venue_context",
        ),
        (
            "travel.home.reference_distance_km",
            "Great-circle distance from the home team's governed reference point "
            "to the governed match venue.",
            "travel_context",
        ),
        (
            "travel.away.reference_distance_km",
            "Great-circle distance from the away team's governed reference point "
            "to the governed match venue.",
            "travel_context",
        ),
        (
            "travel.reference_distance_diff_home_minus_away_km",
            "Home reference distance minus away reference distance.",
            "travel_context",
        ),
    ]

    return tuple(
        FeatureDefinition(
            name=name,
            version=version,
            group=group,
            description=description,
        )
        for name, description, group in specs
    )
