from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from football_analytics.data import LeakageRisk, SourceMetadata
from football_analytics.domain import GeographicPoint, Match
from football_analytics.features import (
    FeatureMissingReason,
    FeatureStatus,
    PredictionContext,
    TeamReferenceLocationObservation,
    TravelContextFeatureProvider,
    VenueLocationObservation,
    build_feature_vector,
)
from football_analytics.features.travel import great_circle_distance_km


CUTOFF = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _metadata(
    record_id: str,
    *,
    available_at: datetime,
    leakage_risk: LeakageRisk = LeakageRisk.SAFE,
) -> SourceMetadata:
    return SourceMetadata(
        source="geospatial_registry",
        source_record_id=record_id,
        available_at=available_at,
        ingested_at=max(
            available_at,
            datetime(2026, 5, 1, tzinfo=UTC),
        ),
        leakage_risk=leakage_risk,
    )


def _match(*, venue_id: str | None = "venue-1", neutral: bool = True) -> Match:
    return Match(
        match_id="arg-fra",
        match_date=date(2026, 6, 3),
        kickoff_at=datetime(2026, 6, 3, 20, 0, tzinfo=UTC),
        home_team_id="ARG",
        away_team_id="FRA",
        competition_id="friendly",
        neutral=neutral,
        venue_id=venue_id,
    )


def _provider() -> TravelContextFeatureProvider:
    return TravelContextFeatureProvider(
        venue_locations=[
            VenueLocationObservation(
                venue_id="venue-1",
                location=GeographicPoint(latitude=0.0, longitude=0.0),
                metadata=_metadata(
                    "venue-location-1",
                    available_at=CUTOFF - timedelta(days=30),
                ),
            )
        ],
        team_reference_locations=[
            TeamReferenceLocationObservation(
                team_id="ARG",
                location=GeographicPoint(latitude=0.0, longitude=0.0),
                metadata=_metadata(
                    "arg-reference-1",
                    available_at=CUTOFF - timedelta(days=30),
                ),
            ),
            TeamReferenceLocationObservation(
                team_id="FRA",
                location=GeographicPoint(latitude=0.0, longitude=1.0),
                metadata=_metadata(
                    "fra-reference-1",
                    available_at=CUTOFF - timedelta(days=30),
                ),
            ),
        ],
    )


def test_geographic_point_validates_coordinate_ranges() -> None:
    with pytest.raises(ValueError, match="latitude"):
        GeographicPoint(latitude=91.0, longitude=0.0)

    with pytest.raises(ValueError, match="longitude"):
        GeographicPoint(latitude=0.0, longitude=181.0)


def test_great_circle_distance_matches_one_degree_at_equator() -> None:
    distance = great_circle_distance_km(
        GeographicPoint(latitude=0.0, longitude=0.0),
        GeographicPoint(latitude=0.0, longitude=1.0),
    )

    assert distance == pytest.approx(111.195, rel=1e-4)


def test_travel_provider_builds_reference_distance_and_neutral_features() -> None:
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [_provider()])
    values = {value.definition.name: value for value in vector.values}

    assert values["context.match.neutral_site"].value == 1.0
    assert values["travel.home.reference_distance_km"].value == pytest.approx(0.0)
    assert values["travel.away.reference_distance_km"].value == pytest.approx(
        111.195,
        rel=1e-4,
    )
    assert values[
        "travel.reference_distance_diff_home_minus_away_km"
    ].value == pytest.approx(-111.195, rel=1e-4)
    assert values["travel.home.reference_distance_km"].lineage.source_record_ids == (
        "arg-reference-1",
        "venue-location-1",
    )


def test_future_team_reference_is_not_available_at_cutoff() -> None:
    provider = TravelContextFeatureProvider(
        venue_locations=[
            VenueLocationObservation(
                venue_id="venue-1",
                location=GeographicPoint(latitude=0.0, longitude=0.0),
                metadata=_metadata(
                    "venue",
                    available_at=CUTOFF - timedelta(days=1),
                ),
            )
        ],
        team_reference_locations=[
            TeamReferenceLocationObservation(
                team_id="ARG",
                location=GeographicPoint(latitude=0.0, longitude=0.0),
                metadata=_metadata(
                    "future-arg",
                    available_at=CUTOFF + timedelta(hours=1),
                ),
            )
        ],
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["travel.home.reference_distance_km"].status is FeatureStatus.MISSING
    assert (
        values["travel.home.reference_distance_km"].missing_reason
        is FeatureMissingReason.TEMPORAL_INTEGRITY
    )


def test_unsafe_location_observation_is_not_used() -> None:
    provider = TravelContextFeatureProvider(
        venue_locations=[
            VenueLocationObservation(
                venue_id="venue-1",
                location=GeographicPoint(latitude=0.0, longitude=0.0),
                metadata=_metadata(
                    "unsafe-venue",
                    available_at=CUTOFF - timedelta(days=1),
                    leakage_risk=LeakageRisk.REVIEW,
                ),
            )
        ],
        team_reference_locations=[],
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["travel.home.reference_distance_km"].status is FeatureStatus.MISSING
    assert (
        values["travel.home.reference_distance_km"].missing_reason
        is FeatureMissingReason.TEMPORAL_INTEGRITY
    )


def test_missing_venue_id_does_not_infer_location() -> None:
    context = PredictionContext(
        match=_match(venue_id=None),
        prediction_time=CUTOFF,
    )

    vector = build_feature_vector(context, [_provider()])
    values = {value.definition.name: value for value in vector.values}

    assert values["context.match.neutral_site"].status is FeatureStatus.OBSERVED
    assert values["travel.home.reference_distance_km"].status is FeatureStatus.MISSING
    assert (
        values["travel.home.reference_distance_km"].missing_reason
        is FeatureMissingReason.UPSTREAM_UNAVAILABLE
    )
