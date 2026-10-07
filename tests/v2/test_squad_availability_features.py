from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from football_analytics.data import LeakageRisk, SourceMetadata
from football_analytics.domain import Match
from football_analytics.features import (
    FeatureMissingReason,
    FeatureStatus,
    PlayerAvailabilityObservation,
    PlayerAvailabilityStatus,
    PredictionContext,
    SquadAvailabilityFeatureProvider,
    build_feature_vector,
)

CUTOFF = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _metadata(
    record_id: str,
    *,
    available_at: datetime,
    leakage_risk: LeakageRisk = LeakageRisk.SAFE,
) -> SourceMetadata:
    return SourceMetadata(
        source="availability_feed",
        source_record_id=record_id,
        available_at=available_at,
        ingested_at=max(available_at, CUTOFF - timedelta(days=30)),
        leakage_risk=leakage_risk,
    )


def _match() -> Match:
    return Match(
        match_id="arg-fra",
        match_date=date(2026, 6, 3),
        kickoff_at=datetime(2026, 6, 3, 20, 0, tzinfo=UTC),
        home_team_id="ARG",
        away_team_id="FRA",
        competition_id="friendly",
        neutral=True,
    )


def _observation(
    *,
    team_id: str,
    player_id: str,
    probability: float,
    available_at: datetime,
    minutes: float | None = None,
    status: PlayerAvailabilityStatus = PlayerAvailabilityStatus.UNKNOWN,
    record_id: str | None = None,
) -> PlayerAvailabilityObservation:
    return PlayerAvailabilityObservation(
        match_id="arg-fra",
        team_id=team_id,
        player_id=player_id,
        status=status,
        availability_probability=probability,
        baseline_expected_minutes=minutes,
        metadata=_metadata(
            record_id or f"{team_id}-{player_id}-{probability}",
            available_at=available_at,
        ),
    )


def test_availability_probability_is_validated() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _observation(
            team_id="ARG",
            player_id="p1",
            probability=1.1,
            available_at=CUTOFF - timedelta(hours=1),
        )


def test_latest_cutoff_eligible_player_observation_is_used() -> None:
    provider = SquadAvailabilityFeatureProvider(
        [
            _observation(
                team_id="ARG",
                player_id="p1",
                probability=0.4,
                minutes=90.0,
                available_at=CUTOFF - timedelta(days=1),
                record_id="arg-p1-old",
            ),
            _observation(
                team_id="ARG",
                player_id="p1",
                probability=0.9,
                minutes=90.0,
                available_at=CUTOFF - timedelta(hours=1),
                record_id="arg-p1-latest",
            ),
            _observation(
                team_id="ARG",
                player_id="p1",
                probability=0.1,
                minutes=90.0,
                available_at=CUTOFF + timedelta(hours=1),
                record_id="arg-p1-future",
            ),
        ]
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["squad.home.player_observation_count"].value == 1.0
    assert values["squad.home.mean_availability_probability"].value == pytest.approx(
        0.9
    )
    assert (
        values["squad.home.mean_availability_probability"].lineage.source_record_ids
        == ("arg-p1-latest",)
    )


def test_status_label_does_not_secretly_override_explicit_probability() -> None:
    provider = SquadAvailabilityFeatureProvider(
        [
            _observation(
                team_id="ARG",
                player_id="p1",
                probability=0.8,
                minutes=90.0,
                status=PlayerAvailabilityStatus.OUT,
                available_at=CUTOFF - timedelta(hours=1),
            )
        ]
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["squad.home.mean_availability_probability"].value == pytest.approx(
        0.8
    )


def test_minutes_weighted_availability_exposes_projection_coverage() -> None:
    provider = SquadAvailabilityFeatureProvider(
        [
            _observation(
                team_id="ARG",
                player_id="p1",
                probability=1.0,
                minutes=90.0,
                available_at=CUTOFF - timedelta(hours=2),
            ),
            _observation(
                team_id="ARG",
                player_id="p2",
                probability=0.0,
                minutes=30.0,
                available_at=CUTOFF - timedelta(hours=2),
            ),
            _observation(
                team_id="ARG",
                player_id="p3",
                probability=0.5,
                minutes=None,
                available_at=CUTOFF - timedelta(hours=2),
            ),
        ]
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["squad.home.player_observation_count"].value == 3.0
    assert values["squad.home.minutes_coverage_ratio"].value == pytest.approx(2 / 3)
    assert values["squad.home.minutes_weighted_availability"].value == pytest.approx(
        0.75
    )
    assert values["squad.home.expected_unavailable_player_count"].value == pytest.approx(
        1.5
    )


def test_future_only_availability_is_temporally_missing() -> None:
    provider = SquadAvailabilityFeatureProvider(
        [
            _observation(
                team_id="ARG",
                player_id="p1",
                probability=0.0,
                available_at=CUTOFF + timedelta(hours=1),
            )
        ]
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["squad.home.player_observation_count"].value == 0.0
    assert values["squad.home.mean_availability_probability"].status is FeatureStatus.MISSING
    assert (
        values["squad.home.mean_availability_probability"].missing_reason
        is FeatureMissingReason.TEMPORAL_INTEGRITY
    )


def test_no_minutes_projection_keeps_weighted_metric_missing() -> None:
    provider = SquadAvailabilityFeatureProvider(
        [
            _observation(
                team_id="ARG",
                player_id="p1",
                probability=0.7,
                minutes=None,
                available_at=CUTOFF - timedelta(hours=1),
            )
        ]
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["squad.home.minutes_coverage_ratio"].value == 0.0
    assert values["squad.home.minutes_weighted_availability"].status is FeatureStatus.MISSING
    assert (
        values["squad.home.minutes_weighted_availability"].missing_reason
        is FeatureMissingReason.UPSTREAM_UNAVAILABLE
    )
