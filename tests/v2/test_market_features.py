from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from football_analytics.data import LeakageRisk, SourceMetadata
from football_analytics.domain import Match
from football_analytics.features import (
    FeatureMissingReason,
    FeatureStatus,
    MarketSnapshotFeatureProvider,
    MarketSnapshotObservation,
    PredictionContext,
    build_feature_vector,
    devig_decimal_odds,
)

CUTOFF = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _metadata(
    record_id: str,
    *,
    available_at: datetime,
    leakage_risk: LeakageRisk = LeakageRisk.SAFE,
) -> SourceMetadata:
    return SourceMetadata(
        source="market_feed",
        source_record_id=record_id,
        available_at=available_at,
        ingested_at=max(available_at, CUTOFF - timedelta(days=30)),
        leakage_risk=leakage_risk,
    )


def _snapshot(
    *,
    source: str,
    home: float,
    draw: float,
    away: float,
    available_at: datetime,
    record_id: str,
) -> MarketSnapshotObservation:
    return MarketSnapshotObservation(
        match_id="arg-fra",
        market_source_id=source,
        home_decimal_odds=home,
        draw_decimal_odds=draw,
        away_decimal_odds=away,
        metadata=_metadata(record_id, available_at=available_at),
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


def test_market_snapshot_rejects_invalid_decimal_odds() -> None:
    with pytest.raises(ValueError, match="greater than 1.0"):
        _snapshot(
            source="book-a",
            home=1.0,
            draw=3.5,
            away=4.0,
            available_at=CUTOFF - timedelta(hours=1),
            record_id="invalid",
        )


def test_devigged_probabilities_remove_overround() -> None:
    snapshot = _snapshot(
        source="book-a",
        home=1.80,
        draw=3.60,
        away=4.00,
        available_at=CUTOFF - timedelta(hours=1),
        record_id="snapshot",
    )

    market = devig_decimal_odds(snapshot)

    assert sum(market.probabilities.as_tuple()) == pytest.approx(1.0)
    assert market.overround > 0.0


def test_latest_cutoff_safe_snapshot_drives_consensus_and_movement() -> None:
    provider = MarketSnapshotFeatureProvider(
        [
            _snapshot(
                source="book-a",
                home=2.0,
                draw=4.0,
                away=4.0,
                available_at=CUTOFF - timedelta(hours=12),
                record_id="open-a",
            ),
            _snapshot(
                source="book-a",
                home=1.8,
                draw=4.0,
                away=4.0,
                available_at=CUTOFF - timedelta(hours=1),
                record_id="latest-a",
            ),
            _snapshot(
                source="book-a",
                home=1.5,
                draw=4.5,
                away=5.0,
                available_at=CUTOFF + timedelta(minutes=30),
                record_id="future-a",
            ),
        ]
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["market.consensus.source_count"].value == 1.0
    assert values["market.consensus.home_fair_probability"].value > 0.5
    assert (
        values[
            "market.consensus.home_fair_probability_move_from_open"
        ].value
        > 0.0
    )
    assert values[
        "market.consensus.home_fair_probability"
    ].lineage.source_record_ids == ("latest-a",)
    assert set(
        values[
            "market.consensus.home_fair_probability_move_from_open"
        ].lineage.source_record_ids
    ) == {"open-a", "latest-a"}


def test_market_consensus_averages_latest_snapshot_per_source() -> None:
    provider = MarketSnapshotFeatureProvider(
        [
            _snapshot(
                source="book-a",
                home=2.0,
                draw=4.0,
                away=4.0,
                available_at=CUTOFF - timedelta(hours=2),
                record_id="a",
            ),
            _snapshot(
                source="book-b",
                home=4.0,
                draw=4.0,
                away=2.0,
                available_at=CUTOFF - timedelta(hours=3),
                record_id="b",
            ),
        ]
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["market.consensus.source_count"].value == 2.0
    assert values["market.consensus.home_fair_probability"].value == pytest.approx(
        0.375
    )
    assert values["market.consensus.away_fair_probability"].value == pytest.approx(
        0.375
    )
    assert values[
        "market.consensus.oldest_source_snapshot_age_hours"
    ].value == pytest.approx(3.0)


def test_future_only_market_data_is_temporally_missing() -> None:
    provider = MarketSnapshotFeatureProvider(
        [
            _snapshot(
                source="book-a",
                home=2.0,
                draw=4.0,
                away=4.0,
                available_at=CUTOFF + timedelta(minutes=1),
                record_id="future",
            )
        ]
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])

    assert all(value.status is FeatureStatus.MISSING for value in vector.values)
    assert all(
        value.missing_reason is FeatureMissingReason.TEMPORAL_INTEGRITY
        for value in vector.values
    )
