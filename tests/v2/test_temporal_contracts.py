from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from football_analytics.data import (
    LeakageRisk,
    PointInTimeRecord,
    SourceMetadata,
    TemporalIntegrityError,
    assert_pre_match_available,
)

KICKOFF = datetime(2026, 11, 14, 20, 0, tzinfo=UTC)


def _metadata(
    *,
    available_at: datetime,
    leakage_risk: LeakageRisk = LeakageRisk.SAFE,
) -> SourceMetadata:
    return SourceMetadata(
        source="example_source",
        source_record_id="row-1",
        source_version="2026-11-14",
        event_time=KICKOFF,
        available_at=available_at,
        ingested_at=available_at + timedelta(minutes=5),
        leakage_risk=leakage_risk,
    )


def test_pre_match_record_is_available_before_cutoff() -> None:
    metadata = _metadata(available_at=KICKOFF - timedelta(hours=3))
    record = PointInTimeRecord(value={"rank": 1}, metadata=metadata)

    assert record.is_available_at(KICKOFF - timedelta(hours=1))
    assert record.require_pre_match(
        kickoff_at=KICKOFF,
        prediction_time=KICKOFF - timedelta(hours=1),
    ) == {"rank": 1}


def test_future_observation_is_rejected() -> None:
    metadata = _metadata(available_at=KICKOFF - timedelta(minutes=30))

    with pytest.raises(TemporalIntegrityError, match="not available"):
        assert_pre_match_available(
            metadata,
            kickoff_at=KICKOFF,
            prediction_time=KICKOFF - timedelta(hours=1),
        )


def test_post_match_only_observation_is_rejected() -> None:
    metadata = _metadata(
        available_at=KICKOFF - timedelta(hours=2),
        leakage_risk=LeakageRisk.POST_MATCH_ONLY,
    )

    with pytest.raises(TemporalIntegrityError, match="Post-match-only"):
        assert_pre_match_available(
            metadata,
            kickoff_at=KICKOFF,
            prediction_time=KICKOFF - timedelta(hours=1),
        )


def test_unknown_availability_is_retained_but_fails_pre_match_check() -> None:
    metadata = SourceMetadata(
        source="historical_archive",
        ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
        available_at=None,
        leakage_risk=LeakageRisk.REVIEW,
    )
    record = PointInTimeRecord(value={"archived_rank": 3}, metadata=metadata)

    assert not record.is_available_at(KICKOFF)

    with pytest.raises(TemporalIntegrityError, match="availability is unknown"):
        record.require_pre_match(kickoff_at=KICKOFF)


def test_prediction_after_kickoff_is_rejected() -> None:
    metadata = _metadata(available_at=KICKOFF - timedelta(hours=2))

    with pytest.raises(TemporalIntegrityError, match="after kickoff"):
        assert_pre_match_available(
            metadata,
            kickoff_at=KICKOFF,
            prediction_time=KICKOFF + timedelta(seconds=1),
        )


def test_naive_temporal_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        SourceMetadata(
            source="example_source",
            available_at=datetime(2026, 11, 14, 18, 0),
            ingested_at=datetime(2026, 11, 14, 18, 5),
        )
