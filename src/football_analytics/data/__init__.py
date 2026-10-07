"""Canonical data contracts and temporal integrity helpers."""

from football_analytics.data.contracts import (
    LeakageRisk,
    PointInTimeRecord,
    SourceMetadata,
    TemporalIntegrityError,
    assert_pre_match_available,
)

__all__ = [
    "LeakageRisk",
    "PointInTimeRecord",
    "SourceMetadata",
    "TemporalIntegrityError",
    "assert_pre_match_available",
]
