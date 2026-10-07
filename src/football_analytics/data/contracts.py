from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Generic, TypeVar

T = TypeVar("T")


class LeakageRisk(StrEnum):
    """How an observation may be used by pre-match forecasting code."""

    SAFE = "safe"
    REVIEW = "review"
    HIGH = "high"
    POST_MATCH_ONLY = "post_match_only"


class TemporalIntegrityError(ValueError):
    """Raised when information violates point-in-time forecasting rules."""


def ensure_utc(value: datetime, field_name: str = "timestamp") -> datetime:
    """Require a timezone-aware timestamp and normalize it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")

    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Provenance and temporal metadata for one source observation."""

    source: str
    available_at: datetime
    ingested_at: datetime
    source_record_id: str | None = None
    source_version: str | None = None
    event_time: datetime | None = None
    leakage_risk: LeakageRisk = LeakageRisk.SAFE
    legal_use_notes: str | None = None

    def __post_init__(self) -> None:
        source = self.source.strip()

        if not source:
            raise ValueError("source must not be blank.")

        available_at = ensure_utc(self.available_at, "available_at")
        ingested_at = ensure_utc(self.ingested_at, "ingested_at")

        if ingested_at < available_at:
            raise ValueError("ingested_at cannot be earlier than available_at.")

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "ingested_at", ingested_at)

        if self.event_time is not None:
            object.__setattr__(
                self,
                "event_time",
                ensure_utc(self.event_time, "event_time"),
            )

        if self.source_record_id is not None:
            value = self.source_record_id.strip()
            object.__setattr__(self, "source_record_id", value or None)

        if self.source_version is not None:
            value = self.source_version.strip()
            object.__setattr__(self, "source_version", value or None)


def assert_pre_match_available(
    metadata: SourceMetadata,
    *,
    kickoff_at: datetime,
    prediction_time: datetime | None = None,
) -> None:
    """Enforce that an observation was usable for a pre-match prediction."""

    kickoff = ensure_utc(kickoff_at, "kickoff_at")
    cutoff = ensure_utc(
        prediction_time if prediction_time is not None else kickoff,
        "prediction_time",
    )

    if cutoff > kickoff:
        raise TemporalIntegrityError("prediction_time cannot be after kickoff_at.")

    if metadata.leakage_risk is LeakageRisk.POST_MATCH_ONLY:
        raise TemporalIntegrityError(
            "Post-match-only observations cannot be used in a pre-match forecast."
        )

    if metadata.available_at > cutoff:
        raise TemporalIntegrityError(
            "Observation was not available at the prediction cutoff."
        )


@dataclass(frozen=True, slots=True)
class PointInTimeRecord(Generic[T]):
    """A value coupled to the metadata needed for point-in-time evaluation."""

    value: T
    metadata: SourceMetadata

    def is_available_at(self, cutoff: datetime) -> bool:
        """Return whether the observation was available by a cutoff."""

        cutoff_utc = ensure_utc(cutoff, "cutoff")
        return self.metadata.available_at <= cutoff_utc

    def require_pre_match(
        self,
        *,
        kickoff_at: datetime,
        prediction_time: datetime | None = None,
    ) -> T:
        """Validate pre-match availability and return the wrapped value."""

        assert_pre_match_available(
            self.metadata,
            kickoff_at=kickoff_at,
            prediction_time=prediction_time,
        )
        return self.value
