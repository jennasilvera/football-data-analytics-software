from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from football_analytics.data import CanonicalMatchRecord, LeakageRisk, SourceMetadata
from football_analytics.domain import Match, MatchStatus
from football_analytics.domain.scores import ScoreBasis
from football_analytics.features import (
    DEFAULT_RESULT_ELIGIBILITY_POLICY,
    ResultAvailabilityError,
    ResultEligibilityBasis,
    completed_record_is_before_cutoff,
)


def _record(
    *,
    match_id: str,
    match_date: date,
    kickoff_at: datetime | None = None,
    available_at: datetime | None = None,
) -> CanonicalMatchRecord:
    return CanonicalMatchRecord(
        score_basis=ScoreBasis.REGULATION_TIME,
        match=Match(
            match_id=match_id,
            match_date=match_date,
            kickoff_at=kickoff_at,
            home_team_id="ARG",
            away_team_id="BRA",
            competition_id="friendly",
            neutral=True,
            status=MatchStatus.COMPLETED,
        ),
        metadata=SourceMetadata(
            source="history",
            ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
            available_at=available_at,
            leakage_risk=LeakageRisk.POST_MATCH_ONLY,
        ),
        source_match_id=f"source-{match_id}",
        source_home_team_name="ARG",
        source_away_team_name="BRA",
        source_competition_name="International Friendly",
        home_score=2,
        away_score=1,
    )


def test_exact_kickoff_without_result_timestamp_uses_next_utc_day() -> None:
    kickoff = datetime(2026, 6, 10, 15, 0, tzinfo=UTC)
    record = _record(
        match_id="exact-no-result-time",
        match_date=date(2026, 6, 10),
        kickoff_at=kickoff,
    )

    eligibility = DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(record)

    assert eligibility.eligible_at == datetime(2026, 6, 11, tzinfo=UTC)
    assert (
        eligibility.basis
        is ResultEligibilityBasis.CONSERVATIVE_NEXT_UTC_DAY
    )
    assert not completed_record_is_before_cutoff(
        record,
        datetime(2026, 6, 10, 23, 59, tzinfo=UTC),
    )
    assert completed_record_is_before_cutoff(
        record,
        datetime(2026, 6, 11, tzinfo=UTC),
    )


def test_exact_kickoff_uses_defensible_source_result_timestamp() -> None:
    kickoff = datetime(2026, 6, 10, 15, 0, tzinfo=UTC)
    available_at = datetime(2026, 6, 10, 17, 5, tzinfo=UTC)
    record = _record(
        match_id="exact-source-time",
        match_date=date(2026, 6, 10),
        kickoff_at=kickoff,
        available_at=available_at,
    )

    eligibility = DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(record)

    assert eligibility.eligible_at == available_at
    assert eligibility.basis is ResultEligibilityBasis.SOURCE_AVAILABLE_AT
    assert not completed_record_is_before_cutoff(
        record,
        datetime(2026, 6, 10, 17, 0, tzinfo=UTC),
    )
    assert completed_record_is_before_cutoff(record, available_at)


def test_date_only_result_never_uses_same_day_chronology() -> None:
    record = _record(
        match_id="date-only",
        match_date=date(2026, 6, 10),
        available_at=datetime(2026, 6, 10, 22, 0, tzinfo=UTC),
    )

    eligibility = DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(record)

    assert eligibility.eligible_at == datetime(2026, 6, 11, tzinfo=UTC)
    assert (
        eligibility.basis
        is ResultEligibilityBasis.CONSERVATIVE_NEXT_UTC_DAY
    )


def test_later_explicit_availability_overrides_date_only_fallback() -> None:
    available_at = datetime(2026, 6, 12, 8, 0, tzinfo=UTC)
    record = _record(
        match_id="date-only-late-source",
        match_date=date(2026, 6, 10),
        available_at=available_at,
    )

    eligibility = DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(record)

    assert eligibility.eligible_at == available_at
    assert eligibility.basis is ResultEligibilityBasis.SOURCE_AVAILABLE_AT


def test_exact_kickoff_rejects_result_timestamp_at_or_before_kickoff() -> None:
    kickoff = datetime(2026, 6, 10, 15, 0, tzinfo=UTC)
    record = _record(
        match_id="invalid-source-time",
        match_date=date(2026, 6, 10),
        kickoff_at=kickoff,
        available_at=kickoff,
    )

    with pytest.raises(ResultAvailabilityError, match="after exact kickoff_at"):
        DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(record)
