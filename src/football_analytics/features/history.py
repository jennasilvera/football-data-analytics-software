from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum

from football_analytics.data.contracts import ensure_utc
from football_analytics.data.normalization import CanonicalMatchRecord
from football_analytics.domain import MatchStatus
from football_analytics.domain.scores import require_regulation_score


class ResultAvailabilityError(ValueError):
    """Raised when completed-result availability metadata is temporally invalid."""


class ResultEligibilityBasis(StrEnum):
    """Evidence used to decide when a completed result may enter model history."""

    SOURCE_AVAILABLE_AT = "source_available_at"
    CONSERVATIVE_NEXT_UTC_DAY = "conservative_next_utc_day"


@dataclass(frozen=True, slots=True)
class ResultEligibility:
    """Training/history eligibility for one completed match result."""

    eligible_at: datetime
    basis: ResultEligibilityBasis

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "eligible_at",
            ensure_utc(self.eligible_at, "eligible_at"),
        )


@dataclass(frozen=True, slots=True)
class CompletedResultEligibilityPolicy:
    """Versioned policy for when final scores may enter historical state.

    Exact source availability is used only when it is defensible relative to a
    known kickoff. When precise result availability cannot be demonstrated, the
    result is conservatively delayed until the next UTC calendar day. This is an
    eligibility boundary, not a claim about the source's actual publication time.
    """

    policy_id: str

    def __post_init__(self) -> None:
        policy_id = self.policy_id.strip()
        if not policy_id:
            raise ValueError("policy_id must not be blank.")
        object.__setattr__(self, "policy_id", policy_id)

    def eligibility_for(self, record: CanonicalMatchRecord) -> ResultEligibility:
        """Return the earliest defensible time a final result may be used."""

        if record.match.status is not MatchStatus.COMPLETED:
            raise ResultAvailabilityError(
                "Result eligibility requires a completed match."
            )
        if record.home_score is None or record.away_score is None:
            raise ResultAvailabilityError(
                "Result eligibility requires a completed final score."
            )

        explicit = record.metadata.available_at
        kickoff = record.match.kickoff_at

        if kickoff is not None:
            fallback = _next_utc_day(kickoff)

            if explicit is None:
                return ResultEligibility(
                    eligible_at=fallback,
                    basis=ResultEligibilityBasis.CONSERVATIVE_NEXT_UTC_DAY,
                )

            explicit_utc = ensure_utc(explicit, "available_at")
            if explicit_utc <= kickoff:
                raise ResultAvailabilityError(
                    "Completed-result available_at must be after exact kickoff_at."
                )

            return ResultEligibility(
                eligible_at=explicit_utc,
                basis=ResultEligibilityBasis.SOURCE_AVAILABLE_AT,
            )

        fallback = datetime.combine(
            record.match.match_date + timedelta(days=1),
            time.min,
            tzinfo=UTC,
        )

        if explicit is None:
            return ResultEligibility(
                eligible_at=fallback,
                basis=ResultEligibilityBasis.CONSERVATIVE_NEXT_UTC_DAY,
            )

        explicit_utc = ensure_utc(explicit, "available_at")
        if explicit_utc.date() < record.match.match_date:
            raise ResultAvailabilityError(
                "Completed-result available_at cannot precede match_date."
            )

        if explicit_utc > fallback:
            return ResultEligibility(
                eligible_at=explicit_utc,
                basis=ResultEligibilityBasis.SOURCE_AVAILABLE_AT,
            )

        return ResultEligibility(
            eligible_at=fallback,
            basis=ResultEligibilityBasis.CONSERVATIVE_NEXT_UTC_DAY,
        )


DEFAULT_RESULT_ELIGIBILITY_POLICY = CompletedResultEligibilityPolicy(
    policy_id="completed_result_eligibility_v1",
)


def completed_record_is_before_cutoff(
    record: CanonicalMatchRecord,
    cutoff: datetime,
    *,
    policy: CompletedResultEligibilityPolicy = DEFAULT_RESULT_ELIGIBILITY_POLICY,
) -> bool:
    """Return whether a completed final result is eligible at a cutoff."""

    cutoff_utc = ensure_utc(cutoff, "cutoff")

    if record.match.status is not MatchStatus.COMPLETED:
        return False
    if record.home_score is None or record.away_score is None:
        return False

    eligible = policy.eligibility_for(record).eligible_at <= cutoff_utc
    if eligible:
        require_regulation_score(record.score_basis, match_id=record.match.match_id)
    return eligible


def team_history_before_cutoff(
    records: Sequence[CanonicalMatchRecord],
    *,
    team_id: str,
    cutoff: datetime,
    policy: CompletedResultEligibilityPolicy = DEFAULT_RESULT_ELIGIBILITY_POLICY,
) -> tuple[CanonicalMatchRecord, ...]:
    """Return completed team results that are eligible at a cutoff."""

    eligible = [
        record
        for record in records
        if team_id in (record.match.home_team_id, record.match.away_team_id)
        and completed_record_is_before_cutoff(
            record,
            cutoff,
            policy=policy,
        )
    ]

    eligible.sort(key=_history_sort_key)
    return tuple(eligible)


def _next_utc_day(kickoff: datetime) -> datetime:
    kickoff_utc = ensure_utc(kickoff, "kickoff_at")
    return datetime.combine(
        kickoff_utc.date() + timedelta(days=1),
        time.min,
        tzinfo=UTC,
    )


def _history_sort_key(record: CanonicalMatchRecord) -> tuple[object, ...]:
    kickoff = record.match.kickoff_at

    return (
        record.match.match_date,
        kickoff is None,
        kickoff.isoformat() if kickoff is not None else "",
        record.match.match_id,
    )
