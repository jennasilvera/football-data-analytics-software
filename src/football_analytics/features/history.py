from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from football_analytics.data.contracts import ensure_utc
from football_analytics.data.normalization import CanonicalMatchRecord
from football_analytics.domain import MatchStatus


def completed_record_is_before_cutoff(
    record: CanonicalMatchRecord,
    cutoff: datetime,
) -> bool:
    """Return whether a completed match result is safely before a cutoff.

    Exact kickoff timestamps are compared directly. Date-only historical rows are
    eligible only on a later UTC calendar date so same-day chronology is never
    invented.
    """

    cutoff_utc = ensure_utc(cutoff, "cutoff")

    if record.match.status is not MatchStatus.COMPLETED:
        return False

    if record.home_score is None or record.away_score is None:
        return False

    if record.match.kickoff_at is not None:
        return record.match.kickoff_at < cutoff_utc

    return record.match.match_date < cutoff_utc.date()


def team_history_before_cutoff(
    records: Sequence[CanonicalMatchRecord],
    *,
    team_id: str,
    cutoff: datetime,
) -> tuple[CanonicalMatchRecord, ...]:
    """Return canonical completed matches for one team before a cutoff."""

    eligible = [
        record
        for record in records
        if team_id in (record.match.home_team_id, record.match.away_team_id)
        and completed_record_is_before_cutoff(record, cutoff)
    ]

    eligible.sort(key=_history_sort_key)
    return tuple(eligible)


def _history_sort_key(record: CanonicalMatchRecord) -> tuple[object, ...]:
    kickoff = record.match.kickoff_at

    return (
        record.match.match_date,
        kickoff is None,
        kickoff.isoformat() if kickoff is not None else "",
        record.match.match_id,
    )
