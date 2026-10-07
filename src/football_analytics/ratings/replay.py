from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from football_analytics.data.normalization import CanonicalMatchRecord
from football_analytics.domain import MatchStatus
from football_analytics.ratings.base import RatingEngine, RatingSnapshot, RatingUpdate
from football_analytics.ratings.legacy_elo import rating_input_from_record


class AmbiguousRatingOrderError(ValueError):
    """Raised when date-only data cannot establish a defensible replay order."""


@dataclass(frozen=True, slots=True)
class RatingReplayResult:
    """Immutable outputs from replaying canonical completed matches."""

    updates: tuple[RatingUpdate, ...]
    snapshots: tuple[RatingSnapshot, ...]


def replay_completed_matches(
    records: Sequence[CanonicalMatchRecord],
    *,
    engine: RatingEngine,
    competition_names: Mapping[str, str],
) -> RatingReplayResult:
    """Replay canonical completed matches through a rating engine.

    Exact kickoff timestamps are used when available. Date-only records remain
    valid provided a team does not appear more than once on the same date. If a
    same-day team sequence is temporally ambiguous, replay stops rather than
    inventing an ordering that could alter ratings.
    """

    _validate_replay_records(records)

    ordered = sorted(records, key=_replay_sort_key)
    updates: list[RatingUpdate] = []

    for record in ordered:
        competition_id = record.match.competition_id

        try:
            competition_name = competition_names[competition_id]
        except KeyError as exc:
            raise KeyError(
                f"Missing competition name for competition_id: {competition_id}"
            ) from exc

        updates.append(
            engine.update(
                rating_input_from_record(
                    record,
                    competition_name=competition_name,
                )
            )
        )

    return RatingReplayResult(
        updates=tuple(updates),
        snapshots=engine.snapshots(),
    )


def _validate_replay_records(records: Sequence[CanonicalMatchRecord]) -> None:
    appearances: dict[tuple[object, str], list[CanonicalMatchRecord]] = {}

    for record in records:
        if record.match.status is not MatchStatus.COMPLETED:
            raise ValueError(
                f"Rating replay requires completed matches: {record.match.match_id}"
            )

        for team_id in (record.match.home_team_id, record.match.away_team_id):
            key = (record.match.match_date, team_id)
            appearances.setdefault(key, []).append(record)

    for (match_date, team_id), team_records in appearances.items():
        if len(team_records) <= 1:
            continue

        if any(record.match.kickoff_at is None for record in team_records):
            raise AmbiguousRatingOrderError(
                "Cannot order multiple same-day matches for "
                f"{team_id} on {match_date}: at least one kickoff time is unknown."
            )


def _replay_sort_key(record: CanonicalMatchRecord) -> tuple[object, ...]:
    kickoff = record.match.kickoff_at

    return (
        record.match.match_date,
        kickoff is None,
        kickoff.isoformat() if kickoff is not None else "",
        record.match.home_team_id,
        record.match.away_team_id,
        record.match.match_id,
    )
