"""Point-in-time match revision inspection with stable commit-order pagination."""

from datetime import UTC, datetime
from typing import Any

from football_analytics.data.contracts import ensure_utc
from football_analytics.storage.repository import Repository


def match_revision_history(
    repo: Repository, match_id: str, *, as_of: datetime | None = None,
    through_sequence: int | None = None, after_sequence: int = 0, limit: int = 100,
) -> dict[str, Any]:
    if not match_id.strip() or type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Declare a match identity and limit in [1, 1000].")
    if through_sequence is not None and as_of is None:
        raise ValueError("Snapshot sequence requires as_of.")
    if after_sequence and (as_of is None or through_sequence is None):
        raise ValueError("History continuation requires as_of and through_sequence.")
    cutoff = ensure_utc(as_of or datetime.now(UTC))
    committed = repo.snapshot_sequence()
    boundary = committed if through_sequence is None else through_sequence
    if type(boundary) is not int or not 0 <= boundary <= committed:
        raise ValueError("Snapshot sequence is not committed.")
    rows = repo.history(
        "match", match_id, as_of=cutoff, through_sequence=boundary,
        after_sequence=after_sequence, limit=limit + 1,
    )
    return {
        "match_id": match_id, "items": rows[:limit], "limit": limit,
        "as_of": cutoff.isoformat(), "through_sequence": boundary,
        "next_cursor": rows[limit - 1]["sequence"] if len(rows) > limit else None,
        "truncated": len(rows) > limit, "order": "commit_sequence_ascending",
    }
