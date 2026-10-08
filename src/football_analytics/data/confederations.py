"""Published complete membership timelines; never infer history from current catalogs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from football_analytics.data.contracts import LeakageRisk, SourceMetadata, ensure_utc
from football_analytics.data.serialization import json_safe, metadata_from_dict
from football_analytics.domain.teams import Confederation
from football_analytics.models.postprocessing import content_id


@dataclass(frozen=True, slots=True)
class MembershipPeriod:
    confederation: Confederation
    valid_from: date
    valid_to: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.confederation, Confederation):
            raise ValueError("Membership requires a canonical confederation.")
        if type(self.valid_from) is not date or (
            self.valid_to is not None and type(self.valid_to) is not date
        ):
            raise ValueError("Membership validity requires dates, not timestamps.")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("Membership interval must be nonempty, with exclusive valid_to.")


@dataclass(frozen=True, slots=True)
class MembershipRelease:
    team_id: str
    periods: tuple[MembershipPeriod, ...]
    metadata: SourceMetadata

    def __post_init__(self) -> None:
        if not self.team_id.strip() or self.team_id != self.team_id.strip():
            raise ValueError("Membership requires an exact nonblank canonical team ID.")
        if (
            self.metadata.available_at is None
            or self.metadata.leakage_risk is not LeakageRisk.SAFE
            or not self.metadata.source_record_id
            or not (self.metadata.legal_use_notes or "").strip()
        ):
            raise ValueError(
                "Membership release requires safe publication provenance and legal notes."
            )
        ordered = tuple(sorted(self.periods, key=lambda p: p.valid_from))
        for left, right in zip(ordered, ordered[1:], strict=False):
            if left.valid_to is None or left.valid_to > right.valid_from:
                raise ValueError("Membership periods overlap within one published release.")
        object.__setattr__(self, "periods", ordered)

    @property
    def release_id(self) -> str:
        return content_id("membership_release_", json_safe(self))


class MembershipHistory:
    """Each release replaces a team's complete timeline, including intentional gaps."""

    def __init__(self, releases: tuple[MembershipRelease, ...]):
        self.releases = tuple(sorted(releases, key=lambda r: (r.team_id, r.metadata.available_at)))
        seen = set()
        for release in self.releases:
            key = (release.team_id, release.metadata.available_at)
            if key in seen:
                raise ValueError("Ambiguous duplicate team membership publication time.")
            seen.add(key)
        self.dataset_id = content_id("membership_history_", json_safe(self.releases))

    def resolve(
        self, team_id: str, match_date: date, prediction_time: datetime
    ) -> tuple[Confederation | None, str | None]:
        cutoff = ensure_utc(prediction_time)
        eligible = [
            r
            for r in self.releases
            if r.team_id == team_id
            and r.metadata.available_at is not None
            and r.metadata.available_at <= cutoff
        ]
        if not eligible:
            return None, None
        selected = eligible[-1]
        for period in selected.periods:
            if period.valid_from <= match_date and (
                period.valid_to is None or match_date < period.valid_to
            ):
                return period.confederation, selected.release_id
        return None, selected.release_id


def load_membership_history(path: Path, *, team_ids: set[str]) -> MembershipHistory:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "releases"}:
        raise ValueError("Membership input requires schema_version and releases.")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported membership schema.")
    if not isinstance(payload["releases"], list):
        raise ValueError("Membership releases must be an array.")
    releases = []
    for row in payload["releases"]:
        if not isinstance(row, dict):
            raise ValueError("Membership release must be an object.")
        if set(row) != {"team_id", "periods", "metadata"} or row["team_id"] not in team_ids:
            raise ValueError("Membership release has invalid fields or unknown canonical team.")
        if not isinstance(row["periods"], list):
            raise ValueError("Membership periods must be an array.")
        periods = []
        for period in row["periods"]:
            if not isinstance(period, dict):
                raise ValueError("Membership period must be an object.")
            if set(period) != {"confederation", "valid_from", "valid_to"}:
                raise ValueError("Membership period fields must be explicit.")
            periods.append(
                MembershipPeriod(
                    Confederation(period["confederation"]),
                    date.fromisoformat(period["valid_from"]),
                    date.fromisoformat(period["valid_to"])
                    if period["valid_to"] is not None
                    else None,
                )
            )
        releases.append(
            MembershipRelease(row["team_id"], tuple(periods), metadata_from_dict(row["metadata"]))
        )
    return MembershipHistory(tuple(releases))
