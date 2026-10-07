from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from football_analytics.data.contracts import LeakageRisk, SourceMetadata, ensure_utc
from football_analytics.data.observations import MatchObservation
from football_analytics.data.scope import GenderCategory, TeamLevel
from football_analytics.domain import MatchStatus


@dataclass(frozen=True, slots=True)
class TabularMatchColumns:
    """Column mapping for a tabular match source."""

    match_date: str
    home_team: str
    away_team: str
    competition: str
    neutral: str
    kickoff_at: str | None = None
    status: str | None = None
    home_score: str | None = None
    away_score: str | None = None
    source_match_id: str | None = None
    available_at: str | None = None
    venue_name: str | None = None


@dataclass(frozen=True, slots=True)
class TabularSourcePolicy:
    """Source-level assertions that must be explicit before row normalization."""

    source_id: str
    gender: GenderCategory
    team_level: TeamLevel
    official: bool | None
    default_status: MatchStatus
    leakage_risk: LeakageRisk
    source_version: str | None = None
    legal_use_notes: str | None = None
    status_map: Mapping[str, MatchStatus] | None = None

    def __post_init__(self) -> None:
        source_id = self.source_id.strip()
        if not source_id:
            raise ValueError("source_id must not be blank.")
        object.__setattr__(self, "source_id", source_id)


def build_match_observations(
    frame: pd.DataFrame,
    *,
    columns: TabularMatchColumns,
    policy: TabularSourcePolicy,
    ingested_at: datetime,
) -> list[MatchObservation]:
    """Translate a dataframe into source observations without canonicalizing entities."""

    if frame.empty:
        return []

    ingested_at_utc = ensure_utc(ingested_at, "ingested_at")
    required = {
        columns.match_date,
        columns.home_team,
        columns.away_team,
        columns.competition,
        columns.neutral,
    }

    for optional in (
        columns.kickoff_at,
        columns.status,
        columns.home_score,
        columns.away_score,
        columns.source_match_id,
        columns.available_at,
        columns.venue_name,
    ):
        if optional is not None:
            required.add(optional)

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Tabular source missing required columns: {missing}")

    observations: list[MatchObservation] = []

    for row_number, row in frame.reset_index(drop=True).iterrows():
        match_date = pd.to_datetime(
            row[columns.match_date],
            errors="raise",
        ).date()

        kickoff_at = _optional_aware_datetime(
            row[columns.kickoff_at] if columns.kickoff_at else None,
            field_name="kickoff_at",
        )
        available_at = _optional_aware_datetime(
            row[columns.available_at] if columns.available_at else None,
            field_name="available_at",
        )

        home_team = _required_text(row[columns.home_team], "home_team", row_number)
        away_team = _required_text(row[columns.away_team], "away_team", row_number)
        competition = _required_text(
            row[columns.competition],
            "competition",
            row_number,
        )

        source_match_id = _source_match_id(
            row=row,
            row_number=row_number,
            configured_column=columns.source_match_id,
            source_id=policy.source_id,
            match_date=match_date.isoformat(),
            home_team=home_team,
            away_team=away_team,
            competition=competition,
        )

        status = _match_status(
            row=row,
            configured_column=columns.status,
            policy=policy,
        )

        home_score = _optional_score(
            row[columns.home_score] if columns.home_score else None,
            "home_score",
        )
        away_score = _optional_score(
            row[columns.away_score] if columns.away_score else None,
            "away_score",
        )

        venue_name = None
        if columns.venue_name is not None:
            venue_name = _optional_text(row[columns.venue_name])

        observations.append(
            MatchObservation(
                source_match_id=source_match_id,
                match_date=match_date,
                kickoff_at=kickoff_at,
                home_team_name=home_team,
                away_team_name=away_team,
                competition_name=competition,
                neutral=_parse_bool(row[columns.neutral]),
                status=status,
                gender=policy.gender,
                team_level=policy.team_level,
                official=policy.official,
                metadata=SourceMetadata(
                    source=policy.source_id,
                    source_record_id=source_match_id,
                    source_version=policy.source_version,
                    event_time=kickoff_at,
                    available_at=available_at,
                    ingested_at=ingested_at_utc,
                    leakage_risk=policy.leakage_risk,
                    legal_use_notes=policy.legal_use_notes,
                ),
                home_score=home_score,
                away_score=away_score,
                venue_name=venue_name,
            )
        )

    return observations


def _required_text(value: object, field_name: str, row_number: int) -> str:
    if pd.isna(value):
        raise ValueError(f"Row {row_number} has missing {field_name}.")

    text = str(value).strip()
    if not text:
        raise ValueError(f"Row {row_number} has blank {field_name}.")
    return text


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_aware_datetime(
    value: object,
    *,
    field_name: str,
) -> datetime | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{field_name} must include timezone information.")

    converted: datetime = timestamp.tz_convert("UTC").to_pydatetime()
    return converted


def _optional_score(value: object, field_name: str) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None

    numeric = float(str(value))
    if numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{field_name} must be a non-negative whole number.")
    return int(numeric)


def _match_status(
    *,
    row: pd.Series,
    configured_column: str | None,
    policy: TabularSourcePolicy,
) -> MatchStatus:
    if configured_column is None:
        return policy.default_status

    raw = _optional_text(row[configured_column])
    if raw is None:
        return policy.default_status

    normalized = raw.casefold()
    if policy.status_map is not None:
        mapping = {str(key).casefold(): value for key, value in policy.status_map.items()}
        if normalized in mapping:
            return mapping[normalized]

    try:
        return MatchStatus(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported match status: {raw}") from exc


def _source_match_id(
    *,
    row: pd.Series,
    row_number: int,
    configured_column: str | None,
    source_id: str,
    match_date: str,
    home_team: str,
    away_team: str,
    competition: str,
) -> str:
    if configured_column is not None:
        return _required_text(row[configured_column], "source_match_id", row_number)

    identity = "|".join(
        [source_id, match_date, home_team, away_team, competition]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"{source_id}:{digest}"


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Could not parse boolean value: {value!r}")
