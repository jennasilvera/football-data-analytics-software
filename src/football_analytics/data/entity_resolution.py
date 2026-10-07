from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from football_analytics.domain import Competition, Team


def normalize_entity_name(value: str) -> str:
    """Normalize a source label without removing meaningful accents."""

    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", normalized)


class ResolutionStatus(StrEnum):
    """Outcome of resolving a source label to a canonical entity."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class TeamResolution:
    status: ResolutionStatus
    source_name: str
    team: Team | None = None
    matched_by: str | None = None


@dataclass(frozen=True, slots=True)
class CompetitionResolution:
    status: ResolutionStatus
    source_name: str
    competition: Competition | None = None
    matched_by: str | None = None


class TeamEntityResolver:
    """Resolve source team names to canonical Team identities.

    Alias mappings point to canonical team IDs. Conflicting aliases are rejected
    when the resolver is created.
    """

    def __init__(
        self,
        teams: Iterable[Team],
        *,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self._by_id: dict[str, Team] = {}
        self._by_name: dict[str, Team] = {}
        self._aliases: dict[str, Team] = {}

        for team in teams:
            if team.team_id in self._by_id:
                raise ValueError(f"Duplicate team_id: {team.team_id}")

            name_key = normalize_entity_name(team.name)
            if name_key in self._by_name:
                raise ValueError(f"Duplicate canonical team name: {team.name}")

            self._by_id[team.team_id] = team
            self._by_name[name_key] = team

        for alias, team_id in (aliases or {}).items():
            alias_key = normalize_entity_name(alias)

            if not alias_key:
                raise ValueError("Team alias must not be blank.")

            if team_id not in self._by_id:
                raise ValueError(
                    f"Team alias {alias!r} references unknown team_id {team_id!r}."
                )

            canonical = self._by_id[team_id]
            direct_match = self._by_name.get(alias_key)

            if direct_match is not None and direct_match.team_id != canonical.team_id:
                raise ValueError(
                    f"Team alias {alias!r} conflicts with canonical team "
                    f"{direct_match.team_id!r}."
                )

            existing_alias = self._aliases.get(alias_key)
            if existing_alias is not None and existing_alias.team_id != canonical.team_id:
                raise ValueError(f"Conflicting team alias: {alias!r}")

            self._aliases[alias_key] = canonical

    def resolve(self, source_name: str) -> TeamResolution:
        source_name = str(source_name).strip()
        key = normalize_entity_name(source_name)

        if not key:
            return TeamResolution(
                status=ResolutionStatus.UNRESOLVED,
                source_name=source_name,
            )

        direct = self._by_name.get(key)
        if direct is not None:
            return TeamResolution(
                status=ResolutionStatus.RESOLVED,
                source_name=source_name,
                team=direct,
                matched_by="canonical_name",
            )

        alias = self._aliases.get(key)
        if alias is not None:
            return TeamResolution(
                status=ResolutionStatus.RESOLVED,
                source_name=source_name,
                team=alias,
                matched_by="alias",
            )

        return TeamResolution(
            status=ResolutionStatus.UNRESOLVED,
            source_name=source_name,
        )


class CompetitionEntityResolver:
    """Resolve source competition labels to canonical Competition identities."""

    def __init__(
        self,
        competitions: Iterable[Competition],
        *,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self._by_id: dict[str, Competition] = {}
        self._by_name: dict[str, Competition] = {}
        self._aliases: dict[str, Competition] = {}

        for competition in competitions:
            if competition.competition_id in self._by_id:
                raise ValueError(
                    f"Duplicate competition_id: {competition.competition_id}"
                )

            name_key = normalize_entity_name(competition.name)
            if name_key in self._by_name:
                raise ValueError(
                    f"Duplicate canonical competition name: {competition.name}"
                )

            self._by_id[competition.competition_id] = competition
            self._by_name[name_key] = competition

        for alias, competition_id in (aliases or {}).items():
            alias_key = normalize_entity_name(alias)

            if not alias_key:
                raise ValueError("Competition alias must not be blank.")

            if competition_id not in self._by_id:
                raise ValueError(
                    f"Competition alias {alias!r} references unknown competition_id "
                    f"{competition_id!r}."
                )

            canonical = self._by_id[competition_id]
            direct_match = self._by_name.get(alias_key)

            if (
                direct_match is not None
                and direct_match.competition_id != canonical.competition_id
            ):
                raise ValueError(
                    f"Competition alias {alias!r} conflicts with canonical "
                    f"competition {direct_match.competition_id!r}."
                )

            existing_alias = self._aliases.get(alias_key)
            if (
                existing_alias is not None
                and existing_alias.competition_id != canonical.competition_id
            ):
                raise ValueError(f"Conflicting competition alias: {alias!r}")

            self._aliases[alias_key] = canonical

    def resolve(self, source_name: str) -> CompetitionResolution:
        source_name = str(source_name).strip()
        key = normalize_entity_name(source_name)

        if not key:
            return CompetitionResolution(
                status=ResolutionStatus.UNRESOLVED,
                source_name=source_name,
            )

        direct = self._by_name.get(key)
        if direct is not None:
            return CompetitionResolution(
                status=ResolutionStatus.RESOLVED,
                source_name=source_name,
                competition=direct,
                matched_by="canonical_name",
            )

        alias = self._aliases.get(key)
        if alias is not None:
            return CompetitionResolution(
                status=ResolutionStatus.RESOLVED,
                source_name=source_name,
                competition=alias,
                matched_by="alias",
            )

        return CompetitionResolution(
            status=ResolutionStatus.UNRESOLVED,
            source_name=source_name,
        )
