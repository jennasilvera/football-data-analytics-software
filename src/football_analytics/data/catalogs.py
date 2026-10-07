from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from football_analytics.data.entity_resolution import (
    CompetitionEntityResolver,
    TeamEntityResolver,
)
from football_analytics.domain import (
    Competition,
    CompetitionKind,
    Confederation,
    Team,
)


@dataclass(frozen=True, slots=True)
class CanonicalCatalogs:
    """Canonical entity catalogs plus their declared source aliases."""

    teams: tuple[Team, ...]
    team_aliases: dict[str, str]
    competitions: tuple[Competition, ...]
    competition_aliases: dict[str, str]

    def team_resolver(self) -> TeamEntityResolver:
        return TeamEntityResolver(self.teams, aliases=self.team_aliases)

    def competition_resolver(self) -> CompetitionEntityResolver:
        return CompetitionEntityResolver(
            self.competitions,
            aliases=self.competition_aliases,
        )


def load_canonical_catalogs(
    *,
    teams_path: str | Path,
    competitions_path: str | Path,
) -> CanonicalCatalogs:
    """Load governed JSON catalogs and validate them by constructing resolvers."""

    team_payload = _load_json(teams_path)
    competition_payload = _load_json(competitions_path)

    team_items = team_payload.get("teams", [])
    competition_items = competition_payload.get("competitions", [])

    if not isinstance(team_items, list):
        raise ValueError("teams must be a JSON array.")
    if not isinstance(competition_items, list):
        raise ValueError("competitions must be a JSON array.")

    teams = tuple(_team_from_dict(item) for item in team_items)
    competitions = tuple(
        _competition_from_dict(item)
        for item in competition_items
    )
    team_aliases = _aliases(team_payload.get("aliases", {}), "team")
    competition_aliases = _aliases(
        competition_payload.get("aliases", {}),
        "competition",
    )

    catalogs = CanonicalCatalogs(
        teams=teams,
        team_aliases=team_aliases,
        competitions=competitions,
        competition_aliases=competition_aliases,
    )

    catalogs.team_resolver()
    catalogs.competition_resolver()
    return catalogs


def _load_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Catalog must contain a JSON object: {source}")

    return payload


def _team_from_dict(item: object) -> Team:
    if not isinstance(item, dict):
        raise ValueError("Each team catalog entry must be an object.")

    confederation = item.get("confederation")
    return Team(
        team_id=str(item["team_id"]),
        name=str(item["name"]),
        fifa_code=_optional_string(item.get("fifa_code")),
        confederation=Confederation(str(confederation))
        if confederation is not None
        else None,
    )


def _competition_from_dict(item: object) -> Competition:
    if not isinstance(item, dict):
        raise ValueError("Each competition catalog entry must be an object.")

    confederation = item.get("confederation")
    senior_flag = item.get("senior_mens_a_international", True)

    if not isinstance(senior_flag, bool):
        raise ValueError("senior_mens_a_international must be boolean.")

    return Competition(
        competition_id=str(item["competition_id"]),
        name=str(item["name"]),
        kind=CompetitionKind(str(item["kind"])),
        confederation=Confederation(str(confederation))
        if confederation is not None
        else None,
        senior_mens_a_international=senior_flag,
    )


def _aliases(value: object, entity_type: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{entity_type} aliases must be a JSON object.")

    return {str(alias): str(entity_id) for alias, entity_id in value.items()}


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
