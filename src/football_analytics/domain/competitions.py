from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from football_analytics.domain.teams import Confederation


class CompetitionKind(StrEnum):
    """Competition categories supported by the platform's formal scope."""

    WORLD_CUP = "world_cup"
    WORLD_CUP_QUALIFIER = "world_cup_qualifier"
    CONTINENTAL_CHAMPIONSHIP = "continental_championship"
    CONTINENTAL_QUALIFIER = "continental_qualifier"
    NATIONS_LEAGUE = "nations_league"
    FRIENDLY = "friendly"
    FIFA_SERIES = "fifa_series"
    REGIONAL_SENIOR_TOURNAMENT = "regional_senior_tournament"
    INTERCONTINENTAL_PLAYOFF = "intercontinental_playoff"
    OTHER_SENIOR_A = "other_senior_a"


@dataclass(frozen=True, slots=True)
class Competition:
    """Canonical competition identity for a senior men's A-international match."""

    competition_id: str
    name: str
    kind: CompetitionKind
    confederation: Confederation | None = None
    senior_mens_a_international: bool = True

    def __post_init__(self) -> None:
        competition_id = self.competition_id.strip()
        name = self.name.strip()

        if not competition_id:
            raise ValueError("competition_id must not be blank.")

        if not name:
            raise ValueError("name must not be blank.")

        object.__setattr__(self, "competition_id", competition_id)
        object.__setattr__(self, "name", name)
