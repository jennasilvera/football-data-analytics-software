from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Confederation(StrEnum):
    """FIFA-recognized confederation identifiers."""

    AFC = "AFC"
    CAF = "CAF"
    CONCACAF = "CONCACAF"
    CONMEBOL = "CONMEBOL"
    OFC = "OFC"
    UEFA = "UEFA"


@dataclass(frozen=True, slots=True)
class Team:
    """Canonical senior national-team identity."""

    team_id: str
    name: str
    fifa_code: str | None = None
    confederation: Confederation | None = None

    def __post_init__(self) -> None:
        team_id = self.team_id.strip()
        name = self.name.strip()

        if not team_id:
            raise ValueError("team_id must not be blank.")

        if not name:
            raise ValueError("name must not be blank.")

        object.__setattr__(self, "team_id", team_id)
        object.__setattr__(self, "name", name)

        if self.fifa_code is None:
            return

        fifa_code = self.fifa_code.strip().upper()

        if len(fifa_code) != 3 or not fifa_code.isalpha():
            raise ValueError("fifa_code must contain exactly three letters.")

        object.__setattr__(self, "fifa_code", fifa_code)
