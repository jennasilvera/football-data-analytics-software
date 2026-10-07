from __future__ import annotations

import math
from dataclasses import dataclass

from football_analytics.domain import MatchOutcome


PROBABILITY_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class OutcomeProbabilities:
    """Validated three-way football outcome probability distribution."""

    home_win: float
    draw: float
    away_win: float

    def __post_init__(self) -> None:
        values = (self.home_win, self.draw, self.away_win)

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Outcome probabilities must be finite.")

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Outcome probabilities must be between 0 and 1.")

        total = sum(values)
        if not math.isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=PROBABILITY_SUM_TOLERANCE,
        ):
            raise ValueError(
                "Outcome probabilities must sum to 1. "
                f"Received {total:.12f}."
            )

    def as_tuple(self) -> tuple[float, float, float]:
        """Return probabilities in canonical home/draw/away order."""

        return (self.home_win, self.draw, self.away_win)

    def probability_for(self, outcome: MatchOutcome) -> float:
        """Return the probability assigned to one canonical outcome."""

        if outcome is MatchOutcome.HOME_WIN:
            return self.home_win
        if outcome is MatchOutcome.DRAW:
            return self.draw
        if outcome is MatchOutcome.AWAY_WIN:
            return self.away_win

        raise ValueError(f"Unsupported match outcome: {outcome}")

    @property
    def predicted_outcome(self) -> MatchOutcome:
        """Return the maximum-probability outcome with deterministic tie order."""

        outcomes = (
            MatchOutcome.HOME_WIN,
            MatchOutcome.DRAW,
            MatchOutcome.AWAY_WIN,
        )
        probabilities = self.as_tuple()
        winning_index = max(
            range(len(probabilities)),
            key=probabilities.__getitem__,
        )
        return outcomes[winning_index]
