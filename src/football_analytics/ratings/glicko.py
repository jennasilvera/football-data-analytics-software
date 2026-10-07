"""Glicko-1 period updates; daily football periods with declared RD inflation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from football_analytics.data.normalization import CanonicalMatchRecord
from football_analytics.features.history import completed_record_is_before_cutoff

Q = math.log(10) / 400


@dataclass(frozen=True, slots=True)
class GlickoState:
    rating: float = 1500.0
    deviation: float = 350.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.rating) or not 0 < self.deviation <= 350:
            raise ValueError("Glicko requires a finite rating and RD in (0,350].")


def update_period(
    state: GlickoState, opponents: Sequence[tuple[GlickoState, float]]
) -> GlickoState:
    if not opponents:
        return state
    information = residual = 0.0
    for other, outcome in opponents:
        if outcome not in (0, 0.5, 1):
            raise ValueError("Glicko outcomes must be 0, 0.5 or 1.")
        g = 1 / math.sqrt(1 + 3 * Q**2 * other.deviation**2 / math.pi**2)
        exponent = max(-300, min(300, -g * (state.rating - other.rating) / 400))
        expected = 1 / (1 + 10**exponent)
        information += Q**2 * g**2 * expected * (1 - expected)
        residual += g * (outcome - expected)
    precision = 1 / state.deviation**2 + information
    return GlickoState(state.rating + Q / precision * residual, math.sqrt(1 / precision))


def inflate(state: GlickoState, days: float, monthly_drift: float = 50) -> GlickoState:
    if not math.isfinite(days) or days < 0 or not math.isfinite(monthly_drift) or monthly_drift < 0:
        raise ValueError("Rating inactivity and drift must be finite and non-negative.")
    return GlickoState(
        state.rating, min(350, math.sqrt(state.deviation**2 + monthly_drift**2 * days / 30))
    )


def rating_history(records: Sequence[CanonicalMatchRecord], *, as_of: datetime) -> list[dict]:
    periods: dict[date, list[CanonicalMatchRecord]] = defaultdict(list)
    seen = set()
    for record in records:
        if record.match.match_id in seen:
            raise ValueError("Duplicate rating fixture.")
        seen.add(record.match.match_id)
        if completed_record_is_before_cutoff(record, as_of):
            if any(type(x) is not int or x < 0 for x in (record.home_score, record.away_score)):
                raise ValueError("Glicko goals must be nonnegative integers.")
            periods[record.match.match_date].append(record)
    states: dict[str, GlickoState] = {}
    last: dict[str, date] = {}
    history = []
    for day, matches in sorted(periods.items()):
        teams = {t for m in matches for t in (m.match.home_team_id, m.match.away_team_id)}
        prior = {
            t: inflate(states.get(t, GlickoState()), (day - last.get(t, day)).days) for t in teams
        }
        games: dict[str, list[tuple[GlickoState, float]]] = defaultdict(list)
        for record in matches:
            h, a = record.match.home_team_id, record.match.away_team_id
            assert record.home_score is not None and record.away_score is not None
            outcome = (
                1.0
                if record.home_score > record.away_score
                else 0.0
                if record.home_score < record.away_score
                else 0.5
            )
            games[h].append((prior[a], outcome))
            games[a].append((prior[h], 1 - outcome))
        for team in sorted(teams):
            states[team] = update_period(prior[team], games[team])
            last[team] = day
            history.append(
                {
                    "team_id": team,
                    "effective_date": day.isoformat(),
                    "rating": states[team].rating,
                    "rating_before": prior[team].rating,
                    "deviation_before": prior[team].deviation,
                    "deviation": states[team].deviation,
                    "rating_change": states[team].rating - prior[team].rating,
                    "model_id": "glicko1_daily_rd50_per_month_v1",
                    "source_match_ids": sorted(
                        m.match.match_id
                        for m in matches
                        if team in (m.match.home_team_id, m.match.away_team_id)
                    ),
                }
            )
    return history
