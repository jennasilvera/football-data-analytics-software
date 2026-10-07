"""As-of team reports and descriptive forecast residual watchlists."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from football_analytics.data.serialization import record_from_dict
from football_analytics.domain import MatchOutcome
from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.features.history import completed_record_is_before_cutoff
from football_analytics.ratings.glicko import GlickoState, inflate, rating_history
from football_analytics.storage.repository import Repository


def available_records(repo: Repository, as_of: datetime) -> list:
    # Operational state is bitemporal: revisions recorded later are never visible here.
    return [record_from_dict(row["payload"]) for row in repo.scan("match", as_of=as_of)]


def ratings(repo: Repository, as_of: datetime) -> list[dict[str, Any]]:
    history = rating_history(available_records(repo, as_of), as_of=as_of)
    latest = {row["team_id"]: row for row in history}
    for row in latest.values():
        from datetime import date

        state = inflate(
            GlickoState(row["rating"], row["deviation"]),
            (as_of.date() - date.fromisoformat(row["effective_date"])).days,
        )
        row["deviation"] = state.deviation
        row["rating_interval"] = [
            state.rating - 1.96 * state.deviation,
            state.rating + 1.96 * state.deviation,
        ]
        row["interval_note"] = "Glicko rating-scale approximation; football coverage unvalidated"
    return sorted(latest.values(), key=lambda row: (-row["rating"], row["team_id"]))


def post_match_diagnostics(repo: Repository, as_of: datetime) -> list[dict[str, Any]]:
    records = {
        r.match.match_id: r
        for r in available_records(repo, as_of)
        if completed_record_is_before_cutoff(r, as_of)
    }
    forecasts = [r["payload"] for r in repo.scan("forecast", as_of=as_of)]
    latest: dict[str, dict[str, Any]] = {}
    for forecast in forecasts:
        match_id = forecast["match"]["match_id"]
        if (
            match_id not in latest
            or forecast["prediction_time"] > latest[match_id]["prediction_time"]
        ):
            latest[match_id] = forecast
    result = []
    for match_id, forecast in latest.items():
        record = records.get(match_id)
        if record is None:
            continue
        assert record.home_score is not None and record.away_score is not None
        actual = (
            MatchOutcome.HOME_WIN
            if record.home_score > record.away_score
            else MatchOutcome.AWAY_WIN
            if record.home_score < record.away_score
            else MatchOutcome.DRAW
        )
        p = OutcomeProbabilities(**forecast["probabilities"])
        home_result = (
            1.0 if actual is MatchOutcome.HOME_WIN else 0.5 if actual is MatchOutcome.DRAW else 0
        )
        residual = home_result - (p.home_win + 0.5 * p.draw)
        result.append(
            {
                "match_id": match_id,
                "forecast_id": forecast["forecast_id"],
                "home_team_id": record.match.home_team_id,
                "away_team_id": record.match.away_team_id,
                "match_date": record.match.match_date.isoformat(),
                "actual": actual.value,
                "actual_probability": p.probability_for(actual),
                "surprisal_nats": -math.log(max(p.probability_for(actual), 1e-15)),
                "home_residual": residual,
                "away_residual": -residual,
                "miss_classification": "correct"
                if p.predicted_outcome is actual
                else "outcome_miss",
                "causal_explanation": None,
            }
        )
    return sorted(result, key=lambda row: (-row["surprisal_nats"], row["match_id"]))


def team_report(repo: Repository, team_id: str, as_of: datetime) -> dict[str, Any]:
    team = repo.get("team", team_id, as_of=as_of)["payload"]
    records = sorted(
        (
            r
            for r in available_records(repo, as_of)
            if team_id in (r.match.home_team_id, r.match.away_team_id)
            and completed_record_is_before_cutoff(r, as_of)
        ),
        key=lambda r: (r.match.match_date, r.match.match_id),
    )
    form = []
    for record in records[-20:]:
        home = team_id == record.match.home_team_id
        gf, ga = (
            (record.home_score, record.away_score)
            if home
            else (record.away_score, record.home_score)
        )
        assert gf is not None and ga is not None
        form.append(
            {
                "match_id": record.match.match_id,
                "date": record.match.match_date.isoformat(),
                "goals_for": gf,
                "goals_against": ga,
                "points": 3 if gf > ga else 1 if gf == ga else 0,
            }
        )
    # Rating history must include opponents' complete histories, not just this team's games.
    all_history = rating_history(available_records(repo, as_of), as_of=as_of)
    history = [row for row in all_history if row["team_id"] == team_id]
    before = {(r["team_id"], r["effective_date"]): r for r in all_history}
    schedule = []
    for record in records[-20:]:
        opponent = (
            record.match.away_team_id
            if record.match.home_team_id == team_id
            else record.match.home_team_id
        )
        day = record.match.match_date.isoformat()
        own, other = before[(team_id, day)], before[(opponent, day)]
        from football_analytics.ratings.glicko import Q

        g = 1 / math.sqrt(
            1
            + 3
            * Q**2
            * (own["deviation_before"] ** 2 + other["deviation_before"] ** 2)
            / math.pi**2
        )
        expected = 1 / (
            1
            + 10 ** max(-300, min(300, -g * (own["rating_before"] - other["rating_before"]) / 400))
        )
        assert record.home_score is not None and record.away_score is not None
        gf, ga = (
            (record.home_score, record.away_score)
            if record.match.home_team_id == team_id
            else (record.away_score, record.home_score)
        )
        observed = 1 if gf > ga else 0.5 if gf == ga else 0
        schedule.append(
            {"opponent_rating": other["rating_before"], "score_residual": observed - expected}
        )
    residuals = sorted(
        (
            row
            for row in post_match_diagnostics(repo, as_of)
            if team_id in (row["home_team_id"], row["away_team_id"])
        ),
        key=lambda row: (row["match_date"], row["match_id"]),
    )[-10:]
    residual = sum(
        row["home_residual"] if row["home_team_id"] == team_id else row["away_residual"]
        for row in residuals
    )
    upcoming = sorted(
        (
            r
            for r in available_records(repo, as_of)
            if r.match.status.value == "scheduled"
            and team_id in (r.match.home_team_id, r.match.away_team_id)
            and r.match.match_date > as_of.date()
        ),
        key=lambda r: (r.match.match_date, r.match.match_id),
    )
    next_match = upcoming[0] if upcoming else None
    next_forecasts = [
        r["payload"]
        for r in repo.scan("forecast", as_of=as_of)
        if next_match and r["payload"]["match"]["match_id"] == next_match.match.match_id
    ]
    next_forecast = max(next_forecasts, key=lambda r: r["prediction_time"], default=None)
    return {
        "team": team,
        "as_of": as_of.isoformat(),
        "matches_available": len(records),
        "rating": next((row for row in ratings(repo, as_of) if row["team_id"] == team_id), None),
        "rating_history": history,
        "recent_form": form,
        "form_windows": {
            str(n): {
                "matches": len(form[-n:]),
                "points_per_match": sum(r["points"] for r in form[-n:]) / len(form[-n:])
                if form
                else None,
                "goal_difference": sum(r["goals_for"] - r["goals_against"] for r in form[-n:]),
            }
            for n in (5, 10, 20)
        },
        "strength_of_schedule": sum(r["opponent_rating"] for r in schedule) / len(schedule)
        if schedule
        else None,
        "opponent_adjusted_score_residual": sum(r["score_residual"] for r in schedule)
        / len(schedule)
        if schedule
        else None,
        "schedule_note": (
            "Last 20 matches, pre-period Glicko opponent strength and score residual; "
            "retrospective as-of replay"
        ),
        "forecast_residual_sum": residual,
        "forecast_sample": len(residuals),
        "watch_signal": "insufficient_history"
        if len(residuals) < 5
        else "positive_residual"
        if residual >= 1.5
        else "negative_residual"
        if residual <= -1.5
        else "none",
        "watch_note": (
            "Descriptive last-ten forecast residuals, not a "
            "validated future regression/breakout prediction"
        ),
        "next_match_outlook": {
            "match_id": next_match.match.match_id,
            "match_date": next_match.match.match_date.isoformat(),
            "latest_forecast": next_forecast,
        }
        if next_match
        else None,
    }
