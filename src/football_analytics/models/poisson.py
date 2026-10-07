"""Canonical migration of the legacy independent attack/defence Poisson model.

No dataframe, filesystem, implicit team fallback, or post-cutoff result is used
by inference. The finite score grid is explicitly conditional on its bounds.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from football_analytics.data import CanonicalMatchRecord
from football_analytics.data.contracts import ensure_utc
from football_analytics.domain import Match, MatchStatus
from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.domain.scores import (
    REGULATION_TARGET_POLICY_ID,
    require_regulation_score,
)
from football_analytics.features.base import PredictionContext
from football_analytics.features.history import DEFAULT_RESULT_ELIGIBILITY_POLICY
from football_analytics.models.base import ModelFamily, ModelTrainingSpec

SCORE_CONTEXT_ID = "canonical_team_pair_neutral_v1"
SCORE_MISSING_POLICY_ID = "reject_unseen_team_v1"


def _digest(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return prefix + hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PoissonConfig:
    max_goals: int = 10
    home_advantage_multiplier: float = 1.10
    min_expected_goals: float = 0.20
    max_expected_goals: float = 5.00

    def __post_init__(self) -> None:
        if type(self.max_goals) is not int or not 3 <= self.max_goals <= 100:
            raise ValueError("max_goals must be an integer between 3 and 100.")
        for field in ("home_advantage_multiplier", "min_expected_goals", "max_expected_goals"):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive.")
            object.__setattr__(self, field, float(value))
        if self.min_expected_goals > self.max_expected_goals or self.max_expected_goals > 100:
            raise ValueError("Expected-goal bounds must be ordered and at most 100.")


DEFAULT_POISSON_CONFIG = PoissonConfig()


def poisson_spec(config: PoissonConfig = DEFAULT_POISSON_CONFIG) -> ModelTrainingSpec:
    return ModelTrainingSpec(
        "independent_poisson_v1",
        ModelFamily.POISSON,
        "legacy_parity_v1",
        parameters=tuple(sorted(asdict(config).items())),
    )


def config_from_spec(spec: ModelTrainingSpec) -> PoissonConfig:
    if spec.family is not ModelFamily.POISSON:
        raise ValueError("Expected Poisson model specification.")
    parameters = spec.parameter_dict()
    if set(parameters) != set(asdict(PoissonConfig())):
        raise ValueError("Poisson specification must declare all supported parameters.")
    max_goals = parameters["max_goals"]
    if type(max_goals) is not int:
        raise ValueError("max_goals must be an integer.")
    return PoissonConfig(
        max_goals=max_goals,
        home_advantage_multiplier=float(parameters["home_advantage_multiplier"]),
        min_expected_goals=float(parameters["min_expected_goals"]),
        max_expected_goals=float(parameters["max_expected_goals"]),
    )


@dataclass(frozen=True, slots=True)
class ScorelineCell:
    home_goals: int
    away_goals: int
    probability: float


@dataclass(frozen=True, slots=True)
class ScorelineForecast:
    expected_home_goals: float
    expected_away_goals: float
    probabilities: OutcomeProbabilities
    most_likely_home_goals: int
    most_likely_away_goals: int
    most_likely_score_probability: float
    outcome_entropy_nats: float
    omitted_tail_probability: float
    scorelines: tuple[ScorelineCell, ...]


def scoreline_forecast(
    home_rate: float, away_rate: float, *, max_goals: int = 10
) -> ScorelineForecast:
    PoissonConfig(max_goals=max_goals)
    if any(not math.isfinite(rate) or not 0 < rate <= 100 for rate in (home_rate, away_rate)):
        raise ValueError("Expected-goal rates must be finite, positive, and at most 100.")

    def pmfs(rate: float) -> list[float]:
        values = [math.exp(-rate)]
        for goals in range(1, max_goals + 1):
            values.append(values[-1] * rate / goals)
        return values

    home, away = pmfs(home_rate), pmfs(away_rate)
    retained = math.fsum(home) * math.fsum(away)
    cells = tuple(
        ScorelineCell(h, a, home[h] * away[a] / retained)
        for h in range(max_goals + 1)
        for a in range(max_goals + 1)
    )
    probabilities = OutcomeProbabilities(
        math.fsum(cell.probability for cell in cells if cell.home_goals > cell.away_goals),
        math.fsum(cell.probability for cell in cells if cell.home_goals == cell.away_goals),
        math.fsum(cell.probability for cell in cells if cell.home_goals < cell.away_goals),
    )
    # Ascending home/away goals break exact ties deterministically.
    mode = max(cells, key=lambda cell: cell.probability)
    entropy = -math.fsum(p * math.log(p) for p in probabilities.as_tuple() if p > 0)
    return ScorelineForecast(
        home_rate,
        away_rate,
        probabilities,
        mode.home_goals,
        mode.away_goals,
        mode.probability,
        entropy,
        max(0.0, 1.0 - retained),
        cells,
    )


def goal_dataset_id(records: Sequence[CanonicalMatchRecord]) -> str:
    """Hash actual goals, canonical context, result availability and provenance."""
    if not records:
        raise ValueError("Goal dataset must not be empty.")
    ids = [record.match.match_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Goal dataset contains duplicate match IDs.")
    fixtures: set[tuple[object, ...]] = set()
    rows = []
    for record in sorted(records, key=lambda item: item.match.match_id):
        require_regulation_score(record.score_basis, match_id=record.match.match_id)
        match = record.match
        identity = (
            match.match_date,
            tuple(sorted((match.home_team_id, match.away_team_id))),
            match.competition_id,
        )
        if identity in fixtures:
            raise ValueError("Goal dataset contains duplicate or reversed fixtures.")
        fixtures.add(identity)
        for score in (record.home_score, record.away_score):
            if type(score) is not int or score < 0:
                raise ValueError("Goal targets must be non-negative integers.")
        eligible = DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(record)
        if type(match.neutral) is not bool:
            raise ValueError("Neutral-site context must be boolean.")
        rows.append(
            {
                "match_id": match.match_id,
                "match_date": match.match_date.isoformat(),
                "kickoff": match.kickoff_at.isoformat() if match.kickoff_at else None,
                "home": match.home_team_id,
                "away": match.away_team_id,
                "competition": match.competition_id,
                "neutral": match.neutral,
                "score_basis": record.score_basis.value,
                "home_score": record.home_score,
                "away_score": record.away_score,
                "target_available_at": eligible.eligible_at.isoformat(),
                "target_availability_basis": eligible.basis.value,
                "source": record.metadata.source,
                "source_record_id": record.source_match_id,
                "source_version": record.metadata.source_version,
            }
        )
    return _digest("goal_dataset_", {
        "schema_version": 2, "target_policy_id": REGULATION_TARGET_POLICY_ID, "rows": rows
    })


@dataclass(frozen=True, slots=True)
class TeamGoalStrength:
    team_id: str
    appearances: int
    attack: float
    defence_weakness: float

    def __post_init__(self) -> None:
        if not self.team_id.strip() or type(self.appearances) is not int or self.appearances <= 0:
            raise ValueError("Team strength requires an ID and positive appearance count.")
        if any(not math.isfinite(x) or x < 0 for x in (self.attack, self.defence_weakness)):
            raise ValueError("Team goal strengths must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class PoissonModel:
    model_id: str
    training_dataset_id: str
    training_cutoff: datetime
    config: PoissonConfig
    global_goals_per_team_match: float
    teams: tuple[TeamGoalStrength, ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "training_cutoff", ensure_utc(self.training_cutoff, "training_cutoff")
        )
        if self.schema_version != 2 or not self.training_dataset_id.startswith("goal_dataset_"):
            raise ValueError("Unsupported Poisson model schema or dataset identity.")
        if (
            not math.isfinite(self.global_goals_per_team_match)
            or self.global_goals_per_team_match <= 0
        ):
            raise ValueError("Global goal rate must be finite and positive.")
        if not self.teams or len({team.team_id for team in self.teams}) != len(self.teams):
            raise ValueError("Model must contain unique team strengths.")
        object.__setattr__(self, "teams", tuple(sorted(self.teams, key=lambda team: team.team_id)))
        if self.model_id != _digest("poisson_model_", self._payload()):
            raise ValueError("Poisson model content does not match its identity.")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_policy_id": REGULATION_TARGET_POLICY_ID,
            "training_dataset_id": self.training_dataset_id,
            "training_cutoff": self.training_cutoff.isoformat(),
            "config": asdict(self.config),
            "global_goals_per_team_match": self.global_goals_per_team_match,
            "teams": [asdict(team) for team in self.teams],
            "result_eligibility_policy_id": DEFAULT_RESULT_ELIGIBILITY_POLICY.policy_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, **self._payload()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PoissonModel:
        if (
            payload.get("result_eligibility_policy_id")
            != DEFAULT_RESULT_ELIGIBILITY_POLICY.policy_id
        ):
            raise ValueError("Unsupported result eligibility policy.")
        if payload.get("target_policy_id") != REGULATION_TARGET_POLICY_ID:
            raise ValueError("Unsupported score target policy.")
        return cls(
            model_id=payload["model_id"],
            training_dataset_id=payload["training_dataset_id"],
            training_cutoff=datetime.fromisoformat(payload["training_cutoff"]),
            config=PoissonConfig(**payload["config"]),
            global_goals_per_team_match=payload["global_goals_per_team_match"],
            teams=tuple(TeamGoalStrength(**team) for team in payload["teams"]),
            schema_version=payload["schema_version"],
        )

    def predict_match(self, match: Match, *, prediction_time: datetime) -> ScorelineForecast:
        context = PredictionContext(match, prediction_time)
        if context.prediction_time < self.training_cutoff:
            raise ValueError("Model training cutoff is after prediction_time.")
        if match.status in (MatchStatus.CANCELLED, MatchStatus.POSTPONED):
            raise ValueError("Cannot forecast a cancelled or postponed fixture.")
        if type(match.neutral) is not bool:
            raise ValueError("Neutral-site context must be boolean.")
        strengths = {team.team_id: team for team in self.teams}
        missing = {match.home_team_id, match.away_team_id} - strengths.keys()
        if missing:
            raise ValueError(f"No fitted goal history for teams: {sorted(missing)}")
        home, away = strengths[match.home_team_id], strengths[match.away_team_id]
        multiplier = 1.0 if match.neutral else self.config.home_advantage_multiplier
        home_rate = (
            self.global_goals_per_team_match * home.attack * away.defence_weakness * multiplier
        )
        away_rate = (
            self.global_goals_per_team_match * away.attack * home.defence_weakness / multiplier
        )

        def clip(rate: float) -> float:
            return max(self.config.min_expected_goals, min(self.config.max_expected_goals, rate))

        return scoreline_forecast(clip(home_rate), clip(away_rate), max_goals=self.config.max_goals)


def fit_poisson(
    records: Sequence[CanonicalMatchRecord],
    *,
    training_cutoff: datetime,
    config: PoissonConfig = DEFAULT_POISSON_CONFIG,
) -> PoissonModel:
    """Fit only the supplied, cutoff-eligible records; never silently filter input."""
    cutoff = ensure_utc(training_cutoff, "training_cutoff")
    dataset_id = goal_dataset_id(records)
    stats: dict[str, list[int]] = {}
    for record in records:
        if DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(record).eligible_at > cutoff:
            raise ValueError("Training result is not available by training_cutoff.")
        assert record.home_score is not None and record.away_score is not None
        for team, scored, conceded in (
            (record.match.home_team_id, record.home_score, record.away_score),
            (record.match.away_team_id, record.away_score, record.home_score),
        ):
            values = stats.setdefault(team, [0, 0, 0])
            values[0] += 1
            values[1] += scored
            values[2] += conceded
    global_rate = sum(values[1] for values in stats.values()) / (2 * len(records))
    if global_rate <= 0:
        raise ValueError("Cannot fit Poisson model with zero average goals.")
    teams = tuple(
        TeamGoalStrength(team, n, (goals_for / n) / global_rate, (goals_against / n) / global_rate)
        for team, (n, goals_for, goals_against) in sorted(stats.items())
    )
    payload = {
        "schema_version": 2, "target_policy_id": REGULATION_TARGET_POLICY_ID,
        "training_dataset_id": dataset_id,
        "training_cutoff": cutoff.isoformat(),
        "config": asdict(config),
        "global_goals_per_team_match": global_rate,
        "teams": [asdict(team) for team in teams],
        "result_eligibility_policy_id": DEFAULT_RESULT_ELIGIBILITY_POLICY.policy_id,
    }
    return PoissonModel(
        _digest("poisson_model_", payload), dataset_id, cutoff, config, global_rate, teams
    )
