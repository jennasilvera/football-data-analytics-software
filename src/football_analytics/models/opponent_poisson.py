"""Penalized joint attack/defence Poisson challenger for governed research.

Fits two goal observations per match, adjusting each attack for its opponents'
defence. This is a statistical goal-rate model, not shot-based expected goals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
from scipy.optimize import minimize

from football_analytics.data import CanonicalMatchRecord
from football_analytics.data.contracts import ensure_utc
from football_analytics.domain import Match, MatchStatus
from football_analytics.domain.scores import REGULATION_TARGET_POLICY_ID
from football_analytics.features.base import PredictionContext
from football_analytics.features.history import DEFAULT_RESULT_ELIGIBILITY_POLICY
from football_analytics.models.base import ModelFamily, ModelTrainingSpec
from football_analytics.models.poisson import (
    PoissonConfig,
    ScorelineForecast,
    _digest,
    goal_dataset_id,
    scoreline_forecast,
)


@dataclass(frozen=True, slots=True)
class OpponentPoissonConfig:
    penalty: float = 5.0
    max_iterations: int = 1000
    tolerance: float = 1e-9
    max_goals: int = 10
    min_expected_goals: float = 0.20
    max_expected_goals: float = 5.0

    def __post_init__(self) -> None:
        for name in ("penalty", "tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, float(value))
        if type(self.max_iterations) is not int or not 1 <= self.max_iterations <= 100000:
            raise ValueError("max_iterations must be an integer in 1..100000.")
        grid = PoissonConfig(
            max_goals=self.max_goals,
            min_expected_goals=self.min_expected_goals,
            max_expected_goals=self.max_expected_goals,
        )
        object.__setattr__(self, "min_expected_goals", grid.min_expected_goals)
        object.__setattr__(self, "max_expected_goals", grid.max_expected_goals)


DEFAULT_OPPONENT_POISSON_CONFIG = OpponentPoissonConfig()


def opponent_poisson_spec(
    config: OpponentPoissonConfig = DEFAULT_OPPONENT_POISSON_CONFIG,
) -> ModelTrainingSpec:
    return ModelTrainingSpec(
        "opponent_adjusted_poisson_v1",
        ModelFamily.OPPONENT_POISSON,
        "joint_ridge_v1",
        parameters=tuple(sorted(asdict(config).items())),
    )


def opponent_config_from_spec(spec: ModelTrainingSpec) -> OpponentPoissonConfig:
    if spec.family is not ModelFamily.OPPONENT_POISSON:
        raise ValueError("Expected opponent-adjusted Poisson specification.")
    parameters: dict[str, Any] = spec.parameter_dict()
    if set(parameters) != set(asdict(OpponentPoissonConfig())):
        raise ValueError("Opponent Poisson specification must declare all supported parameters.")
    return OpponentPoissonConfig(**parameters)


@dataclass(frozen=True, slots=True)
class OpponentStrength:
    team_id: str
    appearances: int
    log_attack: float
    log_defence_weakness: float

    def __post_init__(self) -> None:
        if not self.team_id.strip() or type(self.appearances) is not int or self.appearances < 1:
            raise ValueError("Team strength requires an ID and positive appearance count.")
        if any(
            not math.isfinite(v) or abs(v) > 6.000000001
            for v in (self.log_attack, self.log_defence_weakness)
        ):
            raise ValueError("Joint log strengths must be finite and within centered fit bounds.")


@dataclass(frozen=True, slots=True)
class OpponentPoissonModel:
    model_id: str
    training_dataset_id: str
    training_cutoff: datetime
    config: OpponentPoissonConfig
    log_baseline: float
    home_effect: float
    teams: tuple[OpponentStrength, ...]
    components: tuple[tuple[str, ...], ...]
    iterations: int
    penalized_objective: float
    boundary_parameters: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "training_cutoff", ensure_utc(self.training_cutoff))
        if self.schema_version != 1 or not self.training_dataset_id.startswith("goal_dataset_"):
            raise ValueError("Unsupported joint Poisson schema or training dataset.")
        if not math.isfinite(self.log_baseline) or not -5 <= self.log_baseline <= 3:
            raise ValueError("Baseline log rate is outside fitted bounds.")
        if not math.isfinite(self.home_effect) or not -2 <= self.home_effect <= 2:
            raise ValueError("Home effect is outside fitted bounds.")
        teams = tuple(sorted(self.teams, key=lambda t: t.team_id))
        ids = {t.team_id for t in teams}
        if not teams or len(ids) != len(teams):
            raise ValueError("Joint model needs unique fitted teams.")
        if any(
            abs(math.fsum(getattr(t, field) for t in teams)) > 1e-7
            for field in ("log_attack", "log_defence_weakness")
        ):
            raise ValueError("Joint log strengths must sum to zero.")
        components = tuple(sorted(tuple(sorted(c)) for c in self.components))
        members = [team for c in components for team in c]
        if any(len(c) < 2 for c in components) or set(members) != ids or len(members) != len(ids):
            raise ValueError("Opponent graph components must partition fitted teams.")
        if (
            type(self.iterations) is not int
            or not 0 <= self.iterations <= self.config.max_iterations
        ):
            raise ValueError("Invalid solver iteration audit.")
        if not math.isfinite(self.penalized_objective):
            raise ValueError("Invalid solver objective audit.")
        if type(self.boundary_parameters) is not int or not (
            0 <= self.boundary_parameters <= 2 + 2 * len(teams)
        ):
            raise ValueError("Invalid solver boundary audit.")
        object.__setattr__(self, "teams", teams)
        object.__setattr__(self, "components", components)
        if self.model_id != _digest("opponent_poisson_", self._payload()):
            raise ValueError("Joint Poisson artifact does not match its identity.")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_type": "opponent_adjusted_poisson",
            "target_policy_id": REGULATION_TARGET_POLICY_ID,
            "result_eligibility_policy_id": DEFAULT_RESULT_ELIGIBILITY_POLICY.policy_id,
            "training_dataset_id": self.training_dataset_id,
            "training_cutoff": self.training_cutoff.isoformat(),
            "config": asdict(self.config),
            "log_baseline": self.log_baseline,
            "home_effect": self.home_effect,
            "teams": [asdict(t) for t in self.teams],
            "components": [list(c) for c in self.components],
            "iterations": self.iterations,
            "penalized_objective": self.penalized_objective,
            "boundary_parameters": self.boundary_parameters,
            "solver": "bounded_lbfgsb_analytic_gradient_v1",
        }

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, **self._payload()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OpponentPoissonModel:
        if (
            payload.get("model_type") != "opponent_adjusted_poisson"
            or payload.get("target_policy_id") != REGULATION_TARGET_POLICY_ID
            or payload.get("result_eligibility_policy_id")
            != DEFAULT_RESULT_ELIGIBILITY_POLICY.policy_id
            or payload.get("solver") != "bounded_lbfgsb_analytic_gradient_v1"
        ):
            raise ValueError("Unsupported joint Poisson artifact contract.")
        return cls(
            payload["model_id"],
            payload["training_dataset_id"],
            datetime.fromisoformat(payload["training_cutoff"]),
            OpponentPoissonConfig(**payload["config"]),
            payload["log_baseline"],
            payload["home_effect"],
            tuple(OpponentStrength(**t) for t in payload["teams"]),
            tuple(tuple(c) for c in payload["components"]),
            payload["iterations"],
            payload["penalized_objective"],
            payload["boundary_parameters"],
            payload["schema_version"],
        )

    def predict_match(self, match: Match, *, prediction_time: datetime) -> ScorelineForecast:
        context = PredictionContext(match, prediction_time)
        if context.prediction_time < self.training_cutoff:
            raise ValueError("Model training cutoff is after prediction_time.")
        if match.status in (MatchStatus.CANCELLED, MatchStatus.POSTPONED):
            raise ValueError("Cannot forecast a cancelled or postponed fixture.")
        if type(match.neutral) is not bool:
            raise ValueError("Neutral-site context must be boolean.")
        strengths = {t.team_id: t for t in self.teams}
        if {match.home_team_id, match.away_team_id} - strengths.keys():
            raise ValueError("No fitted goal history for one or both teams.")
        if not any({match.home_team_id, match.away_team_id} <= set(c) for c in self.components):
            raise ValueError("Cannot compare teams in disconnected opponent graphs.")
        home, away = strengths[match.home_team_id], strengths[match.away_team_id]
        home_log = self.log_baseline + home.log_attack + away.log_defence_weakness
        away_log = self.log_baseline + away.log_attack + home.log_defence_weakness
        if not match.neutral:
            home_log += self.home_effect

        def rate(value: float) -> float:
            return min(
                self.config.max_expected_goals, max(self.config.min_expected_goals, math.exp(value))
            )

        return scoreline_forecast(rate(home_log), rate(away_log), max_goals=self.config.max_goals)


def _objective(parameters, attackers, defenders, host, goals, penalty):
    """Summed Poisson NLL (constant log-factorials omitted) plus ridge penalty."""
    n = (len(parameters) - 2) // 2
    attack, defence = parameters[2 : 2 + n], parameters[2 + n :]
    eta = (
        parameters[0]
        + attack[attackers]
        - attack.mean()
        + defence[defenders]
        - defence.mean()
        + parameters[1] * host
    )
    rates = np.exp(eta)
    residual = rates - goals
    objective = float(np.sum(rates - goals * eta) + penalty / 2 * np.sum(parameters[1:] ** 2))
    attack_gradient = np.bincount(attackers, weights=residual, minlength=n)
    defence_gradient = np.bincount(defenders, weights=residual, minlength=n)
    gradient = np.concatenate(
        (
            [residual.sum(), residual @ host + penalty * parameters[1]],
            attack_gradient - attack_gradient.mean() + penalty * attack,
            defence_gradient - defence_gradient.mean() + penalty * defence,
        )
    )
    return objective, gradient


def fit_opponent_poisson(
    records: Sequence[CanonicalMatchRecord],
    *,
    training_cutoff: datetime,
    config: OpponentPoissonConfig = DEFAULT_OPPONENT_POISSON_CONFIG,
) -> OpponentPoissonModel:
    cutoff = ensure_utc(training_cutoff)
    dataset_id = goal_dataset_id(records)
    records = sorted(records, key=lambda r: r.match.match_id)
    if any(
        DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(r).eligible_at > cutoff for r in records
    ):
        raise ValueError("Training result is not available by training_cutoff.")
    ids = sorted({team for r in records for team in (r.match.home_team_id, r.match.away_team_id)})
    index = {team: i for i, team in enumerate(ids)}
    attackers: list[int] = []
    defenders: list[int] = []
    host: list[int] = []
    goals: list[int] = []
    adjacency: dict[str, set[str]] = {team: set() for team in ids}
    appearances = dict.fromkeys(ids, 0)
    for r in records:
        assert r.home_score is not None and r.away_score is not None
        home, away = r.match.home_team_id, r.match.away_team_id
        attackers.extend((index[home], index[away]))
        defenders.extend((index[away], index[home]))
        host.extend((int(not r.match.neutral), 0))
        goals.extend((r.home_score, r.away_score))
        adjacency[home].add(away)
        adjacency[away].add(home)
        appearances[home] += 1
        appearances[away] += 1
    observed = np.asarray(goals, dtype=float)
    if observed.mean() <= 0:
        raise ValueError("Cannot fit Poisson model with zero average goals.")
    n = len(ids)
    initial = np.zeros(2 + 2 * n)
    initial[0] = np.clip(np.log(observed.mean()), -5, 3)
    bounds = [(-5.0, 3.0), (-2.0, 2.0)] + [(-3.0, 3.0)] * (2 * n)
    result = minimize(
        _objective,
        initial,
        args=(
            np.asarray(attackers),
            np.asarray(defenders),
            np.asarray(host),
            observed,
            config.penalty,
        ),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": config.max_iterations,
            "ftol": config.tolerance,
            "gtol": config.tolerance,
        },
    )
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError(f"Joint Poisson optimization did not converge: {result.message}")
    attack = result.x[2 : 2 + n] - result.x[2 : 2 + n].mean()
    defence = result.x[2 + n :] - result.x[2 + n :].mean()
    teams = tuple(
        OpponentStrength(team, appearances[team], float(attack[i]), float(defence[i]))
        for i, team in enumerate(ids)
    )
    components = []
    remaining = set(ids)
    while remaining:
        component, pending = set(), [min(remaining)]
        while pending:
            team = pending.pop()
            if team not in component:
                component.add(team)
                pending.extend(adjacency[team] - component)
        remaining -= component
        components.append(tuple(sorted(component)))
    payload = {
        "schema_version": 1,
        "model_type": "opponent_adjusted_poisson",
        "target_policy_id": REGULATION_TARGET_POLICY_ID,
        "result_eligibility_policy_id": DEFAULT_RESULT_ELIGIBILITY_POLICY.policy_id,
        "training_dataset_id": dataset_id,
        "training_cutoff": cutoff.isoformat(),
        "config": asdict(config),
        "log_baseline": float(result.x[0]),
        "home_effect": float(result.x[1]),
        "teams": [asdict(t) for t in teams],
        "components": [list(c) for c in sorted(components)],
        "iterations": int(result.nit),
        "penalized_objective": float(result.fun),
        "boundary_parameters": sum(
            min(abs(float(value) - lo), abs(float(value) - hi)) < 1e-7
            for value, (lo, hi) in zip(result.x, bounds, strict=True)
        ),
        "solver": "bounded_lbfgsb_analytic_gradient_v1",
    }
    return OpponentPoissonModel.from_dict(
        {
            "model_id": _digest("opponent_poisson_", payload),
            **payload,
        }
    )
