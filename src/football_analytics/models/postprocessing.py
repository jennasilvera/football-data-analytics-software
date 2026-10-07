"""Learn probability transforms from explicitly held-out, available results."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from football_analytics.data.contracts import ensure_utc
from football_analytics.domain import MatchOutcome
from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.domain.scores import REGULATION_TARGET_POLICY_ID

EPSILON = 1e-12
OUTCOMES = (MatchOutcome.HOME_WIN, MatchOutcome.DRAW, MatchOutcome.AWAY_WIN)


def content_id(prefix: str, payload: object) -> str:
    return (
        prefix
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


@dataclass(frozen=True, slots=True)
class BaseFitEvidence:
    spec_id: str
    model_id: str
    fitted_at: datetime
    train_match_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fitted_at", ensure_utc(self.fitted_at, "fitted_at"))
        if not self.spec_id.strip() or not self.model_id.strip() or not self.train_match_ids:
            raise ValueError("Base model evidence requires spec/model IDs and training matches.")
        if len(set(self.train_match_ids)) != len(self.train_match_ids):
            raise ValueError("Base training matches must be unique.")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "fitted_at": self.fitted_at.isoformat()}


@dataclass(frozen=True, slots=True)
class HeldOutPrediction:
    match_id: str
    prediction_time: datetime
    target_available_at: datetime
    actual: MatchOutcome
    probabilities: tuple[OutcomeProbabilities, ...]

    def __post_init__(self) -> None:
        for field in ("prediction_time", "target_available_at"):
            object.__setattr__(self, field, ensure_utc(getattr(self, field), field))
        if not self.match_id.strip() or not self.probabilities:
            raise ValueError("Held-out predictions require match ID and probabilities.")
        if self.target_available_at <= self.prediction_time:
            raise ValueError("Target must become available after prediction time.")
        if not isinstance(self.actual, MatchOutcome):
            raise TypeError("actual must be a MatchOutcome.")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "prediction_time": self.prediction_time.isoformat(),
            "target_available_at": self.target_available_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ProbabilityTransform:
    """Portable transform plus the exact fitting sample and base-model evidence."""

    model_id: str
    method: Literal["temperature", "convex_ensemble"]
    fitted_at: datetime
    members: tuple[BaseFitEvidence, ...]
    examples: tuple[HeldOutPrediction, ...]
    min_examples: int
    temperature: float = 1.0
    weights: tuple[float, ...] = ()
    regularization: float = 0.01

    def __post_init__(self) -> None:
        object.__setattr__(self, "fitted_at", ensure_utc(self.fitted_at, "fitted_at"))
        _validate_sample(self.members, self.examples, self.fitted_at, self.min_examples)
        if self.method not in ("temperature", "convex_ensemble"):
            raise ValueError("Unsupported probability transform.")
        if not math.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and non-negative.")
        if not math.isfinite(self.temperature) or not 0.25 <= self.temperature <= 4:
            raise ValueError("Temperature must lie in [0.25, 4].")
        if self.method == "temperature":
            if len(self.members) != 1 or self.weights:
                raise ValueError("Temperature scaling requires exactly one member and no weights.")
        elif (
            len(self.members) < 2
            or len(self.weights) != len(self.members)
            or any(not math.isfinite(w) or w < 0 for w in self.weights)
            or not math.isclose(sum(self.weights), 1.0, abs_tol=1e-9)
            or self.temperature != 1.0
        ):
            raise ValueError("Ensemble requires at least two members and simplex weights.")
        if self.model_id != content_id("probability_transform_", self._payload()):
            raise ValueError("Probability transform content does not match its identity.")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "target_policy_id": REGULATION_TARGET_POLICY_ID,
            "method": self.method,
            "fitted_at": self.fitted_at.isoformat(),
            "members": [m.to_dict() for m in self.members],
            "examples": [e.to_dict() for e in self.examples],
            "min_examples": self.min_examples,
            "temperature": self.temperature,
            "weights": self.weights,
            "regularization": self.regularization,
            "epsilon": EPSILON,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, **self._payload()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProbabilityTransform:
        if (
            payload.get("schema_version") != 1
            or payload.get("target_policy_id") != REGULATION_TARGET_POLICY_ID
            or payload.get("epsilon") != EPSILON
        ):
            raise ValueError("Unsupported probability transform schema or policy.")
        return cls(
            model_id=payload["model_id"],
            method=payload["method"],
            fitted_at=datetime.fromisoformat(payload["fitted_at"]),
            members=tuple(
                BaseFitEvidence(
                    m["spec_id"],
                    m["model_id"],
                    datetime.fromisoformat(m["fitted_at"]),
                    tuple(m["train_match_ids"]),
                )
                for m in payload["members"]
            ),
            examples=tuple(
                HeldOutPrediction(
                    e["match_id"],
                    datetime.fromisoformat(e["prediction_time"]),
                    datetime.fromisoformat(e["target_available_at"]),
                    MatchOutcome(e["actual"]),
                    tuple(OutcomeProbabilities(**p) for p in e["probabilities"]),
                )
                for e in payload["examples"]
            ),
            min_examples=payload["min_examples"],
            temperature=payload["temperature"],
            weights=tuple(payload["weights"]),
            regularization=payload["regularization"],
        )

    def predict(
        self,
        probabilities: dict[str, OutcomeProbabilities],
        *,
        prediction_time: datetime,
    ) -> OutcomeProbabilities:
        if ensure_utc(prediction_time, "prediction_time") < self.fitted_at:
            raise ValueError("Transform fitting cutoff is after prediction time.")
        if set(probabilities) != {m.spec_id for m in self.members}:
            raise ValueError("Prediction member specs must exactly match the fitted transform.")
        values = np.array([probabilities[m.spec_id].as_tuple() for m in self.members])
        if self.method == "temperature":
            result = _scaled(values[0], self.temperature)
        else:
            result = np.asarray(self.weights) @ values
            result /= result.sum()
        return OutcomeProbabilities(*map(float, result))


def _validate_sample(
    members: tuple[BaseFitEvidence, ...],
    examples: tuple[HeldOutPrediction, ...],
    fitted_at: datetime,
    min_examples: int,
) -> None:
    if type(min_examples) is not int or min_examples < 1 or len(examples) < min_examples:
        raise ValueError("Insufficient held-out examples for declared minimum.")
    if not members or len({m.spec_id for m in members}) != len(members):
        raise ValueError("Base member spec IDs must be unique.")
    if len({e.match_id for e in examples}) != len(examples):
        raise ValueError("Held-out match IDs must be unique.")
    if {e.actual for e in examples} != set(OUTCOMES):
        raise ValueError("Probability fitting requires all three outcome classes.")
    for example in examples:
        if len(example.probabilities) != len(members):
            raise ValueError("Probability count does not match base members.")
        if example.prediction_time >= fitted_at or example.target_available_at > fitted_at:
            raise ValueError("Held-out results must be available before transform evaluation.")
        for member in members:
            if member.fitted_at > example.prediction_time:
                raise ValueError("Base model was fitted after held-out prediction time.")
            if example.match_id in member.train_match_ids:
                raise ValueError("Held-out match appears in base model training data.")


def _scaled(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.maximum(values, EPSILON)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return np.asarray(exp / exp.sum(axis=-1, keepdims=True))


def fit_probability_transform(
    *,
    method: Literal["temperature", "convex_ensemble"],
    members: tuple[BaseFitEvidence, ...],
    examples: tuple[HeldOutPrediction, ...],
    fitted_at: datetime,
    min_examples: int = 30,
    regularization: float = 0.01,
) -> ProbabilityTransform:
    fitted_at = ensure_utc(fitted_at, "fitted_at")
    examples = tuple(sorted(examples, key=lambda e: (e.prediction_time, e.match_id)))
    _validate_sample(members, examples, fitted_at, min_examples)
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and non-negative.")
    values = np.array([[p.as_tuple() for p in e.probabilities] for e in examples])
    targets = np.array([OUTCOMES.index(e.actual) for e in examples])
    indices = np.arange(len(examples))
    temperature = 1.0
    weights: tuple[float, ...] = ()
    if method == "temperature":
        if len(members) != 1:
            raise ValueError("Temperature scaling requires exactly one member.")

        def loss(t: float) -> float:
            return float(-np.log(_scaled(values[:, 0], t)[indices, targets]).mean())

        fitted = minimize_scalar(loss, bounds=(0.25, 4.0), method="bounded")
        if not fitted.success or not math.isfinite(fitted.fun):
            raise ValueError("Temperature optimization failed.")
        temperature = min((1.0, 0.25, 4.0, float(fitted.x)), key=loss)
    elif method == "convex_ensemble":
        if len(members) < 2:
            raise ValueError("Ensemble requires at least two members.")
        uniform = np.full(len(members), 1 / len(members))
        actual_probs = values[indices, :, targets]

        def objective(w: np.ndarray) -> float:
            return float(
                -np.log(np.maximum(actual_probs @ w, EPSILON)).mean()
                + regularization * np.square(w - uniform).sum()
            )

        fitted = minimize(
            objective,
            uniform,
            method="SLSQP",
            bounds=[(0, 1)] * len(members),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not fitted.success or not np.isfinite(fitted.x).all():
            raise ValueError("Ensemble optimization failed.")
        solution = np.clip(fitted.x, 0, 1)
        if solution.sum() <= 0 or abs(float(fitted.x.sum()) - 1) > 1e-6:
            raise ValueError("Ensemble optimizer returned invalid simplex weights.")
        solution /= solution.sum()
        if objective(solution) > objective(uniform) + 1e-9:
            raise ValueError("Ensemble optimizer failed to improve its initialization.")
        weights = tuple(map(float, solution))
    else:
        raise ValueError("Unsupported probability transform.")
    payload = {
        "schema_version": 1,
        "target_policy_id": REGULATION_TARGET_POLICY_ID,
        "method": method,
        "fitted_at": fitted_at.isoformat(),
        "members": [m.to_dict() for m in members],
        "examples": [e.to_dict() for e in examples],
        "min_examples": min_examples,
        "temperature": temperature,
        "weights": weights,
        "regularization": regularization,
        "epsilon": EPSILON,
    }
    return ProbabilityTransform(
        content_id("probability_transform_", payload),
        method,
        fitted_at,
        members,
        examples,
        min_examples,
        temperature,
        weights,
        regularization,
    )
