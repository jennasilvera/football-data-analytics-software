"""Descriptive diagnostics of held-out forecasts, with strict canonical joins."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime

from football_analytics.data.contracts import ensure_utc
from football_analytics.data.normalization import CanonicalMatchRecord
from football_analytics.domain import MatchOutcome
from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.domain.scores import REGULATION_TARGET_POLICY_ID, require_regulation_score
from football_analytics.evaluation.backtest import TemporalBacktestResult
from football_analytics.evaluation.metrics import (
    EvaluationMetrics,
    ScoredPrediction,
    evaluate_predictions,
)
from football_analytics.features.base import PredictionContext
from football_analytics.features.history import DEFAULT_RESULT_ELIGIBILITY_POLICY


@dataclass(frozen=True, slots=True)
class MatchDiagnostic:
    match_id: str
    fold_id: str
    model_id: str
    prediction_time_iso: str
    match_date: str
    competition_id: str
    home_team_id: str
    away_team_id: str
    neutral: bool
    home_score: int
    away_score: int
    actual: MatchOutcome
    predicted: MatchOutcome
    probabilities: OutcomeProbabilities
    actual_probability: float
    correct: bool
    log_loss: float
    brier_score: float
    ranked_probability_score: float
    entropy_nats: float


@dataclass(frozen=True, slots=True)
class EvaluationSlice:
    dimension: str
    key: str
    match_ids: tuple[str, ...]
    metrics: EvaluationMetrics
    home_wins: int
    draws: int
    away_wins: int
    below_minimum: bool


@dataclass(frozen=True, slots=True)
class EvaluationDiagnostics:
    diagnostic_id: str
    backtest_run_id: str
    model_spec_id: str
    min_sample: int
    metrics: EvaluationMetrics
    matches: tuple[MatchDiagnostic, ...]
    slices: tuple[EvaluationSlice, ...]
    schema_version: int = 1
    target_policy_id: str = REGULATION_TARGET_POLICY_ID


def build_evaluation_diagnostics(
    backtest: TemporalBacktestResult,
    records: Sequence[CanonicalMatchRecord],
    *,
    min_sample: int = 30,
) -> EvaluationDiagnostics:
    """Recompute metrics from predictions; never silently intersect join samples.

    Competition, venue and year partition matches. Team slices overlap: each
    match belongs to both participants, retaining the home/draw/away convention.
    """
    if type(min_sample) is not int or min_sample < 1:
        raise ValueError("Diagnostic minimum sample must be a positive integer.")
    if backtest.target_policy_id != REGULATION_TARGET_POLICY_ID:
        raise ValueError("Diagnostics require regulation-time outcomes.")
    by_id = {record.match.match_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate canonical match IDs in diagnostic source.")
    matches = []
    seen: set[str] = set()
    for fold in backtest.folds:
        if not fold.train_match_ids:
            raise ValueError("Diagnostics require explicit training match identities.")
        for prediction in fold.predictions:
            if prediction.match_id in seen:
                raise ValueError("Diagnostics cannot count repeated evaluation matches.")
            seen.add(prediction.match_id)
            if prediction.match_id in fold.train_match_ids:
                raise ValueError("Evaluation match appears in its own training fold.")
            if prediction.fold_id != fold.fold_id or prediction.model_id != fold.model_id:
                raise ValueError("Prediction model/fold identity does not match its container.")
            record = by_id.get(prediction.match_id)
            if record is None:
                raise ValueError(f"Missing canonical diagnostic match: {prediction.match_id}")
            require_regulation_score(record.score_basis, match_id=prediction.match_id)
            eligible = DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(record)
            context = PredictionContext(
                record.match, datetime.fromisoformat(prediction.prediction_time_iso)
            )
            cutoff = ensure_utc(datetime.fromisoformat(fold.cutoff_iso), "fold cutoff")
            if context.prediction_time < cutoff:
                raise ValueError("Evaluation prediction precedes its fold cutoff.")
            if eligible.eligible_at <= context.prediction_time:
                raise ValueError("Evaluation target was already available at prediction time.")
            home, away = record.home_score, record.away_score
            assert home is not None and away is not None
            if any(type(score) is not int or score < 0 for score in (home, away)):
                raise ValueError("Diagnostic goals must be non-negative integers.")
            if type(record.match.neutral) is not bool:
                raise ValueError("Diagnostic venue neutrality must be boolean.")
            actual = (
                MatchOutcome.HOME_WIN
                if home > away
                else MatchOutcome.AWAY_WIN
                if home < away
                else MatchOutcome.DRAW
            )
            if actual is not prediction.actual:
                raise ValueError("Evaluation target conflicts with the canonical result.")
            probabilities = prediction.probabilities
            metrics = evaluate_predictions([ScoredPrediction(actual, probabilities)])
            matches.append(
                MatchDiagnostic(
                    prediction.match_id,
                    prediction.fold_id,
                    prediction.model_id,
                    context.prediction_time.isoformat(),
                    record.match.match_date.isoformat(),
                    record.match.competition_id,
                    record.match.home_team_id,
                    record.match.away_team_id,
                    record.match.neutral,
                    home,
                    away,
                    actual,
                    probabilities.predicted_outcome,
                    probabilities,
                    probabilities.probability_for(actual),
                    actual is probabilities.predicted_outcome,
                    metrics.log_loss,
                    metrics.multiclass_brier_score,
                    metrics.ranked_probability_score,
                    -sum(p * math.log(p) for p in probabilities.as_tuple() if p > 0),
                )
            )
    if not matches:
        raise ValueError("Diagnostics require held-out predictions.")
    matches.sort(key=lambda row: (row.prediction_time_iso, row.match_id))
    groups: dict[tuple[str, str], list[MatchDiagnostic]] = {}
    for row in matches:
        keys = (
            ("competition", row.competition_id),
            ("year", row.match_date[:4]),
            ("venue", "neutral" if row.neutral else "home"),
            ("team", row.home_team_id),
            ("team", row.away_team_id),
        )
        for key in keys:
            groups.setdefault(key, []).append(row)
    slices = tuple(
        EvaluationSlice(
            dimension,
            key,
            tuple(row.match_id for row in rows),
            _metrics(rows),
            sum(row.actual is MatchOutcome.HOME_WIN for row in rows),
            sum(row.actual is MatchOutcome.DRAW for row in rows),
            sum(row.actual is MatchOutcome.AWAY_WIN for row in rows),
            len(rows) < min_sample,
        )
        for (dimension, key), rows in sorted(groups.items())
    )
    metrics = _metrics(matches)
    payload = {
        "schema_version": 1,
        "target_policy_id": REGULATION_TARGET_POLICY_ID,
        "backtest_run_id": backtest.backtest_run_id,
        "model_spec_id": backtest.model_spec_id,
        "min_sample": min_sample,
        "metrics": asdict(metrics),
        "matches": [asdict(row) for row in matches],
        "slices": [asdict(row) for row in slices],
    }
    identity = (
        "diagnostics_"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )
    return EvaluationDiagnostics(
        identity,
        backtest.backtest_run_id,
        backtest.model_spec_id,
        min_sample,
        metrics,
        tuple(matches),
        slices,
    )


def _metrics(rows: Sequence[MatchDiagnostic]) -> EvaluationMetrics:
    return evaluate_predictions([ScoredPrediction(row.actual, row.probabilities) for row in rows])
