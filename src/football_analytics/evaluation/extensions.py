"""Paired research extensions; explicit coverage and dependence assumptions."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np

from football_analytics.data.contracts import LeakageRisk
from football_analytics.data.normalization import CanonicalMatchRecord
from football_analytics.domain.scores import ScoreBasis
from football_analytics.evaluation.backtest import TemporalBacktestResult
from football_analytics.evaluation.comparison import compare_backtests
from football_analytics.evaluation.metrics import ScoredPrediction, evaluate_predictions
from football_analytics.features.market import MarketSnapshotObservation, devig_decimal_odds
from football_analytics.models.postprocessing import content_id


def paired_block_uncertainty(
    reference: TemporalBacktestResult,
    challenger: TemporalBacktestResult,
    *,
    block_days: int = 30,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    compare_backtests([reference, challenger], reference_backtest_run_id=reference.backtest_run_id)
    if (
        type(block_days) is not int
        or block_days < 1
        or not 100 <= repetitions <= 100000
        or seed < 0
    ):
        raise ValueError(
            "Declare positive block days, 100..100000 repetitions and nonnegative seed."
        )
    left = {p.match_id: p for f in reference.folds for p in f.predictions}
    right = {p.match_id: p for f in challenger.folds for p in f.predictions}
    blocks: dict[int, list[list[float]]] = {}
    names = ("log_loss", "multiclass_brier_score", "ranked_probability_score")
    for match_id, a in sorted(left.items()):
        b = right[match_id]
        ma, mb = (
            evaluate_predictions([ScoredPrediction(p.actual, p.probabilities)]) for p in (a, b)
        )
        t = datetime.fromisoformat(a.prediction_time_iso).astimezone(UTC)
        key = (t.date() - datetime(1970, 1, 1).date()).days // block_days
        blocks.setdefault(key, []).append([getattr(mb, n) - getattr(ma, n) for n in names])
    body: dict[str, Any] = {
        "reference": reference.backtest_run_id,
        "challenger": challenger.backtest_run_id,
        "block_days": block_days,
        "repetitions": repetitions,
        "seed": seed,
        "blocks": len(blocks),
        "paired_matches": len(left),
        "assumption": ("UTC calendar blocks are exchangeable; cross-block dependence is unmodeled"),
        "method": "paired_calendar_block_percentile_bootstrap_v1",
        "confidence_level": 0.95,
    }
    if len(blocks) < 4:
        body.update(status="insufficient_blocks", intervals=None)
    else:
        values = [np.asarray(rows) for _, rows in sorted(blocks.items())]
        sums = np.array([rows.sum(axis=0) for rows in values])
        counts = np.array([len(rows) for rows in values])
        rng = np.random.default_rng(seed)
        estimates = np.empty((repetitions, 3))
        for i in range(repetitions):
            draw = rng.integers(0, len(values), size=len(values))
            estimates[i] = sums[draw].sum(axis=0) / counts[draw].sum()
        lo, hi = np.quantile(estimates, [0.025, 0.975], axis=0)
        point = sums.sum(axis=0) / counts.sum()
        body.update(
            status="estimated",
            intervals={
                n: {"delta": float(point[i]), "lower": float(lo[i]), "upper": float(hi[i])}
                for i, n in enumerate(names)
            },
        )
    return {"comparison_id": content_id("uncertainty_", body), **body}


def market_benchmark(
    backtest: TemporalBacktestResult,
    records: tuple[CanonicalMatchRecord, ...],
    snapshots: list[tuple[MarketSnapshotObservation, ScoreBasis]],
    *,
    timing: str = "prediction_time",
) -> dict[str, Any]:
    if timing not in ("prediction_time", "closing"):
        raise ValueError("Market timing must be prediction_time or closing.")
    source = {r.match.match_id: r for r in records}
    if len(source) != len(records):
        raise ValueError("Duplicate canonical market-benchmark matches.")
    predictions = [p for f in backtest.folds for p in f.predictions]
    if len({p.match_id for p in predictions}) != len(predictions):
        raise ValueError("Repeated market evaluation match.")
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    seen = set()
    for snapshot, basis in snapshots:
        key = (snapshot.match_id, snapshot.market_source_id, snapshot.metadata.available_at)
        if key in seen:
            raise ValueError("Ambiguous duplicate bookmaker snapshot timestamp.")
        seen.add(key)
        if snapshot.metadata.leakage_risk is not LeakageRisk.SAFE:
            raise ValueError("Market benchmark requires pre-match-safe snapshots.")
        if basis is not ScoreBasis.REGULATION_TIME:
            raise ValueError("Market benchmark requires explicit regulation-time settlement.")
    for prediction in predictions:
        if prediction.match_id not in source:
            raise ValueError("Missing canonical market benchmark match.")
        match = source[prediction.match_id].match
        cutoff = datetime.fromisoformat(prediction.prediction_time_iso)
        if timing == "closing":
            if match.kickoff_at is None:
                excluded.append({"match_id": match.match_id, "reason": "unknown_exact_kickoff"})
                continue
            cutoff = match.kickoff_at
        latest: dict[str, MarketSnapshotObservation] = {}
        for snapshot, _basis in snapshots:
            available = snapshot.metadata.available_at
            if (
                snapshot.match_id == match.match_id
                and available is not None
                and available <= cutoff
            ):
                previous = latest.get(snapshot.market_source_id)
                if previous is None or (
                    previous.metadata.available_at is not None
                    and previous.metadata.available_at < available
                ):
                    latest[snapshot.market_source_id] = snapshot
        if not latest:
            excluded.append({"match_id": match.match_id, "reason": "no_eligible_market_snapshot"})
            continue
        fair = [devig_decimal_odds(value).probabilities.as_tuple() for value in latest.values()]
        from football_analytics.domain.probabilities import OutcomeProbabilities

        consensus = OutcomeProbabilities(*map(float, np.mean(fair, axis=0)))
        model_metrics = evaluate_predictions(
            [ScoredPrediction(prediction.actual, prediction.probabilities)]
        )
        market_metrics = evaluate_predictions([ScoredPrediction(prediction.actual, consensus)])
        rows.append(
            {
                "match_id": match.match_id,
                "actual": prediction.actual,
                "model": asdict(prediction.probabilities),
                "market": asdict(consensus),
                "model_log_loss": model_metrics.log_loss,
                "market_log_loss": market_metrics.log_loss,
                "bookmakers": sorted(latest),
                "cutoff": cutoff.isoformat(),
                "source_record_ids": sorted(
                    s.metadata.source_record_id or "unknown" for s in latest.values()
                ),
            }
        )
    body = {
        "backtest_run_id": backtest.backtest_run_id,
        "timing": timing,
        "closing_is_evaluation_only": timing == "closing",
        "total_predictions": len(predictions),
        "matched_predictions": len(rows),
        "coverage": len(rows) / len(predictions) if predictions else 0,
        "rows": rows,
        "excluded": excluded,
        "mean_log_loss_delta_vs_market": sum(
            r["model_log_loss"] - r["market_log_loss"] for r in rows
        )
        / len(rows)
        if rows
        else None,
    }
    return {"benchmark_id": content_id("market_benchmark_", body), **body}
