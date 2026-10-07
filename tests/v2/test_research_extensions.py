from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.data import load_canonical_catalogs
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.data.contracts import LeakageRisk, SourceMetadata
from football_analytics.domain.scores import ScoreBasis
from football_analytics.evaluation import ExpandingWindowPolicy
from football_analytics.evaluation.drift import feature_drift
from football_analytics.evaluation.extensions import market_benchmark, paired_block_uncertainty
from football_analytics.features.composition import ManualContextProvider
from football_analytics.features.market import MarketSnapshotObservation
from football_analytics.features.materialization import MaterializedFeatureRow
from football_analytics.models import logistic_regression_spec
from football_analytics.models.frequency import class_frequency_spec
from football_analytics.models.poisson import poisson_spec
from football_analytics.services.ablation import run_feature_ablation
from football_analytics.services.research import run_research, run_research_comparison

SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"


@pytest.fixture
def inputs():
    return dict(
        observations=build_legacy_mens_results_observations(
            pd.read_csv(SAMPLE / "results.csv"), ingested_at=datetime(2020, 6, 1, tzinfo=UTC)
        ),
        catalogs=load_canonical_catalogs(
            teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
        ),
        split_policy=ExpandingWindowPolicy(
            "extensions", (datetime(2020, 3, 1, tzinfo=UTC),), timedelta(days=80), 12
        ),
    )


def test_composed_features_and_ablation_preserve_outer_population(inputs):
    result = run_feature_ablation(
        feature_groups=("form", "schedule", "competition"),
        model_spec=logistic_regression_spec(),
        **inputs,
    )
    assert len(result.runs) == 4
    assert result.removed_groups == ("none", "form", "schedule", "competition")
    assert len({r.backtest.feature_set_id for r in result.runs}) == 4
    assert len({r.backtest.prediction_count for r in result.runs}) == 1
    strengths = run_research(
        feature_groups=("form", "strength", "competition"),
        model_spec=logistic_regression_spec(),
        **inputs,
    )
    assert strengths.backtest.prediction_count == result.runs[0].backtest.prediction_count


def test_block_bootstrap_determinism_and_insufficient_blocks(inputs):
    result = run_research_comparison(model_specs=(class_frequency_spec(), poisson_spec()), **inputs)
    left, right = (r.backtest for r in result.runs)
    short = paired_block_uncertainty(left, right, block_days=90, repetitions=100)
    assert short["status"] == "insufficient_blocks"
    valid = paired_block_uncertainty(left, right, block_days=10, repetitions=100)
    assert valid["status"] == "estimated"
    assert valid == paired_block_uncertainty(left, right, block_days=10, repetitions=100)
    assert all(r["lower"] <= r["upper"] for r in valid["intervals"].values())
    with pytest.raises(ValueError):
        paired_block_uncertainty(left, right, block_days=0)


def test_market_join_coverage_excludes_future_quotes_and_unknown_settlement(inputs):
    run = run_research(model_spec=class_frequency_spec(), **inputs)
    prediction = run.backtest.folds[0].predictions[0]
    cutoff = datetime.fromisoformat(prediction.prediction_time_iso)
    quote = MarketSnapshotObservation(
        prediction.match_id,
        "book",
        2.5,
        3.5,
        2.8,
        SourceMetadata(
            "book",
            cutoff,
            available_at=cutoff,
            source_record_id="quote",
            leakage_risk=LeakageRisk.SAFE,
        ),
    )
    future = replace(
        quote,
        market_source_id="future",
        metadata=replace(
            quote.metadata,
            ingested_at=cutoff + timedelta(days=1),
            available_at=cutoff + timedelta(days=1),
        ),
    )
    report = market_benchmark(
        run.backtest,
        run.normalization.normalized,
        [(quote, ScoreBasis.REGULATION_TIME), (future, ScoreBasis.REGULATION_TIME)],
    )
    assert report["matched_predictions"] == 1
    assert report["rows"][0]["bookmakers"] == ["book"]
    assert len(report["excluded"]) + 1 == report["total_predictions"]
    closing = market_benchmark(
        run.backtest,
        run.normalization.normalized,
        [(quote, ScoreBasis.REGULATION_TIME)],
        timing="closing",
    )
    assert closing["matched_predictions"] == 0
    assert closing["excluded"][0]["reason"] == "unknown_exact_kickoff"
    with pytest.raises(ValueError):
        market_benchmark(run.backtest, run.normalization.normalized, [(quote, ScoreBasis.UNKNOWN)])
    with pytest.raises(ValueError):
        market_benchmark(
            run.backtest, run.normalization.normalized, [(quote, ScoreBasis.REGULATION_TIME)] * 2
        )


def test_drift_uses_reference_boundaries_and_rejects_schema_changes():
    row = MaterializedFeatureRow("m", "2020-01-01T00:00:00Z", "f", "i", (("x", 1.0),), ())
    left = [replace(row, match_id=str(i), columns=(("x", float(i)),)) for i in range(20)]
    same = feature_drift(left, left)
    assert same["features"][0]["psi"] == 0
    right = [replace(r, columns=(("x", r.columns[0][1] + 100),)) for r in left]
    drift = feature_drift(left, right)
    assert drift["features"][0]["psi"] > 0
    assert drift["features"][0]["outside_reference_range"] == 1
    with pytest.raises(ValueError):
        feature_drift(left, [replace(row, feature_set_id="other")])


def test_manual_intelligence_rejects_hindsight_and_invalid_ranges():
    from datetime import date

    from football_analytics.data.serialization import json_safe
    from football_analytics.domain import Match
    from football_analytics.features import PredictionContext

    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    row = {
        "match_id": "fixture",
        "team_id": "A",
        "field": "rotation_risk",
        "value": 0.8,
        "metadata": json_safe(
            SourceMetadata(
                "manual",
                cutoff + timedelta(days=1),
                available_at=cutoff + timedelta(days=1),
                source_record_id="later",
                legal_use_notes="Synthetic",
            )
        ),
    }
    provider = ManualContextProvider([row])
    values = provider.compute(
        PredictionContext(Match("fixture", date(2020, 1, 3), "A", "B", "f", True), cutoff)
    )
    assert all(value.value is None for value in values)
    with pytest.raises(ValueError):
        ManualContextProvider([{**row, "value": 10}])
    with pytest.raises(ValueError):
        ManualContextProvider([row, row])
