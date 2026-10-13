"""Synthetic identifiability, temporal integrity and challenger integration."""

import json
import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_analytics.cli import main
from football_analytics.data import load_canonical_catalogs, normalize_match_batch
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.domain import Match, MatchStatus
from football_analytics.domain.scores import ScoreBasis
from football_analytics.evaluation import ExpandingWindowPolicy, RollingWindowPolicy
from football_analytics.models.artifacts import load_poisson_model, save_poisson_model
from football_analytics.models.opponent_poisson import (
    OpponentPoissonConfig,
    OpponentPoissonModel,
    _objective,
    fit_opponent_poisson,
    opponent_config_from_spec,
    opponent_poisson_spec,
)
from football_analytics.models.poisson import poisson_spec
from football_analytics.services.research import run_research, run_research_comparison

SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"
CUTOFF = datetime(2020, 3, 1, tzinfo=UTC)


@pytest.fixture
def inputs():
    return {
        "observations": build_legacy_mens_results_observations(
            pd.read_csv(SAMPLE / "results.csv"), ingested_at=datetime(2020, 6, 1, tzinfo=UTC)
        ),
        "catalogs": load_canonical_catalogs(
            teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
        ),
        "split_policy": ExpandingWindowPolicy("joint-test", (CUTOFF,), timedelta(days=30), 12),
    }


@pytest.fixture
def records(inputs):
    return normalize_match_batch(
        inputs["observations"],
        team_resolver=inputs["catalogs"].team_resolver(),
        competition_resolver=inputs["catalogs"].competition_resolver(),
    ).normalized


def fixture():
    return Match("future", date(2020, 7, 1), "ARG", "BRA", "friendly", True)


def test_analytic_objective_gradient_matches_finite_differences():
    rng = np.random.default_rng(9)
    parameters = rng.normal(0, 0.2, 10)
    attackers = np.array([0, 1, 0, 3, 2, 1])
    defenders = np.array([1, 0, 3, 0, 1, 2])
    args = (attackers, defenders, np.array([1, 0, 0, 0, 1, 0]), np.array([2, 0, 1, 1, 0, 3]), 5)
    value, gradient = _objective(parameters, *args)
    assert math.isfinite(value)
    for i in range(len(parameters)):
        step = np.zeros(len(parameters))
        step[i] = 1e-6
        difference = (
            _objective(parameters + step, *args)[0] - _objective(parameters - step, *args)[0]
        )
        assert gradient[i] == pytest.approx(difference / 2e-6, abs=1e-7)


def test_joint_fit_recovers_generating_effects_under_unequal_schedules(records):
    rng = np.random.default_rng(47)
    ids = ("A", "B", "C", "D")
    attack = np.array([0.4, -0.2, 0.15, -0.35])
    defence = np.array([-0.4, 0.3, -0.1, 0.2])
    # A plays strong-defence C much more often than weak-defence B.
    pairs = [(0, 2)] * 8 + [(i, j) for i in range(4) for j in range(4) if i != j]
    sample = []
    for i in range(2400):
        home, away = pairs[int(rng.integers(len(pairs)))]
        neutral = bool(rng.integers(2))
        hg = int(rng.poisson(math.exp(0.3 + attack[home] + defence[away] + 0.25 * (not neutral))))
        ag = int(rng.poisson(math.exp(0.3 + attack[away] + defence[home])))
        match = Match(
            str(i),
            date(2000, 1, 1) + timedelta(days=i),
            ids[home],
            ids[away],
            "friendly",
            neutral,
            status=MatchStatus.COMPLETED,
        )
        sample.append(
            replace(records[0], match=match, source_match_id=str(i), home_score=hg, away_score=ag)
        )
    model = fit_opponent_poisson(
        sample, training_cutoff=CUTOFF, config=OpponentPoissonConfig(penalty=1)
    )
    assert model.log_baseline == pytest.approx(0.3, abs=0.08)
    assert model.home_effect == pytest.approx(0.25, abs=0.08)
    assert [t.log_attack for t in model.teams] == pytest.approx(attack, abs=0.10)
    assert [t.log_defence_weakness for t in model.teams] == pytest.approx(defence, abs=0.10)
    assert model.boundary_parameters == 0
    assert model.components == (("A", "B", "C", "D"),)


def test_fit_is_order_independent_and_replays_safe_artifact(records, tmp_path):
    training = records[:20]
    model = fit_opponent_poisson(training, training_cutoff=CUTOFF)
    assert model == fit_opponent_poisson(tuple(reversed(training)), training_cutoff=CUTOFF)
    assert math.fsum(t.log_attack for t in model.teams) == pytest.approx(0, abs=1e-12)
    assert math.fsum(t.log_defence_weakness for t in model.teams) == pytest.approx(0, abs=1e-12)
    path = save_poisson_model(model, tmp_path)
    loaded = load_poisson_model(path)
    assert loaded == model
    forecast = loaded.predict_match(fixture(), prediction_time=CUTOFF)
    assert sum(forecast.probabilities.as_tuple()) == pytest.approx(1)
    swapped = loaded.predict_match(
        replace(fixture(), home_team_id="BRA", away_team_id="ARG"), prediction_time=CUTOFF
    )
    assert forecast.expected_home_goals == pytest.approx(swapped.expected_away_goals)
    assert forecast.probabilities.home_win == pytest.approx(swapped.probabilities.away_win)
    payload = json.loads(path.read_text())
    payload["home_effect"] += 0.01
    with pytest.raises(ValueError, match="identity"):
        OpponentPoissonModel.from_dict(payload)
    payload = model.to_dict()
    payload["solver"] = "unknown"
    with pytest.raises(ValueError, match="contract"):
        OpponentPoissonModel.from_dict(payload)


def test_joint_fit_rejects_bad_targets_and_nonconvergence(records):
    with pytest.raises(ValueError, match="not available"):
        fit_opponent_poisson(records, training_cutoff=CUTOFF)
    late = replace(
        records[0], metadata=replace(records[0].metadata, available_at=CUTOFF + timedelta(days=1))
    )
    with pytest.raises(ValueError, match="not available"):
        fit_opponent_poisson([late, *records[1:20]], training_cutoff=CUTOFF)
    with pytest.raises(ValueError, match="regulation"):
        fit_opponent_poisson(
            [replace(records[0], score_basis=ScoreBasis.UNKNOWN)], training_cutoff=CUTOFF
        )
    with pytest.raises(ValueError, match="duplicate"):
        fit_opponent_poisson([records[0], records[0]], training_cutoff=CUTOFF)
    with pytest.raises(ValueError, match="zero average"):
        fit_opponent_poisson(
            [replace(r, home_score=0, away_score=0) for r in records[:20]], training_cutoff=CUTOFF
        )
    with pytest.raises(ValueError, match="did not converge"):
        fit_opponent_poisson(
            records[:20], training_cutoff=CUTOFF, config=OpponentPoissonConfig(max_iterations=1)
        )


def test_joint_inference_fails_closed_and_rejects_disconnected_teams(records):
    sample = []
    for i, row in enumerate(records[:20]):
        home, away = ("ARG", "BRA") if i % 2 else ("C", "D")
        sample.append(replace(row, match=replace(row.match, home_team_id=home, away_team_id=away)))
    model = fit_opponent_poisson(sample, training_cutoff=CUTOFF)
    assert len(model.components) == 2
    with pytest.raises(ValueError, match="disconnected"):
        model.predict_match(replace(fixture(), away_team_id="C"), prediction_time=CUTOFF)
    with pytest.raises(ValueError, match="No fitted"):
        model.predict_match(replace(fixture(), away_team_id="NEW"), prediction_time=CUTOFF)
    with pytest.raises(ValueError, match="training cutoff"):
        model.predict_match(fixture(), prediction_time=CUTOFF - timedelta(days=1))
    with pytest.raises(ValueError):
        model.predict_match(fixture(), prediction_time=datetime(2020, 7, 2, tzinfo=UTC))
    with pytest.raises(ValueError, match="cancelled"):
        model.predict_match(
            replace(fixture(), status=MatchStatus.CANCELLED), prediction_time=CUTOFF
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"penalty": 0},
        {"penalty": float("nan")},
        {"penalty": True},
        {"max_iterations": True},
        {"max_iterations": 0},
        {"tolerance": 0},
        {"max_goals": 2},
        {"min_expected_goals": 6},
    ],
)
def test_joint_configuration_contract(parameters):
    with pytest.raises(ValueError):
        OpponentPoissonConfig(**parameters)


def test_temporal_comparison_uses_same_matches_and_untouched_future_labels(inputs):
    compared = run_research_comparison(
        **inputs, model_specs=(poisson_spec(), opponent_poisson_spec())
    )
    left, right = compared.runs
    assert left.manifest.feature_set_id == right.manifest.feature_set_id
    assert left.manifest.imputation_policy_id == right.manifest.imputation_policy_id
    assert [p.match_id for p in left.backtest.folds[0].predictions] == [
        p.match_id for p in right.backtest.folds[0].predictions
    ]
    altered = [
        replace(row, home_score=(row.home_score or 0) + 5)
        if row.match_date >= CUTOFF.date()
        else row
        for row in inputs["observations"]
    ]
    changed = run_research(
        **{**inputs, "observations": altered}, model_spec=opponent_poisson_spec()
    )
    assert changed.score_models == right.score_models
    assert [r.forecast for r in changed.score_forecasts] == [
        r.forecast for r in right.score_forecasts
    ]
    assert changed.backtest.aggregate_metrics != right.backtest.aggregate_metrics
    rolling = run_research(
        **{
            **inputs,
            "split_policy": RollingWindowPolicy(
                "joint-rolling", (CUTOFF,), timedelta(days=30), timedelta(days=50), 12
            ),
        },
        model_spec=opponent_poisson_spec(),
    )
    assert rolling.backtest.prediction_count == right.backtest.prediction_count
    assert rolling.score_models[0].training_dataset_id != right.score_models[0].training_dataset_id
    assert opponent_config_from_spec(opponent_poisson_spec()) == OpponentPoissonConfig()


def test_cli_comparison_exports_and_replays_joint_model(tmp_path, capsys):
    main(
        [
            "research",
            "--results",
            str(SAMPLE / "results.csv"),
            "--teams",
            str(SAMPLE / "teams.json"),
            "--competitions",
            str(SAMPLE / "competitions.json"),
            "--source-id",
            "synthetic-joint-test",
            "--assert-senior-mens-a",
            "--ingested-at",
            "2020-06-01T00:00:00Z",
            "--cutoff",
            CUTOFF.isoformat(),
            "--evaluation-days",
            "30",
            "--min-train",
            "12",
            "--model",
            "poisson-comparison",
            "--paired-uncertainty",
            "--output",
            str(tmp_path),
        ]
    )
    paths = json.loads(capsys.readouterr().out)
    report = json.loads(Path(paths["report"]).read_text())
    assert len(report["runs"]) == 2
    assert report["model_spec"]["spec_id"] == poisson_spec().spec_id
    assert report["runs"][1]["backtest"]["model_spec_id"] == opponent_poisson_spec().spec_id
    assert report["paired_uncertainty"][0]["paired_matches"] > 0
    model_path = next(tmp_path.rglob("opponent_poisson_*.json"))
    main(
        [
            "predict-poisson",
            "--model-artifact",
            str(model_path),
            "--match-id",
            "future",
            "--home-id",
            "ARG",
            "--away-id",
            "BRA",
            "--competition-id",
            "friendly",
            "--match-date",
            "2020-07-01",
            "--prediction-time",
            CUTOFF.isoformat(),
            "--venue",
            "neutral",
            "--output",
            str(tmp_path / "prediction"),
        ]
    )
    prediction = json.loads(capsys.readouterr().out)
    assert prediction["model_id"].startswith("opponent_poisson_")


def test_ridge_shrinks_effects_and_neutral_only_training_has_no_home_effect(records):
    sample = [replace(r, match=replace(r.match, neutral=True)) for r in records[:20]]
    weak = fit_opponent_poisson(
        sample, training_cutoff=CUTOFF, config=OpponentPoissonConfig(penalty=0.1)
    )
    strong = fit_opponent_poisson(
        sample, training_cutoff=CUTOFF, config=OpponentPoissonConfig(penalty=1000)
    )
    assert weak.home_effect == 0
    assert strong.home_effect == 0
    assert sum(t.log_attack**2 + t.log_defence_weakness**2 for t in strong.teams) < sum(
        t.log_attack**2 + t.log_defence_weakness**2 for t in weak.teams
    )
    assert weak.model_id != strong.model_id
