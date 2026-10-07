from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.cli import main
from football_analytics.data import load_canonical_catalogs, normalize_match_batch
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.domain import Match, MatchStatus
from football_analytics.evaluation import ExpandingWindowPolicy, RollingWindowPolicy
from football_analytics.evaluation.comparison import compare_backtests
from football_analytics.models import logistic_regression_spec
from football_analytics.models.artifacts import load_poisson_model, save_poisson_model
from football_analytics.models.frequency import class_frequency_spec
from football_analytics.models.poisson import (
    PoissonConfig,
    PoissonModel,
    fit_poisson,
    goal_dataset_id,
    poisson_spec,
    scoreline_forecast,
)
from football_analytics.services.research import run_research, run_research_comparison
from wc_forecast.models.poisson import PoissonGoalsModel

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
        "split_policy": ExpandingWindowPolicy("poisson_test_v1", (CUTOFF,), timedelta(days=30), 12),
    }


@pytest.fixture
def records(inputs):
    return normalize_match_batch(
        inputs["observations"],
        team_resolver=inputs["catalogs"].team_resolver(),
        competition_resolver=inputs["catalogs"].competition_resolver(),
    ).normalized


def _fixture(neutral=True):
    return Match("future", date(2020, 7, 1), "ARG", "BRA", "friendly", neutral)


@pytest.mark.parametrize("neutral", [True, False])
@pytest.mark.parametrize("max_goals", [3, 10, 15])
def test_native_expected_goals_and_full_grid_match_legacy(records, neutral, max_goals):
    training = records[:20]
    frame = pd.DataFrame(
        [
            {
                "home_team": row.match.home_team_id,
                "away_team": row.match.away_team_id,
                "home_score": row.home_score,
                "away_score": row.away_score,
            }
            for row in training
        ]
    )
    legacy = PoissonGoalsModel(max_goals=max_goals)
    legacy.fit(frame)
    model = fit_poisson(training, training_cutoff=CUTOFF, config=PoissonConfig(max_goals=max_goals))
    forecast = model.predict_match(_fixture(neutral), prediction_time=CUTOFF)
    expected = legacy.predict_match("ARG", "BRA", neutral=neutral)
    assert forecast.expected_home_goals == pytest.approx(expected.expected_home_goals)
    assert forecast.expected_away_goals == pytest.approx(expected.expected_away_goals)
    assert forecast.probabilities.as_tuple() == pytest.approx(
        (
            expected.prob_home_win,
            expected.prob_draw,
            expected.prob_away_win,
        )
    )
    grid = legacy.scoreline_probabilities(
        expected.expected_home_goals, expected.expected_away_goals
    )
    assert [cell.probability for cell in forecast.scorelines] == pytest.approx(grid.probability)


def test_tail_mass_normalization_entropy_symmetry_and_tie_order():
    forecast = scoreline_forecast(1, 1, max_goals=3)
    retained = (math.exp(-1) * sum(1 / math.factorial(k) for k in range(4))) ** 2
    assert forecast.omitted_tail_probability == pytest.approx(1 - retained)
    assert sum(cell.probability for cell in forecast.scorelines) == pytest.approx(1)
    assert forecast.probabilities.home_win == pytest.approx(forecast.probabilities.away_win)
    assert (forecast.most_likely_home_goals, forecast.most_likely_away_goals) == (0, 0)
    assert 0 <= forecast.outcome_entropy_nats <= math.log(3)
    wider = scoreline_forecast(1, 1, max_goals=10)
    assert wider.omitted_tail_probability < forecast.omitted_tail_probability


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), 0, -1, 101])
def test_invalid_poisson_rates_rejected(rate):
    with pytest.raises(ValueError, match="rates"):
        scoreline_forecast(rate, 1.0)


@pytest.mark.parametrize(
    "config",
    [
        {"max_goals": True},
        {"max_goals": 2},
        {"home_advantage_multiplier": 0},
        {"max_expected_goals": float("nan")},
    ],
)
def test_invalid_model_config_rejected(config):
    with pytest.raises(ValueError):
        PoissonConfig(**config)


def test_fit_rejects_future_results_and_late_publications(records):
    with pytest.raises(ValueError, match="not available"):
        fit_poisson(records, training_cutoff=CUTOFF)
    training = list(records[:20])
    training[0] = replace(
        training[0], metadata=replace(training[0].metadata, available_at=CUTOFF + timedelta(days=1))
    )
    with pytest.raises(ValueError, match="not available"):
        fit_poisson(training, training_cutoff=CUTOFF)


def test_inference_rejects_unseen_teams_and_future_model(records):
    model = fit_poisson(records[:20], training_cutoff=CUTOFF)
    with pytest.raises(ValueError, match="No fitted goal history"):
        model.predict_match(replace(_fixture(), home_team_id="NEW"), prediction_time=CUTOFF)
    with pytest.raises(ValueError, match="training cutoff"):
        model.predict_match(_fixture(), prediction_time=CUTOFF - timedelta(days=1))
    with pytest.raises(ValueError, match="before match_date"):
        model.predict_match(_fixture(), prediction_time=datetime(2020, 7, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="cancelled"):
        model.predict_match(
            replace(_fixture(), status=MatchStatus.CANCELLED), prediction_time=CUTOFF
        )


def test_training_identity_includes_exact_scores_even_when_outcome_is_unchanged(records):
    training = list(records[:20])
    first = fit_poisson(training, training_cutoff=CUTOFF)
    training[0] = replace(training[0], home_score=3)
    second = fit_poisson(training, training_cutoff=CUTOFF)
    assert first.training_dataset_id != second.training_dataset_id
    assert first.model_id != second.model_id
    assert fit_poisson(tuple(reversed(training)), training_cutoff=CUTOFF) == second


def test_invalid_goals_and_duplicate_fixtures_rejected(records):
    with pytest.raises(ValueError, match="non-negative integers"):
        goal_dataset_id([replace(records[0], home_score=-1)])
    with pytest.raises(ValueError, match="duplicate match"):
        goal_dataset_id([records[0], records[0]])
    reversed_fixture = replace(
        records[0],
        match=replace(
            records[0].match,
            match_id="reversed",
            home_team_id=records[0].match.away_team_id,
            away_team_id=records[0].match.home_team_id,
        ),
    )
    with pytest.raises(ValueError, match="reversed fixtures"):
        goal_dataset_id([records[0], reversed_fixture])


def test_zero_goal_dataset_fails_explicitly(records):
    with pytest.raises(ValueError, match="zero average"):
        fit_poisson(
            [replace(row, home_score=0, away_score=0) for row in records[:20]],
            training_cutoff=CUTOFF,
        )


def test_artifact_round_trip_tamper_detection_and_no_overwrite(records, tmp_path):
    model = fit_poisson(records[:20], training_cutoff=CUTOFF)
    path = save_poisson_model(model, tmp_path)
    loaded = load_poisson_model(path)
    assert loaded.predict_match(_fixture(), prediction_time=CUTOFF) == model.predict_match(
        _fixture(),
        prediction_time=CUTOFF,
    )
    assert save_poisson_model(model, tmp_path) == path
    changed = model.to_dict()
    changed["global_goals_per_team_match"] += 0.1
    with pytest.raises(ValueError, match="identity"):
        PoissonModel.from_dict(changed)
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="content conflict"):
        save_poisson_model(model, tmp_path)


def test_poisson_backtest_forecasts_do_not_change_with_future_result(inputs):
    original = run_research(**inputs, model_spec=poisson_spec())
    inputs["observations"][-1] = replace(inputs["observations"][-1], home_score=99)
    changed = run_research(**inputs, model_spec=poisson_spec())
    assert original.backtest == changed.backtest
    assert original.score_models == changed.score_models
    assert original.score_forecasts == changed.score_forecasts
    assert len(original.score_forecasts) == original.backtest.prediction_count == 10


def test_frequency_prior_is_learned_from_training_only_and_supports_missing_classes(inputs):
    for i in range(20):
        inputs["observations"][i] = replace(inputs["observations"][i], home_score=1, away_score=0)
    result = run_research(**inputs, model_spec=class_frequency_spec(smoothing=1.0))
    for prediction in result.backtest.folds[0].predictions:
        assert prediction.probabilities.as_tuple() == pytest.approx((21 / 23, 1 / 23, 1 / 23))


@pytest.mark.parametrize("rolling", [False, True])
def test_paired_comparison_uses_same_train_and_test_samples(inputs, rolling):
    if rolling:
        inputs["split_policy"] = RollingWindowPolicy(
            "rolling_v1", (CUTOFF,), timedelta(days=30), timedelta(days=30), 6
        )
    result = run_research_comparison(
        **inputs,
        model_specs=(
            class_frequency_spec(),
            logistic_regression_spec(),
            poisson_spec(),
        ),
    )
    assert len(result.comparison.rows) == 3
    assert result.comparison.paired_prediction_count == 10
    reference = result.runs[0].backtest
    row = next(row for row in result.comparison.rows if row.model_family == "class_frequency")
    assert row.log_loss_delta_vs_reference == 0
    assert (
        compare_backtests(
            [run.backtest for run in reversed(result.runs)],
            reference_backtest_run_id=reference.backtest_run_id,
        )
        == result.comparison
    )


def test_comparison_rejects_missing_predictions_or_changed_training(inputs):
    result = run_research_comparison(**inputs, model_specs=(class_frequency_spec(), poisson_spec()))
    reference, other = [run.backtest for run in result.runs]
    fold = other.folds[0]
    for changed_fold in (
        replace(fold, predictions=fold.predictions[1:]),
        replace(fold, train_match_ids=fold.train_match_ids[1:]),
    ):
        changed = replace(other, folds=(changed_fold,))
        with pytest.raises(ValueError, match="identical"):
            compare_backtests(
                [reference, changed], reference_backtest_run_id=reference.backtest_run_id
            )


def test_cli_comparison_persists_models_and_forecasts_can_be_replayed(tmp_path, capsys):
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
            "synthetic",
            "--ingested-at",
            "2020-06-01T00:00:00Z",
            "--assert-senior-mens-a",
            "--cutoff",
            CUTOFF.isoformat(),
            "--evaluation-days",
            "30",
            "--min-train",
            "12",
            "--model",
            "all",
            "--output",
            str(tmp_path),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    report = json.loads(Path(output["report"]).read_text())
    assert len(report["runs"]) == 4
    assert len(report["comparison"]["rows"]) == 4
    assert "class_frequency" in Path(output["comparison_report"]).read_text()
    assert len(list((tmp_path / "experiments").glob("*.json"))) == 4
    assert len(output["model_artifacts"]) == 1
    main(
        [
            "predict-poisson",
            "--model-artifact",
            output["model_artifacts"][0],
            "--home-id",
            "ARG",
            "--away-id",
            "BRA",
            "--match-id",
            "future",
            "--competition-id",
            "friendly",
            "--match-date",
            "2020-07-01",
            "--prediction-time",
            "2020-06-30T12:00:00Z",
            "--venue",
            "neutral",
            "--output",
            str(tmp_path / "forecasts"),
        ]
    )
    predicted = json.loads(capsys.readouterr().out)
    forecast = json.loads(Path(predicted["forecast_report"]).read_text())
    assert len(forecast["forecast"]["scorelines"]) == 121
    assert forecast["model_id"] == report["runs"][-1]["score_models"][0]["model_id"]
