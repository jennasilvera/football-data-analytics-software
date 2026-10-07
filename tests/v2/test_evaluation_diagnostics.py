import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.cli import main
from football_analytics.data import load_canonical_catalogs
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.domain import MatchOutcome
from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.domain.scores import ScoreBasis
from football_analytics.evaluation import ExpandingWindowPolicy
from football_analytics.evaluation.diagnostics import build_evaluation_diagnostics
from football_analytics.models.frequency import class_frequency_spec
from football_analytics.reports.diagnostics import render_evaluation_diagnostics
from football_analytics.services.research import run_research

SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"


@pytest.fixture
def research():
    return run_research(
        observations=build_legacy_mens_results_observations(
            pd.read_csv(SAMPLE / "results.csv"), ingested_at=datetime(2020, 6, 1, tzinfo=UTC)
        ),
        catalogs=load_canonical_catalogs(
            teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
        ),
        split_policy=ExpandingWindowPolicy(
            "diagnostics", (datetime(2020, 3, 1, tzinfo=UTC),), timedelta(days=30), 12
        ),
        model_spec=class_frequency_spec(),
    )


def test_partition_metrics_reconstruct_aggregate_and_team_slices_overlap(research):
    report = build_evaluation_diagnostics(research.backtest, research.normalization.normalized)
    assert report.metrics == research.backtest.aggregate_metrics
    assert len(report.matches) == 10
    assert all(row.below_minimum for row in report.slices)
    for dimension in ("competition", "year", "venue"):
        rows = [row for row in report.slices if row.dimension == dimension]
        assert sum(row.metrics.n_predictions for row in rows) == 10
        assert sum(
            row.metrics.log_loss * row.metrics.n_predictions for row in rows
        ) / 10 == pytest.approx(report.metrics.log_loss)
    teams = [row for row in report.slices if row.dimension == "team"]
    assert sum(row.metrics.n_predictions for row in teams) == 20
    assert all(
        row.home_wins + row.draws + row.away_wins == row.metrics.n_predictions
        for row in report.slices
    )
    # Source order is not part of diagnostic identity.
    assert report == build_evaluation_diagnostics(
        research.backtest, tuple(reversed(research.normalization.normalized))
    )


def test_individual_losses_and_zero_probability_are_finite(research):
    fold = research.backtest.folds[0]
    first = fold.predictions[0]
    wrong = (
        OutcomeProbabilities(0, 1, 0)
        if first.actual is not MatchOutcome.DRAW
        else OutcomeProbabilities(1, 0, 0)
    )
    changed = replace(
        research.backtest,
        folds=(
            replace(fold, predictions=(replace(first, probabilities=wrong), *fold.predictions[1:])),
        ),
    )
    report = build_evaluation_diagnostics(changed, research.normalization.normalized)
    row = next(row for row in report.matches if row.match_id == first.match_id)
    assert row.actual_probability == 0
    assert row.log_loss == pytest.approx(-math.log(1e-15))
    assert row.brier_score == pytest.approx(2)
    assert row.entropy_nats == 0
    assert not row.correct
    assert report.metrics != research.backtest.aggregate_metrics


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "duplicate_source",
        "duplicate_forecast",
        "in_sample",
        "wrong_outcome",
        "wrong_model",
        "wrong_fold",
        "unknown_period",
        "negative_goal",
        "late_prediction",
    ],
)
def test_invalid_prediction_source_joins_fail_closed(research, failure):
    run = research.backtest
    fold = run.folds[0]
    first = fold.predictions[0]
    records = list(research.normalization.normalized)
    i = next(i for i, record in enumerate(records) if record.match.match_id == first.match_id)
    if failure == "missing":
        records.pop(i)
    elif failure == "duplicate_source":
        records.append(records[i])
    elif failure == "duplicate_forecast":
        fold = replace(fold, predictions=(*fold.predictions, first))
    elif failure == "in_sample":
        fold = replace(fold, train_match_ids=(*fold.train_match_ids, first.match_id))
    elif failure == "wrong_outcome":
        actual = (
            MatchOutcome.DRAW if first.actual is not MatchOutcome.DRAW else MatchOutcome.HOME_WIN
        )
        fold = replace(fold, predictions=(replace(first, actual=actual), *fold.predictions[1:]))
    elif failure in ("wrong_model", "wrong_fold"):
        fields = {"model_id" if failure == "wrong_model" else "fold_id": "incorrect"}
        fold = replace(fold, predictions=(replace(first, **fields), *fold.predictions[1:]))
    elif failure == "unknown_period":
        records[i] = replace(records[i], score_basis=ScoreBasis.UNKNOWN)
    elif failure == "negative_goal":
        records[i] = replace(records[i], home_score=-1)
    else:
        fold = replace(
            fold,
            predictions=(
                replace(first, prediction_time_iso="2030-01-01T00:00:00Z"),
                *fold.predictions[1:],
            ),
        )
    with pytest.raises(ValueError):
        build_evaluation_diagnostics(replace(run, folds=(fold,)), records)


def test_threshold_changes_flags_and_identity_without_hiding_metrics(research):
    small = build_evaluation_diagnostics(research.backtest, research.normalization.normalized)
    all_visible = build_evaluation_diagnostics(
        research.backtest, research.normalization.normalized, min_sample=1
    )
    assert small.diagnostic_id != all_visible.diagnostic_id
    assert small.metrics == all_visible.metrics
    assert small.matches == all_visible.matches
    assert not any(row.below_minimum for row in all_visible.slices)
    for invalid in (0, -1, True):
        with pytest.raises(ValueError):
            build_evaluation_diagnostics(
                research.backtest, research.normalization.normalized, min_sample=invalid
            )


def test_render_explains_overlap_and_includes_stable_match_identifiers(research):
    report = build_evaluation_diagnostics(research.backtest, research.normalization.normalized)
    markdown = render_evaluation_diagnostics([report])
    assert "Team slices overlap" in markdown
    assert "Small sample" in markdown
    assert "not that team's win rate" in markdown
    assert report.matches[0].match_id in markdown
    assert report.diagnostic_id in markdown


def test_cli_persists_diagnostics_alongside_research(tmp_path, capsys):
    args = [
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
        "2020-03-01T00:00:00Z",
        "--evaluation-days",
        "30",
        "--min-train",
        "12",
        "--model",
        "class-frequency",
        "--output",
        str(tmp_path),
    ]
    main(args)
    first = json.loads(capsys.readouterr().out)
    main(args)
    assert json.loads(capsys.readouterr().out) == first
    payload = json.loads(Path(first["report"]).read_text())
    assert payload["schema_version"] == 5
    assert len(payload["diagnostics"]) == 1
    assert payload["diagnostics"][0]["metrics"] == payload["backtest"]["aggregate_metrics"]
    assert (
        Path(first["diagnostics_report"]).read_text().startswith("# Held-out forecast diagnostics")
    )


def test_multiple_competition_and_venue_groups_reconcile_without_dropping_rows(research):
    records = tuple(
        replace(
            record,
            match=replace(
                record.match,
                competition_id="competition_a" if i % 2 else "competition_b",
                neutral=bool(i % 3),
            ),
        )
        for i, record in enumerate(research.normalization.normalized)
    )
    report = build_evaluation_diagnostics(research.backtest, records)
    for dimension in ("competition", "venue"):
        groups = [row for row in report.slices if row.dimension == dimension]
        assert len(groups) == 2
        ids = [match_id for group in groups for match_id in group.match_ids]
        assert len(ids) == len(set(ids)) == report.metrics.n_predictions
        for metric in (
            "log_loss",
            "multiclass_brier_score",
            "ranked_probability_score",
            "accuracy",
        ):
            weighted = (
                sum(getattr(row.metrics, metric) * row.metrics.n_predictions for row in groups)
                / report.metrics.n_predictions
            )
            assert weighted == pytest.approx(getattr(report.metrics, metric))
