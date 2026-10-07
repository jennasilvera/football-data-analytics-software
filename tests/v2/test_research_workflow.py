from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.cli import main
from football_analytics.data import load_canonical_catalogs
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.evaluation import ExpandingWindowPolicy, RollingWindowPolicy
from football_analytics.experiments import JsonExperimentRegistry
from football_analytics.models import logistic_regression_spec
from football_analytics.services.research import ResearchInputError, run_form_research

SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"


def _inputs():
    return {
        "observations": build_legacy_mens_results_observations(
            pd.read_csv(SAMPLE / "results.csv"),
            ingested_at=datetime(2020, 6, 1, tzinfo=UTC),
        ),
        "catalogs": load_canonical_catalogs(
            teams_path=SAMPLE / "teams.json",
            competitions_path=SAMPLE / "competitions.json",
        ),
        "split_policy": ExpandingWindowPolicy(
            "integration_v1", (datetime(2020, 3, 1, tzinfo=UTC),), timedelta(days=30), 12
        ),
        "model_spec": logistic_regression_spec(),
    }


def _args(output: Path) -> list[str]:
    return [
        "research",
        "--results",
        str(SAMPLE / "results.csv"),
        "--teams",
        str(SAMPLE / "teams.json"),
        "--competitions",
        str(SAMPLE / "competitions.json"),
        "--source-id",
        "synthetic-test",
        "--ingested-at",
        "2020-06-01T00:00:00Z",
        "--assert-senior-mens-a",
        "--cutoff",
        "2020-03-01T00:00:00Z",
        "--evaluation-days",
        "30",
        "--min-train",
        "12",
        "--output",
        str(output),
    ]


def test_cli_persists_reproducible_predictions_calibration_and_manifest(tmp_path, capsys):
    main(_args(tmp_path))
    first = json.loads(capsys.readouterr().out)
    main(_args(tmp_path))
    second = json.loads(capsys.readouterr().out)
    assert first == second
    report = json.loads(Path(first["report"]).read_text())
    manifest = JsonExperimentRegistry(tmp_path / "experiments").get(
        report["manifest"]["experiment_id"]
    )
    assert manifest is not None
    assert manifest.prediction_count == 10
    assert report["backtest"]["folds"][0]["train_count"] == 20
    assert len(report["backtest"]["folds"][0]["predictions"]) == 10
    assert report["calibration"]["report_id"] == manifest.calibration_report_id
    assert report["runtime"]["scikit-learn"]
    assert len(list(tmp_path.glob("research_*.json"))) == 1


def test_unknown_entity_stops_research_with_row_audit():
    inputs = _inputs()
    inputs["observations"][0] = replace(inputs["observations"][0], home_team_name="Unknown")
    with pytest.raises(ResearchInputError) as error:
        run_form_research(**inputs)
    assert len(error.value.report.quarantined) == 1


def test_overlapping_evaluation_windows_cannot_double_count_matches():
    inputs = _inputs()
    inputs["split_policy"] = ExpandingWindowPolicy(
        "overlap_v1",
        (datetime(2020, 3, 1, tzinfo=UTC), datetime(2020, 3, 15, tzinfo=UTC)),
        timedelta(days=30),
        12,
    )
    with pytest.raises(ValueError, match="must not overlap"):
        run_form_research(**inputs)


def test_future_results_cannot_change_earlier_forecasts():
    inputs = _inputs()
    baseline = run_form_research(**inputs)
    inputs["observations"][-1] = replace(inputs["observations"][-1], home_score=99, away_score=0)
    changed = run_form_research(**inputs)
    assert baseline.backtest.folds[0].predictions == changed.backtest.folds[0].predictions
    assert baseline.manifest == changed.manifest


def test_rolling_window_limits_training_history():
    inputs = _inputs()
    inputs["split_policy"] = RollingWindowPolicy(
        "rolling_integration_v1",
        (datetime(2020, 3, 1, tzinfo=UTC),),
        timedelta(days=30),
        timedelta(days=30),
        6,
    )
    result = run_form_research(**inputs)
    assert result.backtest.folds[0].train_count == 9
    assert result.backtest.prediction_count == 10


def test_cli_rejects_unknown_entity_without_registering_experiment(tmp_path, capsys):
    frame = pd.read_csv(SAMPLE / "results.csv")
    frame.loc[0, "home_team"] = "Unknown"
    results = tmp_path / "bad.csv"
    frame.to_csv(results, index=False)
    output = tmp_path / "output"
    args = _args(output)
    args[args.index("--results") + 1] = str(results)
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "quarantined" in capsys.readouterr().err
    assert (output / "rejected-input/quarantined_matches.csv").exists()
    assert not (output / "experiments").exists()


def test_cli_requires_explicit_source_scope_assertion(tmp_path):
    args = _args(tmp_path)
    args.remove("--assert-senior-mens-a")
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("module", ["features", "models", "evaluation", "experiments"])
def test_public_packages_import_in_fresh_process(module):
    # Collection order can conceal cross-package cycles in an otherwise green suite.
    result = subprocess.run(
        [sys.executable, "-c", f"import football_analytics.{module}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert importlib.import_module(f"football_analytics.{module}")
