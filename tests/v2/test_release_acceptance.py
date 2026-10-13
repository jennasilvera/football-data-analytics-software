"""Evidence tampering, approval lifecycle and dashboard navigation acceptance."""

import copy
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.data import load_canonical_catalogs
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.operations import execute, parser
from football_analytics.services.forecasting import train_forecast_bundle
from football_analytics.services.governance import require_approved_model, validate_research
from football_analytics.storage.repository import Repository

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/sample/v2"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    work = tmp_path_factory.mktemp("evidence")
    (work / "odds.json").write_text("[]")
    output = subprocess.check_output([
        sys.executable, "-m", "football_analytics", "research",
        "--results", str(SAMPLE / "results.csv"),
        "--teams", str(SAMPLE / "teams.json"),
        "--competitions", str(SAMPLE / "competitions.json"),
        "--source-id", "synthetic", "--assert-senior-mens-a",
        "--ingested-at", "2020-06-01T00:00:00Z", "--cutoff", "2020-03-01T00:00:00Z",
        "--evaluation-days", "30", "--min-train", "12", "--model", "all",
        "--postprocess-days", "21", "--min-postprocess", "5", "--output", str(work),
        "--market-snapshots", str(work / "odds.json"), "--market-timing", "all",
    ], text=True)
    report = json.loads(Path(json.loads(output)["report"]).read_text())
    observations = build_legacy_mens_results_observations(
        pd.read_csv(SAMPLE / "results.csv"), ingested_at=datetime(2020, 6, 1, tzinfo=UTC),
        source_id="synthetic",
        source_version=hashlib.sha256((SAMPLE / "results.csv").read_bytes()).hexdigest(),
    )
    catalogs = load_canonical_catalogs(
        teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
    )
    bundle = train_forecast_bundle(
        observations, catalogs, cutoff=datetime(2020, 5, 15, tzinfo=UTC), min_calibration=5,
    )
    return report, bundle


def test_valid_research_reproduces_metrics_and_covers_bundle(evidence):
    validate_research(*evidence)
    report, _ = evidence
    assert len(report["market_benchmarks"]) == 3 * len(report["runs"])
    assert {r["timing"] for r in report["market_benchmarks"]} == {
        "prediction_time", "opening", "closing"
    }
    assert all(r["paired_bookmaker_matches"] == 0 for r in report["market_timing_comparisons"])


@pytest.mark.parametrize("tamper", [
    "probabilities", "actual", "fold_metrics", "calibration", "lineage", "cutoff",
    "training_overlap", "missing_models", "missing_transform", "catalogs", "feature_contract",
    "source", "context",
])
def test_inconsistent_research_is_rejected(evidence, tamper):
    original, bundle = evidence
    report = copy.deepcopy(original)
    run = report["runs"][0]
    fold = run["backtest"]["folds"][0]
    prediction = fold["predictions"][0]
    if tamper == "probabilities":
        prediction["probabilities"] = {"home_win": 0.9, "draw": 0.05, "away_win": 0.05}
    elif tamper == "actual":
        prediction["actual"] = "draw" if prediction["actual"] != "draw" else "home_win"
    elif tamper == "fold_metrics":
        fold["metrics"]["log_loss"] = 0.0
    elif tamper == "calibration":
        run["calibration"]["macro_expected_calibration_error"] = 0.0
    elif tamper == "lineage":
        prediction["model_id"] = "other"
    elif tamper == "cutoff":
        prediction["prediction_time_iso"] = "1900-01-01T00:00:00Z"
    elif tamper == "training_overlap":
        fold["train_match_ids"][0] = prediction["match_id"]
    elif tamper == "missing_models":
        report["runs"] = report["runs"][:1]
    elif tamper == "missing_transform":
        report["runs"] = [r for r in report["runs"] if r["backtest"]["model_spec_id"] != (
            "temperature_logistic_regression_v1"
        )]
    elif tamper == "context":
        report["feature_context"] = {"rankings": []}
    elif tamper == "catalogs":
        report["catalogs"]["teams"] = []
    elif tamper == "feature_contract":
        run["backtest"]["feature_set_id"] = "other"
    else:
        report["source_sha256"] = "other"
    with pytest.raises(ValueError):
        validate_research(report, bundle)


def test_review_binding_reimport_revocation_and_recovery(evidence, tmp_path):
    report, bundle = evidence
    repo = Repository(tmp_path / "db")
    now = datetime(2020, 7, 1, tzinfo=UTC)
    model_id = bundle["bundle_id"]
    repo.put("model", model_id, bundle, available_at=now, recorded_at=now)
    path = tmp_path / "research.json"
    path.write_text(json.dumps(report))

    def command(name, *args):
        nonlocal now
        now += timedelta(seconds=1)
        return execute(parser().parse_args([name, "--model-id", model_id, *args]), repo, now)

    command("import-research", "--input", str(path))
    with pytest.raises(ValueError):
        require_approved_model(repo, model_id, as_of=now)
    approval = command("approve-model", "--reason", "Synthetic acceptance")
    assert require_approved_model(repo, model_id, as_of=now)["metrics_record_id"] == (
        approval["metrics_record_id"]
    )
    reviewed_at = now
    command("import-research", "--input", str(path))
    assert repo.get("model_release", model_id, as_of=now)["payload"]["status"] == "candidate"
    with pytest.raises(ValueError):
        require_approved_model(repo, model_id, as_of=now)
    assert require_approved_model(repo, model_id, as_of=reviewed_at)["status"] == "approved"
    command("approve-model", "--reason", "Reviewed new revision")
    command("revoke-model", "--reason", "Retire model")
    with pytest.raises(ValueError):
        require_approved_model(repo, model_id, as_of=now)
    command("approve-model", "--reason", "Revalidated")
    now += timedelta(seconds=1)
    # Even bypassing the import command cannot silently carry approval forward.
    repo.put("model_metrics", model_id, report, available_at=now, recorded_at=now)
    with pytest.raises(ValueError, match="superseded"):
        require_approved_model(repo, model_id, as_of=now)
    command("approve-model", "--reason", "Reviewed exact replacement")
    backup = tmp_path / "backup"
    repo.backup(backup)
    assert require_approved_model(Repository(backup), model_id, as_of=now) == (
        require_approved_model(repo, model_id, as_of=now)
    )


def test_dashboard_cursor_navigation_preserves_snapshot(monkeypatch):
    import urllib.request
    from unittest.mock import MagicMock
    from urllib.parse import parse_qs, urlparse

    from streamlit.testing.v1 import AppTest

    calls = []

    def respond(request, **kwargs):
        url = urlparse(request.full_url)
        params = parse_qs(url.query)
        calls.append((url.path, params))
        second = "after" in params
        payload = {
            "items": [{"team_id": "Z" if second else "A"}],
            "as_of": "2020-01-01T00:00:00+00:00",
            "next_cursor": None if second else "A", "truncated": not second,
        }
        if url.path.startswith("/reports/team/"):
            payload = {"rating": {}, "recent_form": [], "watch_note": "", "rating_history": []}
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(payload).encode()
        return response

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    app = AppTest.from_file(str(ROOT / "src/football_analytics/dashboard.py"))
    app.run(timeout=15)
    app.sidebar.radio[0].set_value("Team intelligence").run(timeout=15)
    assert app.selectbox[0].options == ["A"]
    next(b for b in app.button if b.label == "Next teams").click().run(timeout=15)
    assert not app.exception
    assert app.selectbox[0].options == ["Z"]
    assert ("/teams", {"after": ["A"], "as_of": ["2020-01-01T00:00:00+00:00"]}) in calls
    next(b for b in app.button if b.label == "Previous teams").click().run(timeout=15)
    assert app.selectbox[0].options == ["A"]
    next(b for b in app.button if b.label == "Refresh teams").click().run(timeout=15)
    assert ("/teams", {}) == [call for call in calls if call[0] == "/teams"][-1]
