import json
import sqlite3
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_analytics.api.main import create_app
from football_analytics.config import Settings
from football_analytics.data import load_canonical_catalogs
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.data.serialization import json_safe
from football_analytics.domain import Match
from football_analytics.features.materialization import MaterializedFeatureRow
from football_analytics.models.portable import PortableClassifier, export_classifier
from football_analytics.models.sklearn_models import SklearnOutcomeModel
from football_analytics.ratings.glicko import GlickoState, inflate, update_period
from football_analytics.services.forecasting import forecast_match, train_forecast_bundle
from football_analytics.storage.repository import Repository

SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"


@pytest.fixture(scope="module")
def bundle():
    catalogs = load_canonical_catalogs(
        teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
    )
    observations = build_legacy_mens_results_observations(
        pd.read_csv(SAMPLE / "results.csv"), ingested_at=datetime(2020, 6, 1, tzinfo=UTC)
    )
    return train_forecast_bundle(
        observations,
        catalogs,
        cutoff=datetime(2020, 5, 15, tzinfo=UTC),
        holdout_days=30,
        min_calibration=5,
    )


@pytest.mark.parametrize("kind", ["logistic", "boosting"])
def test_portable_classifier_matches_sklearn_on_unseen_numeric_inputs(kind):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(150, 4))
    y = np.array(["home_win", "draw", "away_win"])[np.argmax(x[:, :3], axis=1)]
    estimator = (
        Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])
        if kind == "logistic"
        else HistGradientBoostingClassifier(max_iter=20, min_samples_leaf=5, random_state=1)
    )
    estimator.fit(x, y)
    model = SklearnOutcomeModel("fitted", ("a", "b", "c", "d"), "features", "imputation", estimator)
    artifact = export_classifier(model)
    loaded = PortableClassifier(json.loads(json.dumps(artifact.payload)))
    for values in rng.normal(size=(60, 4)):
        row = MaterializedFeatureRow(
            "fixture",
            "2020-01-01T00:00:00Z",
            "features",
            "imputation",
            tuple(zip(model.feature_names, values, strict=True)),
            (),
        )
        assert loaded.predict_row(row).as_tuple() == pytest.approx(
            model.predict_row(row).as_tuple(), abs=1e-12
        )
    with pytest.raises(ValueError, match="identity"):
        PortableClassifier({**artifact.payload, "model_id": "tampered"})
    with pytest.raises(ValueError, match="contract"):
        loaded.predict_row(replace(row, feature_set_id="other"))


def test_bundle_roundtrip_all_methods_and_pre_match_restrictions(bundle):
    match = Match("future", date(2020, 7, 1), "ARG", "BRA", "friendly", True)
    now = datetime(2020, 6, 1, tzinfo=UTC)
    for method in (
        "ensemble",
        *bundle["classifiers"],
        "independent_poisson_v1",
        "calibrated_logistic_regression_v1",
    ):
        # Specs are versioned; derive the logistic transform name from its actual member.
        if method.startswith("calibrated_"):
            method = "calibrated_" + bundle["transforms"][1]["members"][0]["spec_id"]
        first = forecast_match(bundle, match, prediction_time=now, method=method)
        assert first == forecast_match(
            json.loads(json.dumps(bundle)), match, prediction_time=now, method=method
        )
        assert sum(first["probabilities"].values()) == pytest.approx(1)
        assert first["uncertainty_interval"] is None
    with pytest.raises(ValueError, match="cutoff"):
        forecast_match(bundle, match, prediction_time=datetime(2020, 4, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        forecast_match(bundle, match, prediction_time=datetime(2020, 7, 2, tzinfo=UTC))
    with pytest.raises(ValueError, match="identity"):
        forecast_match({**bundle, "training_cutoff": now.isoformat()}, match, prediction_time=now)


def test_bitemporal_revisions_idempotence_conflicts_and_backup(tmp_path):
    repo = Repository(tmp_path / "db.sqlite3")
    first = datetime(2020, 1, 1, tzinfo=UTC)
    second = first + timedelta(days=1)
    identity = repo.put("match", "m", {"score": 1}, available_at=first, recorded_at=first)
    assert repo.put("match", "m", {"score": 1}, available_at=first, recorded_at=first) == identity
    repo.put("match", "m", {"score": 2}, available_at=first, recorded_at=second)
    assert repo.get("match", "m", as_of=first)["payload"] == {"score": 1}
    assert repo.get("match", "m", as_of=second)["payload"] == {"score": 2}
    with pytest.raises(sqlite3.IntegrityError):
        repo.put("match", "m", {"score": 3}, available_at=first, recorded_at=second)
    target = tmp_path / "backup.sqlite3"
    repo.backup(target)
    restored = Repository(target)
    assert restored.health()
    assert restored.get("match", "m", as_of=first) == repo.get("match", "m", as_of=first)
    with pytest.raises(ValueError):
        repo.backup(target)


def test_repository_detects_corrupt_payload(tmp_path):
    repo = Repository(tmp_path / "db")
    now = datetime.now(UTC)
    repo.put("team", "A", {"name": "A"}, available_at=now, recorded_at=now)
    with repo.connection() as db:
        db.execute("UPDATE records SET payload='{}'")
    with pytest.raises(ValueError, match="integrity"):
        repo.get("team", "A")


def test_glicko_published_example_and_inactivity():
    state = update_period(
        GlickoState(1500, 200),
        [(GlickoState(1400, 30), 1), (GlickoState(1550, 100), 0), (GlickoState(1700, 300), 0)],
    )
    assert state.rating == pytest.approx(1464.106, abs=0.001)
    assert state.deviation == pytest.approx(151.399, abs=0.001)
    assert inflate(state, 30).deviation > state.deviation
    assert inflate(state, 100000).deviation == 350


def test_native_api_auth_approval_forecast_and_read_routes(tmp_path, bundle):
    settings = Settings(
        tmp_path / "api.sqlite3",
        "a-secure-test-key-of-more-than-24-characters",
        max_model_age_days=5000,
    )
    client = TestClient(create_app(settings))
    repo = client.app.state.repository
    now = datetime.now(UTC) - timedelta(seconds=1)
    for team in bundle["catalogs"]["teams"]:
        repo.put("team", team["team_id"], team, available_at=now, recorded_at=now)
    for record in bundle["records"]:
        repo.put("match", record["match"]["match_id"], record, available_at=now, recorded_at=now)
    match = Match("future", (now + timedelta(days=2)).date(), "ARG", "BRA", "friendly", True)
    source = {
        **bundle["records"][0],
        "match": json_safe(match),
        "home_score": None,
        "away_score": None,
    }
    repo.put("match", "future", source, available_at=now, recorded_at=now)
    repo.put("model", bundle["bundle_id"], bundle, available_at=now, recorded_at=now)
    repo.put(
        "model_release",
        bundle["bundle_id"],
        {"status": "candidate"},
        available_at=now,
        recorded_at=now,
    )
    assert client.get("/health").status_code == 200
    assert client.get("/teams").status_code == 401
    client.headers["Authorization"] = "Bearer " + settings.api_key
    first_page = client.get("/teams?limit=2").json()
    second_page = client.get(
        "/teams",
        params={"limit": 2, "after": first_page["next_cursor"], "as_of": first_page["as_of"]},
    ).json()
    assert len(first_page["items"] + second_page["items"]) == 4
    assert not set(r["team_id"] for r in first_page["items"]) & set(
        r["team_id"] for r in second_page["items"]
    )
    assert client.get("/teams?as_of=2020-01-01").status_code == 422
    assert client.post("/predict", content=b"x" * 16385).status_code == 413
    assert client.get("/teams?limit=0").status_code == 422
    assert client.get("/teams/missing").status_code == 404
    request = {"model_id": bundle["bundle_id"], "match_id": "future"}
    assert client.post("/predict", json=request).status_code == 409
    later = now + timedelta(milliseconds=1)
    repo.put(
        "model_release",
        bundle["bundle_id"],
        {"status": "approved"},
        available_at=later,
        recorded_at=later,
    )
    response = client.post("/predict", json=request)
    assert response.status_code == 200, response.text
    assert response.headers.get("x-request-id")
    assert (
        client.get("/predictions").json()["items"][0]["forecast_id"]
        == response.json()["forecast_id"]
    )
    for path in (
        "/version",
        "/teams/ARG",
        "/teams/ARG/ratings",
        "/teams/ARG/report",
        "/matches",
        "/matches/future",
        "/ratings",
        "/ratings/movers",
        "/models",
        "/reports/match/future",
        "/reports/team/ARG",
        "/diagnostics/upsets",
        "/diagnostics/regression-watchlist",
        "/diagnostics/breakout-watchlist",
    ):
        assert client.get(path).status_code == 200, path
    assert client.get(f"/models/{bundle['bundle_id']}/metrics").status_code == 404


def test_production_config_requires_credentials():
    with pytest.raises(ValueError):
        Settings()


def test_atomic_repository_batch_rollback_and_complete_scan(tmp_path):
    repo = Repository(tmp_path / "batch")
    now = datetime.now(UTC)
    repo.put("x", "conflict", {"v": 1}, available_at=now, recorded_at=now)
    with pytest.raises(sqlite3.IntegrityError):
        repo.put_many([("x", "new", {}, now), ("x", "conflict", {"v": 2}, now)], recorded_at=now)
    with pytest.raises(KeyError):
        repo.get("x", "new")
    repo.put_many([("x", f"id{i:04d}", {"v": i}, now) for i in range(1005)], recorded_at=now)
    assert len(repo.scan("x")) == 1006
    first = repo.list("x", limit=20)
    second = repo.list("x", limit=20, after_id=first[-1]["entity_id"])
    assert not {r["entity_id"] for r in first} & {r["entity_id"] for r in second}


def test_dashboard_empty_state_and_api_failure(monkeypatch):
    import urllib.request
    from unittest.mock import MagicMock

    from streamlit.testing.v1 import AppTest

    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b'{"status":"ok","items":[]}'
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: response)
    app = AppTest.from_file(str(SAMPLE.parents[2] / "src/football_analytics/dashboard.py"))
    app.run(timeout=15)
    assert not app.exception
    assert app.title[0].value == "Football Data Analytics Software"
    from urllib.error import URLError

    def fail(*args, **kwargs):
        raise URLError("test")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    app.run(timeout=15)
    assert not app.exception
    assert len(app.error) == 1


def test_unsupported_storage_schema_is_not_modified(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE schema_version(version INTEGER PRIMARY KEY)")
        db.execute("INSERT INTO schema_version VALUES(2)")
    with pytest.raises(ValueError, match="schema"):
        Repository(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version FROM schema_version").fetchall() == [(2,)]
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='records'").fetchall()


def test_composed_forecast_features_are_replayed_from_bundle():
    catalogs = load_canonical_catalogs(
        teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
    )
    observations = build_legacy_mens_results_observations(
        pd.read_csv(SAMPLE / "results.csv"), ingested_at=datetime(2020, 6, 1, tzinfo=UTC)
    )
    model = train_forecast_bundle(
        observations,
        catalogs,
        cutoff=datetime(2020, 5, 15, tzinfo=UTC),
        holdout_days=30,
        min_calibration=5,
        feature_groups=("form", "schedule", "competition"),
    )
    match = Match("future", date(2020, 7, 1), "ARG", "BRA", "friendly", True)
    forecast = forecast_match(
        json.loads(json.dumps(model)), match, prediction_time=datetime(2020, 6, 1, tzinfo=UTC)
    )
    assert abs(sum(forecast["probabilities"].values()) - 1) < 1e-12


def test_dashboard_all_populated_pages(monkeypatch, bundle):
    import urllib.request
    from unittest.mock import MagicMock

    from streamlit.testing.v1 import AppTest

    match = Match("future", date(2020, 7, 1), "ARG", "BRA", "friendly", True)
    forecast = forecast_match(bundle, match, prediction_time=datetime(2020, 6, 1, tzinfo=UTC))
    rating = {"team_id": "ARG", "rating": 1500, "deviation": 100, "effective_date": "2020-05-01"}
    replies = {
        "/health": {"status": "ok"},
        "/teams": {"items": [{"team_id": "ARG"}]},
        "/models": {"items": [{"model_id": "test-model"}]},
        "/ratings": [rating],
        "/predictions": {"items": [forecast], "truncated": False},
        "/reports/team/ARG": {
            "rating": rating,
            "recent_form": [{"points": 3}],
            "watch_note": "descriptive",
            "rating_history": [rating],
        },
        "/models/test-model/metrics": {
            "comparison": {"rows": []},
            "runs": [{"backtest": {"model_spec_id": "test"}, "calibration": {"count": 20}}],
        },
        "/diagnostics/upsets": [{"match_id": "prior", "surprisal_nats": 1.2}],
    }

    def respond(request, **kwargs):
        from urllib.parse import urlparse

        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(replies[urlparse(request.full_url).path]).encode()
        return response

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    app = AppTest.from_file(str(SAMPLE.parents[2] / "src/football_analytics/dashboard.py"))
    app.run(timeout=15)
    for page in (
        "Ratings",
        "Match forecasts",
        "Team intelligence",
        "Model diagnostics",
        "Post-match review",
    ):
        app.sidebar.radio[0].set_value(page).run(timeout=15)
        assert not app.exception, page
        assert not app.error, page
