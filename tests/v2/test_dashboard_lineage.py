"""Forecast lineage UI exercised through the native API and exact stored revisions."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from football_analytics.api.main import create_app
from football_analytics.config import Settings
from football_analytics.services.lineage import attach_operational_lineage

ROOT = Path(__file__).resolve().parents[2]
KEY = "forecast-lineage-ui-secret-24-chars"


def prepare(tmp_path, monkeypatch):
    import urllib.request

    app = create_app(Settings(tmp_path / "db", KEY))
    repo = app.state.repository
    now = datetime.now(UTC) - timedelta(seconds=5)
    refs = {}
    for role, kind, entity in [
        ("model_record_id", "model", "model"),
        ("metrics_record_id", "model_metrics", "model"),
        ("approval_record_id", "model_release", "model"),
        ("match_record_id", "match", "fixture"),
    ]:
        refs[role] = repo.put(
            kind, entity, {"status": "approved", "training_cutoff": now.isoformat(),
                           "transforms": []}, available_at=now, recorded_at=now,
        )
    payload = {
        "forecast_id": "research", "bundle_id": "model", "method": "synthetic",
        "match": {"match_id": "fixture", "home_team_id": "ARG", "away_team_id": "BRA"},
        "probabilities": {"home_win": 0.5, "draw": 0.3, "away_win": 0.2},
        "data_timestamp": now.isoformat(), "explanation_note": "Synthetic fixture",
        "explanatory_drivers": [], "market_disagreement": None,
    }
    forecast = attach_operational_lineage(payload, **refs)
    repo.put("forecast", forecast["forecast_id"], forecast, available_at=now, recorded_at=now)
    repo.put("forecast", "research", payload, available_at=now, recorded_at=now)
    repo.put("model_release", "model", {"status": "revoked"}, available_at=datetime.now(UTC))
    client = TestClient(app)
    calls = []
    def respond(request, **kwargs):
        url = urlparse(request.full_url)
        calls.append(url.path)
        response = client.get(url.path + ("?" + url.query if url.query else ""),
                              headers={"Authorization": request.get_header("Authorization", "")})
        if response.is_error:
            raise HTTPError(request.full_url, response.status_code, response.text, {}, None)
        wrapped = MagicMock()
        wrapped.__enter__.return_value = wrapped
        wrapped.read.return_value = response.content
        return wrapped
    monkeypatch.setenv("FOOTBALL_API_KEY", KEY)
    monkeypatch.delenv("FOOTBALL_API_URL", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", respond)
    return repo, forecast, refs, calls


def inspect(app):
    next(b for b in app.button if b.label == "Inspect forecast inputs").click().run(timeout=15)


def open_forecast(forecast_id):
    app = AppTest.from_file(str(ROOT / "src/football_analytics/dashboard.py"))
    app.run(timeout=15)
    app.sidebar.radio[0].set_value("Match forecasts").run(timeout=15)
    app.selectbox[0].set_value(forecast_id).run(timeout=15)
    return app


def test_lineage_inspection_after_revocation_and_legacy_selection(tmp_path, monkeypatch):
    _, forecast, refs, calls = prepare(tmp_path, monkeypatch)
    app = open_forecast(forecast["forecast_id"])
    assert not app.exception
    assert not any(path.endswith("/lineage") for path in calls)
    inspect(app)
    assert not app.exception and not app.error
    report = app.dataframe[-1].value
    assert set(report["record_id"]) == set(refs.values())
    assert any("historical inputs" in c.value for c in app.caption)
    assert any("current model approval" in c.value for c in app.caption)
    assert calls[-1] == "/api/v1/predictions/" + forecast["forecast_id"] + "/lineage"
    app.selectbox[0].set_value("research").run(timeout=15)
    assert not any(s.value == "Forecast input revisions" for s in app.subheader)
    inspect(app)
    assert not app.exception
    assert any("not recorded" in i.value for i in app.info)
    assert not any(s.value == "Forecast input revisions" for s in app.subheader)


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_lineage_inspection_never_reuses_success_after_storage_damage(
    tmp_path, monkeypatch, damage
):
    repo, forecast, refs, _ = prepare(tmp_path, monkeypatch)
    app = open_forecast(forecast["forecast_id"])
    inspect(app)
    assert not app.exception and not app.error
    with repo.connection() as db:
        if damage == "missing":
            db.execute("DELETE FROM records WHERE record_id=?", (refs["approval_record_id"],))
        else:
            db.execute("UPDATE records SET payload='{}' WHERE record_id=?",
                       (refs["approval_record_id"],))
    inspect(app)
    assert not app.exception and len(app.error) == 1
    assert "HTTPError" in app.error[0].value
    assert not any(s.value == "Forecast input revisions" for s in app.subheader)
    assert not any("not recorded" in i.value for i in app.info)
