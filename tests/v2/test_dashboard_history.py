"""Dashboard history navigation against the actual API and SQLite revisions."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from football_analytics.api.main import create_app
from football_analytics.config import Settings

ROOT = Path(__file__).resolve().parents[2]
START = datetime(2020, 1, 1, tzinfo=UTC)
KEY = "dashboard-history-secret-24-chars"


def setup_dashboard(tmp_path, monkeypatch):
    import urllib.request

    api = create_app(Settings(tmp_path / "db", KEY))
    client = TestClient(api)
    calls = []
    def respond(request, **kwargs):
        url = urlparse(request.full_url)
        calls.append((url.path, parse_qs(url.query)))
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
    return api.state.repository, calls


def put_revision(repo, match_id, version, recorded):
    repo.put("match", match_id, {
        "match": {"match_id": match_id, "status": "scheduled"},
        "version": version, "metadata": {"source": "synthetic-history"},
    }, available_at=START, recorded_at=recorded)


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_history_pages_freeze_snapshot_refresh_and_match_selection(tmp_path, monkeypatch):
    repo, calls = setup_dashboard(tmp_path, monkeypatch)
    for version in range(26):
        put_revision(repo, "A", version, START + timedelta(minutes=version))
    put_revision(repo, "B", 100, START)
    app = AppTest.from_file(str(ROOT / "src/football_analytics/dashboard.py"))
    app.run(timeout=15)
    app.sidebar.radio[0].set_value("Match history").run(timeout=15)
    assert not app.exception
    assert app.selectbox[0].value == "A"
    assert len(app.selectbox[1].options) == 25
    first = [c for c in calls if c[0] == "/api/v1/matches/A/history"][-1]
    assert first[1] == {"limit": ["25"]}
    put_revision(repo, "A", 26, START + timedelta(hours=1))
    button(app, "Next revisions").click().run(timeout=15)
    assert not app.exception
    assert len(app.selectbox[1].options) == 1
    params = [c for c in calls if c[0] == first[0]][-1][1]
    assert params["after_sequence"] == ["25"]
    assert "through_sequence" in params and "as_of" in params
    second_snapshot = params["through_sequence"]
    button(app, "Previous revisions").click().run(timeout=15)
    assert len(app.selectbox[1].options) == 25
    button(app, "Refresh revisions").click().run(timeout=15)
    button(app, "Next revisions").click().run(timeout=15)
    assert len(app.selectbox[1].options) == 2
    params = [c for c in calls if c[0] == first[0]][-1][1]
    assert params["through_sequence"] != second_snapshot
    app.selectbox[0].set_value("B").run(timeout=15)
    assert not app.exception
    assert len(app.selectbox[1].options) == 1
    assert [c for c in calls if c[0] == "/api/v1/matches/B/history"][-1][1] == {"limit": ["25"]}
    assert "synthetic-history" in app.json[0].value


def test_history_cutoff_empty_state_and_invalid_timezone(tmp_path, monkeypatch):
    repo, calls = setup_dashboard(tmp_path, monkeypatch)
    put_revision(repo, "A", 1, START)
    put_revision(repo, "A", 2, START + timedelta(days=1))
    app = AppTest.from_file(str(ROOT / "src/football_analytics/dashboard.py"))
    app.run(timeout=15)
    app.sidebar.radio[0].set_value("Match history").run(timeout=15)
    app.text_input[0].set_value(START.isoformat()).run(timeout=15)
    assert not app.exception
    assert len(app.selectbox[1].options) == 1
    app.text_input[0].set_value("2019-01-01T00:00:00Z").run(timeout=15)
    assert not app.exception
    assert any("No revisions" in i.value for i in app.info)
    count = len(calls)
    app.text_input[0].set_value("2020-01-01").run(timeout=15)
    assert not app.exception and len(app.error) == 1
    assert "timezone" in app.error[0].value
    assert not any(path.endswith("/history") for path, _ in calls[count:])
    app.text_input[0].set_value("").run(timeout=15)
    assert not app.exception and len(app.selectbox[1].options) == 2
    assert json.loads(app.json[0].value)["payload"]["version"] == 1


def test_history_workspace_without_imported_matches(tmp_path, monkeypatch):
    setup_dashboard(tmp_path, monkeypatch)
    app = AppTest.from_file(str(ROOT / "src/football_analytics/dashboard.py"))
    app.run(timeout=15)
    app.sidebar.radio[0].set_value("Match history").run(timeout=15)
    assert not app.exception
    assert any("No matches" in i.value for i in app.info)


def test_history_storage_error_is_visible_and_never_shown_as_empty(tmp_path, monkeypatch):
    repo, _ = setup_dashboard(tmp_path, monkeypatch)
    put_revision(repo, "A", 1, START)
    put_revision(repo, "A", 2, START + timedelta(days=1))
    with repo.connection() as db:
        db.execute("UPDATE records SET payload='{}' WHERE sequence=1")
    app = AppTest.from_file(str(ROOT / "src/football_analytics/dashboard.py"))
    app.run(timeout=15)
    app.sidebar.radio[0].set_value("Match history").run(timeout=15)
    assert not app.exception
    assert len(app.error) == 1
    assert "HTTPError" in app.error[0].value
    assert not any("No revisions" in i.value for i in app.info)
