"""Storage exhaustion/lock rollback, API errors and subsequent recovery."""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from football_analytics.api import main as api
from football_analytics.config import Settings
from football_analytics.storage.repository import Repository

KEY = "storage-failure-tests-secret-24"


def test_full_database_rolls_back_entire_batch_then_recovers(tmp_path, monkeypatch):
    repo = Repository(tmp_path / "db")
    connection = repo.connection

    @contextmanager
    def constrained():
        with connection() as db:
            pages = db.execute("PRAGMA page_count").fetchone()[0]
            db.execute(f"PRAGMA max_page_count={pages}")
            yield db

    monkeypatch.setattr(repo, "connection", constrained)
    now = datetime.now(UTC)
    with pytest.raises(sqlite3.OperationalError) as exc:
        repo.put_many(
            [
                ("forecast", "small", {}, now),
                ("forecast_batch", "large", {"x": "x" * 1000000}, now),
            ],
            recorded_at=now,
        )
    assert exc.value.sqlite_errorcode == sqlite3.SQLITE_FULL
    assert not repo.scan("forecast")
    assert not repo.scan("forecast_batch")
    monkeypatch.setattr(repo, "connection", connection)
    with repo.connection() as db:
        db.execute("PRAGMA max_page_count=100000")
    repo.put("forecast", "recovered", {"ok": True}, available_at=now)
    assert repo.health()
    assert repo.get("forecast", "recovered")["payload"] == {"ok": True}


def test_api_write_lock_failure_is_correlated_and_recovers(tmp_path, monkeypatch, caplog):
    app = api.create_app(Settings(tmp_path / "db", KEY))
    repo = app.state.repository
    connection = repo.connection

    @contextmanager
    def short_wait():
        with connection() as db:
            db.execute("PRAGMA busy_timeout=1")
            yield db

    def store_forecast(repository, **kwargs):
        now = datetime.now(UTC)
        repository.put("forecast", "f", {"ok": True}, available_at=now)
        return {"forecast_id": "f"}

    monkeypatch.setattr(repo, "connection", short_wait)
    monkeypatch.setattr(api, "predict_and_publish", store_forecast)
    client = TestClient(app)
    client.headers["Authorization"] = "Bearer " + KEY
    caplog.set_level(logging.INFO, logger="football_analytics.api")
    with connection() as writer:
        writer.execute("BEGIN IMMEDIATE")
        response = client.post("/predict", json={"model_id": "m", "match_id": "f"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Storage unavailable; operation not confirmed"}
        assert not repo.scan("forecast")
    request_id = response.headers["X-Request-ID"]
    errors = [r for r in caplog.records if r.message == "storage_unavailable"]
    assert errors[0].request_id == request_id
    assert errors[0].sqlite_errorcode == sqlite3.SQLITE_BUSY
    assert any(
        r.message == "request_completed" and r.request_id == request_id for r in caplog.records
    )
    assert KEY not in caplog.text
    assert client.post("/predict", json={"model_id": "m", "match_id": "f"}).status_code == 200
    assert len(repo.scan("forecast")) == 1


@pytest.mark.parametrize("route", ["/health", "/teams"])
def test_api_read_storage_errors_do_not_expose_internal_details(tmp_path, monkeypatch, route):
    app = api.create_app(Settings(tmp_path / "db", KEY))
    repo = app.state.repository

    @contextmanager
    def unavailable():
        raise sqlite3.DatabaseError("/private/database/path: internal SQL details")
        yield  # Make this a context manager whose entry fails.

    monkeypatch.setattr(repo, "connection", unavailable)
    client = TestClient(app)
    response = client.get(route, headers={"Authorization": "Bearer " + KEY})
    assert response.status_code == 503
    assert response.headers["X-Request-ID"]
    assert "/private" not in response.text
    assert "SQL" not in response.text
    assert client.get("/version").status_code == 200
    assert client.get("/teams").status_code == 401
