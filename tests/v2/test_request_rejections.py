"""Malformed and oversized requests stay bounded, correlated and JSON-safe."""

import logging

import pytest
from fastapi.testclient import TestClient

from football_analytics.api.main import create_app
from football_analytics.config import Settings

KEY = "request-rejection-test-secret-24"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_invalid_input_returns_json_validation_error(tmp_path, caplog, value):
    app = create_app(Settings(tmp_path / "db", KEY))
    client = TestClient(app, raise_server_exceptions=False)
    caplog.set_level(logging.INFO, logger="football_analytics.api")
    response = client.post(
        "/predict", content='{"model_id":' + value + ',"match_id":"f"}',
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "model_id"]
    assert any(
        r.message == "request_completed" and r.status == 422
        and r.request_id == response.headers["X-Request-ID"] for r in caplog.records
    )
    assert app.state.repository.scan("forecast") == []


@pytest.mark.parametrize("body", [
    '{"model_id":"m","match_id":"f","private_token":"sensitive-value"}',
    '{"model_id":"sensitive-value",',
])
def test_invalid_request_does_not_echo_submitted_values(tmp_path, body):
    app = create_app(Settings(tmp_path / "db", KEY))
    response = TestClient(app).post(
        "/predict", content=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "sensitive-value" not in response.text
    assert all("input" not in error and "ctx" not in error for error in response.json()["detail"])


def test_oversized_request_has_completion_log_and_request_id(tmp_path, caplog):
    app = create_app(Settings(tmp_path / "db", KEY))
    caplog.set_level(logging.INFO, logger="football_analytics.api")
    response = TestClient(app).post("/predict", content=b"x" * 16385)
    assert response.status_code == 413
    request_id = response.headers["X-Request-ID"]
    assert any(
        r.message == "request_completed" and r.status == 413 and r.request_id == request_id
        for r in caplog.records
    )
    assert app.state.repository.scan("forecast") == []


def test_body_limit_accepts_exact_boundary_and_preserves_auth(tmp_path):
    app = create_app(Settings(tmp_path / "db", KEY))
    client = TestClient(app)
    body = b'{"model_id":"missing","match_id":"f"}'
    body += b" " * (16384 - len(body))
    headers = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
    assert client.post("/predict", content=body, headers=headers).status_code == 404
    assert client.post("/predict", content=body + b" ", headers=headers).status_code == 413
    assert client.post("/predict", json={"model_id": 123, "match_id": "f"}).status_code == 401


def test_chunked_oversize_stops_reading_and_never_dispatches(tmp_path, monkeypatch):
    import asyncio
    import json

    from football_analytics.api import routes as api_routes

    app = create_app(Settings(tmp_path / "db", KEY))
    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("Oversized request reached forecasting")
    monkeypatch.setattr(api_routes, "predict_and_publish", unexpected_dispatch)
    messages = []
    consumed = 0

    async def exercise():
        async def receive():
            nonlocal consumed
            consumed += 1
            if consumed > 3:
                pytest.fail("Continued reading after exceeding body limit")
            return {"type": "http.request", "body": b"x" * [8000, 8384, 1][consumed - 1],
                    "more_body": True}
        async def send(message):
            messages.append(message)
        await asyncio.wait_for(app({
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/predict", "raw_path": b"/predict",
            "query_string": b"", "root_path": "", "server": ("test", 80),
            "client": ("test", 123), "headers": [(b"content-type", b"application/json"),
            (b"authorization", ("Bearer " + KEY).encode())],
        }, receive, send), timeout=5)
    asyncio.run(exercise())
    assert consumed == 3
    start = next(m for m in messages if m["type"] == "http.response.start")
    assert start["status"] == 413
    assert b"x-request-id" in dict(start["headers"])
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    assert json.loads(body) == {"detail": "Request body too large"}
