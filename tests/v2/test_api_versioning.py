"""Canonical API v1 and compatibility aliases use the same application contracts."""

import sqlite3

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from football_analytics.api.main import create_app
from football_analytics.api.routes import create_router
from football_analytics.config import Settings

KEY = "versioned-api-test-secret-24"


def test_route_inventory_shares_handlers_and_documents_only_v1(tmp_path):
    app = create_app(Settings(tmp_path / "db", KEY))
    router = create_router(app.state.repository, Settings(tmp_path / "db", KEY))
    routes = [r for r in router.routes if isinstance(r, APIRoute)]
    canonical = {"/api/v1" + route.path for route in routes}
    assert len(canonical) >= 20
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == canonical
    for route in routes:
        # Both route trees enforce authentication, including parameterized URLs.
        for method in route.methods:
            path = route.path
            for parameter in route.param_convertors:
                path = path.replace("{" + parameter + "}", "missing")
            canonical_response = client.request(method, "/api/v1" + path)
            alias_response = client.request(method, path)
            assert canonical_response.status_code == alias_response.status_code
            assert canonical_response.json() == alias_response.json()
            expected = 200 if path in ["/health", "/version"] else 401
            assert canonical_response.status_code == expected
    operations = [op["operationId"] for path in schema["paths"].values() for op in path.values()]
    assert len(operations) == len(set(operations))
    assert "/api/v1/predict" in schema["paths"]


@pytest.mark.parametrize("prefix", ["", "/api/v1"])
def test_shared_auth_validation_limits_and_storage_errors(tmp_path, prefix, monkeypatch):
    app = create_app(Settings(tmp_path / "db", KEY))
    client = TestClient(app)
    assert client.get(prefix + "/health").status_code == 200
    assert client.get(prefix + "/version").json()["api_schema"] == 1
    assert client.get(prefix + "/teams").status_code == 401
    client.headers["Authorization"] = "Bearer " + KEY
    invalid = client.post(prefix + "/predict", json={"model_id": 123, "match_id": "m"})
    assert invalid.status_code == 422
    assert "input" not in invalid.json()["detail"][0]
    large = client.post(prefix + "/predict", content=b"x" * 16385)
    assert large.status_code == 413
    assert large.headers["X-Request-ID"]
    assert client.get(prefix + "/teams/missing").status_code == 404
    def unavailable(*args, **kwargs):
        raise sqlite3.DatabaseError("private storage details")
    monkeypatch.setattr(app.state.repository, "snapshot_sequence", unavailable)
    response = client.get(prefix + "/teams")
    assert response.status_code == 503
    assert response.headers["X-Request-ID"]
    assert "private" not in response.text


def test_versioned_snapshot_can_continue_through_compatibility_alias(tmp_path):
    from datetime import UTC, datetime

    app = create_app(Settings(tmp_path / "db", KEY))
    now = datetime.now(UTC)
    app.state.repository.put_many(
        [("team", name, {"team_id": name}, now) for name in ["A", "B", "C"]], recorded_at=now
    )
    client = TestClient(app, headers={"Authorization": "Bearer " + KEY})
    first = client.get("/api/v1/teams", params={"limit": 1}).json()
    params = {key: first[key] for key in ["as_of", "through_sequence"]}
    params["after"] = first["next_cursor"]
    canonical = client.get("/api/v1/teams", params=params).json()
    assert client.get("/teams", params=params).json() == canonical
    assert [row["team_id"] for row in canonical["items"]] == ["B", "C"]
