from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from football_analytics.api.main import create_app
from football_analytics.config import Settings
from football_analytics.services.history import match_revision_history
from football_analytics.storage.repository import Repository, RepositoryIntegrityError

START = datetime(2020, 1, 1, tzinfo=UTC)
CUTOFF = START + timedelta(days=3)


def test_history_retains_superseded_revisions_and_freezes_commit_boundary(tmp_path):
    repo = Repository(tmp_path / "db")
    first = repo.put("match", "m", {"status": "scheduled"}, available_at=START, recorded_at=START)
    repo.put("match", "m", {"status": "completed"}, available_at=START,
             recorded_at=START + timedelta(days=1))
    page = match_revision_history(repo, "m", as_of=CUTOFF, limit=1)
    assert page["items"][0]["record_id"] == first
    # Later commit is backdated before the frozen timestamp; it must stay excluded.
    repo.put("match", "m", {"status": "corrected"}, available_at=START,
             recorded_at=START + timedelta(days=2))
    second = match_revision_history(
        repo, "m", as_of=CUTOFF, through_sequence=page["through_sequence"],
        after_sequence=page["next_cursor"],
    )
    assert [r["payload"]["status"] for r in second["items"]] == ["completed"]
    assert second["next_cursor"] is None
    assert len(match_revision_history(repo, "m", as_of=CUTOFF)["items"]) == 3
    assert len(match_revision_history(repo, "m", as_of=START)["items"]) == 1
    assert match_revision_history(repo, "missing", as_of=CUTOFF)["items"] == []
    target = tmp_path / "backup"
    repo.backup(target)
    assert match_revision_history(
        Repository(target), "m", as_of=CUTOFF, through_sequence=page["through_sequence"]
    )["items"] == match_revision_history(
        repo, "m", as_of=CUTOFF, through_sequence=page["through_sequence"]
    )["items"]


@pytest.mark.parametrize("prefix", ["", "/api/v1"])
def test_api_history_auth_cutoffs_and_corruption(tmp_path, prefix):
    app = create_app(Settings(tmp_path / "db", "history-test-secret-24-characters"))
    repo = app.state.repository
    repo.put("match", "m", {"v": 1}, available_at=START, recorded_at=START)
    repo.put("match", "m", {"v": 2}, available_at=START,
             recorded_at=START + timedelta(days=1))
    client = TestClient(app)
    path = prefix + "/matches/m/history"
    assert client.get(path).status_code == 401
    client.headers["Authorization"] = "Bearer history-test-secret-24-characters"
    response = client.get(path, params={"as_of": START.isoformat()})
    assert response.status_code == 200
    assert [r["payload"] for r in response.json()["items"]] == [{"v": 1}]
    assert response.headers["X-Request-ID"]
    for params in [{"after_sequence": 1}, {"through_sequence": 1}, {"limit": 0},
                   {"as_of": "2020-01-01"}, {"through_sequence": 999, "as_of": CUTOFF.isoformat()}]:
        assert client.get(path, params=params).status_code == 422
    with repo.connection() as db:
        db.execute("UPDATE records SET payload='{}' WHERE sequence=1")
    # Latest entity reads still succeed; revision inspection verifies old content.
    assert repo.get("match", "m")["payload"] == {"v": 2}
    assert client.get(path).status_code == 503
    with pytest.raises(RepositoryIntegrityError):
        match_revision_history(repo, "m")




def test_cli_history_pages_match_api_snapshot(tmp_path, capsys):
    import json

    from football_analytics.cli import main

    repo = Repository(tmp_path / "db")
    for day in range(3):
        repo.put("match", "m", {"version": day}, available_at=START,
                 recorded_at=START + timedelta(days=day))
    command = ["match-history", "--database", str(repo.path), "--match-id", "m",
               "--as-of", CUTOFF.isoformat(), "--limit", "1"]
    main(command)
    first = json.loads(capsys.readouterr().out)
    main([*command, "--through-sequence", str(first["through_sequence"]),
          "--after-sequence", str(first["next_cursor"])])
    second = json.loads(capsys.readouterr().out)
    assert second["items"][0]["payload"] == {"version": 1}
    assert second["through_sequence"] == first["through_sequence"]
    with pytest.raises(SystemExit) as exc:
        main(["match-history", "--database", str(repo.path), "--match-id", "m",
              "--after-sequence", "1"])
    assert exc.value.code == 2
    assert "continuation requires" in capsys.readouterr().err


def test_repository_history_validates_numeric_bounds_and_utc(tmp_path):
    repo = Repository(tmp_path / "db")
    for options in [{"through_sequence": -1}, {"through_sequence": 2**63},
                    {"through_sequence": True}, {"after_sequence": -1},
                    {"after_sequence": 1.5}, {"limit": 0}, {"limit": True}]:
        with pytest.raises(ValueError):
            repo.history("match", "m", **{"as_of": CUTOFF, "through_sequence": 0, **options})
    with pytest.raises(ValueError):
        repo.history("match", "m", as_of=datetime(2020, 1, 1), through_sequence=0)
    assert repo.history("match", "m", as_of=CUTOFF, through_sequence=0) == []
