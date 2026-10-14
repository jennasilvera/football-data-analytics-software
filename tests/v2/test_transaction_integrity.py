"""Integrity must hold inside publication and duplicate-write transactions."""

from datetime import UTC, datetime

import pytest

from football_analytics.storage.repository import Repository, RepositoryIntegrityError


@pytest.mark.parametrize("operation", ["put_many", "publish_once"])
@pytest.mark.parametrize("payload", ['{"changed":true}', "{invalid", "[]"])
def test_unchanged_input_id_does_not_hide_corruption(tmp_path, operation, payload):
    repo = Repository(tmp_path / "db")
    now = datetime.now(UTC)
    expected = repo.put("model_release", "model", {"status": "approved"}, available_at=now)
    # Simulates corruption after a consumer captured the reviewed revision.
    with repo.connection() as db:
        db.execute("UPDATE records SET payload=? WHERE record_id=?", (payload, expected))
    items = [("forecast", "f", {"value": 1}, now)]
    options = {
        "recorded_at": now,
        "expected_records": [("model_release", "model", expected)],
    }
    with pytest.raises(RepositoryIntegrityError):
        if operation == "put_many":
            repo.put_many(items, **options)
        else:
            repo.publish_once(
                "forecast_batch", "run", {"request_id": "r"}, items,
                available_at=now, **options,
            )
    assert not repo.scan("forecast")
    assert not repo.scan("forecast_batch")


@pytest.mark.parametrize("operation", ["put_many", "publish_once"])
def test_corrupted_duplicate_rolls_back_other_writes(tmp_path, operation):
    repo = Repository(tmp_path / "db")
    now = datetime.now(UTC)
    duplicate = ("forecast", "existing", {"value": 1}, now)
    identity = repo.put_many([duplicate], recorded_at=now)[0]
    with repo.connection() as db:
        db.execute("UPDATE records SET payload='{}' WHERE record_id=?", (identity,))
    items = [("forecast", "new", {"value": 2}, now), duplicate]
    with pytest.raises(RepositoryIntegrityError):
        if operation == "put_many":
            repo.put_many(items, recorded_at=now)
        else:
            repo.publish_once(
                "forecast_batch", "run", {"request_id": "r"}, items,
                available_at=now, recorded_at=now,
            )
    with repo.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM records").fetchone() == (1,)
    assert not repo.scan("forecast_batch")
    # Existing corruption is retained for investigation, not silently repaired.
    with pytest.raises(RepositoryIntegrityError):
        repo.get("forecast", "existing")
