"""Atomic snapshot publication and failure recovery."""

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from threading import Barrier

import pytest

from football_analytics.storage.repository import Repository


@pytest.fixture
def repository(tmp_path):
    repo = Repository(tmp_path / "source.sqlite3")
    now = datetime.now(UTC)
    repo.put("team", "A", {"name": "A"}, available_at=now, recorded_at=now)
    return repo


def test_backup_is_self_contained_and_not_visible_until_verified(repository, tmp_path, monkeypatch):
    destination = tmp_path / "backup.sqlite3"
    original = os.link

    def checked_link(source, target):
        assert not destination.exists()
        with sqlite3.connect(source) as db:
            assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
            assert db.execute("PRAGMA journal_mode").fetchone() == ("delete",)
            assert db.execute("SELECT COUNT(*) FROM records").fetchone() == (1,)
        original(source, target)

    monkeypatch.setattr(os, "link", checked_link)
    repository.backup(destination)
    assert not list(tmp_path.glob(".backup-*"))
    assert not destination.with_name(destination.name + "-wal").exists()
    assert Repository(destination).get("team", "A") == repository.get("team", "A")


def test_concurrent_backups_preserve_one_complete_snapshot(repository, tmp_path, monkeypatch):
    destination = tmp_path / "backup.sqlite3"
    original = os.link
    ready = Barrier(2)

    def racing_link(source, target):
        ready.wait(timeout=5)
        original(source, target)

    monkeypatch.setattr(os, "link", racing_link)

    def backup():
        try:
            repository.backup(destination)
            return "published"
        except ValueError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: backup(), range(2)))
    assert sorted(results) == ["conflict", "published"]
    assert Repository(destination).get("team", "A") == repository.get("team", "A")
    assert not list(tmp_path.glob(".backup-*"))


def test_failed_copy_leaves_no_destination_or_staging_files(repository, tmp_path, monkeypatch):
    destination = tmp_path / "backup.sqlite3"
    original = repository.connection

    class FailedCopy:
        def backup(self, target):
            target.execute("CREATE TABLE partial(value)")
            target.commit()
            raise sqlite3.DatabaseError("injected copy failure")

    @contextmanager
    def interrupted_connection():
        with original():
            yield FailedCopy()

    monkeypatch.setattr(repository, "connection", interrupted_connection)
    with pytest.raises(sqlite3.DatabaseError, match="injected"):
        repository.backup(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".backup-*"))
    monkeypatch.setattr(repository, "connection", original)
    repository.backup(destination)
    assert Repository(destination).health()


def test_prepublication_sync_failure_does_not_publish(repository, tmp_path, monkeypatch):
    destination = tmp_path / "backup.sqlite3"

    def failed_sync(fd):
        raise OSError("injected sync failure")

    monkeypatch.setattr(os, "fsync", failed_sync)
    with pytest.raises(OSError, match="injected"):
        repository.backup(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".backup-*"))


def test_destination_created_during_copy_is_never_overwritten(repository, tmp_path, monkeypatch):
    destination = tmp_path / "backup.sqlite3"
    original = os.link

    def create_conflict(source, target):
        destination.write_bytes(b"another owner's file")
        original(source, target)

    monkeypatch.setattr(os, "link", create_conflict)
    with pytest.raises(ValueError, match="new file"):
        repository.backup(destination)
    assert destination.read_bytes() == b"another owner's file"
    assert not list(tmp_path.glob(".backup-*"))


def test_backup_rejects_dangling_symlink_without_writing_target(repository, tmp_path):
    destination = tmp_path / "backup.sqlite3"
    missing = tmp_path / "missing"
    destination.symlink_to(missing)
    with pytest.raises(ValueError, match="new file"):
        repository.backup(destination)
    assert destination.is_symlink()
    assert not missing.exists()
    assert not list(tmp_path.glob(".backup-*"))


def test_directory_sync_failure_retains_complete_backup(repository, tmp_path, monkeypatch):
    destination = tmp_path / "backup.sqlite3"
    original = os.fsync
    calls = 0

    def fail_second_sync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory sync failed")
        original(fd)

    monkeypatch.setattr(os, "fsync", fail_second_sync)
    with pytest.raises(OSError, match="directory sync"):
        repository.backup(destination)
    assert Repository(destination).get("team", "A") == repository.get("team", "A")
    assert not list(tmp_path.glob(".backup-*"))


def test_failed_integrity_check_prevents_publication(repository, tmp_path, monkeypatch):
    destination = tmp_path / "backup.sqlite3"
    original = sqlite3.connect

    class FailedIntegrity(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA integrity_check":
                return super().execute("SELECT 'injected structural failure'")
            return super().execute(sql, *args, **kwargs)

    def connect(path, *args, **kwargs):
        if str(path).endswith("/snapshot.sqlite3"):
            kwargs["factory"] = FailedIntegrity
        return original(path, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    with pytest.raises(sqlite3.DatabaseError, match="integrity"):
        repository.backup(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".backup-*"))


@pytest.mark.parametrize("hidden", ["superseded", "future"])
@pytest.mark.parametrize("raw", ['{"altered":true}', "{invalid", "[]", '{"x":NaN}'])
def test_backup_rejects_corrupt_hidden_revisions(tmp_path, hidden, raw):
    from football_analytics.storage.repository import RepositoryIntegrityError

    repo = Repository(tmp_path / "db")
    first = datetime(2020, 1, 1, tzinfo=UTC)
    current = datetime(2021, 1, 1, tzinfo=UTC)
    hidden_at = first if hidden == "superseded" else datetime(2100, 1, 1, tzinfo=UTC)
    bad = repo.put("team", "A", {"name": "old"}, available_at=hidden_at, recorded_at=hidden_at)
    repo.put("team", "A", {"name": "current"}, available_at=current, recorded_at=current)
    with repo.connection() as db:
        db.execute("UPDATE records SET payload=? WHERE record_id=?", (raw, bad))
    assert repo.health()  # SQLite structure remains valid.
    assert repo.get("team", "A")["payload"] == {"name": "current"}
    destination = tmp_path / "backup"
    with pytest.raises(RepositoryIntegrityError, match="content integrity"):
        repo.backup(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".backup-*"))
    # Failure does not delete or repair the source evidence.
    with repo.connection() as db:
        assert db.execute("SELECT payload FROM records WHERE record_id=?", (bad,)).fetchone() == (
            raw,
        )
