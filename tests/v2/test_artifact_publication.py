"""Immutable artifact publication must sync its name before acknowledging success."""

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from football_analytics.storage.files import publish_immutable


@pytest.mark.parametrize("existing", [False, True])
def test_directory_sync_failure_is_reported_and_retry_recovers(tmp_path, monkeypatch, existing):
    target = tmp_path / "report.json"
    data = b'{"complete":true}\n'
    if existing:
        publish_immutable(target, data)
    original = os.fsync
    calls = []

    def fail_directory(fd):
        directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        calls.append(directory)
        if directory:
            # The full bytes must be visible before syncing their directory entry.
            assert target.read_bytes() == data
            raise OSError("directory sync failed")
        original(fd)

    monkeypatch.setattr(os, "fsync", fail_directory)
    with pytest.raises(OSError, match="directory sync failed"):
        publish_immutable(target, data)
    assert calls == [False, True]
    assert list(tmp_path.iterdir()) == [target]
    assert target.read_bytes() == data
    monkeypatch.setattr(os, "fsync", original)
    assert publish_immutable(target, data) == target
    assert target.read_bytes() == data


def test_file_sync_failure_publishes_nothing(tmp_path, monkeypatch):
    def fail_sync(fd):
        raise OSError("file sync failed")
    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="file sync failed"):
        publish_immutable(tmp_path / "report", b"complete")
    assert list(tmp_path.iterdir()) == []


def test_conflicting_retry_retains_original_and_cleans_staging(tmp_path):
    target = tmp_path / "report"
    publish_immutable(target, b"original")
    with pytest.raises(ValueError, match="content conflict"):
        publish_immutable(target, b"replacement")
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("same_content", [False, True])
def test_concurrent_publication_leaves_one_complete_result(tmp_path, same_content):
    target = tmp_path / "report"
    barrier = Barrier(2)
    values = [b"a" * 100000, (b"a" if same_content else b"b") * 100000]

    def publish(value):
        barrier.wait(timeout=5)
        try:
            publish_immutable(target, value)
            return value
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, values))
    winners = [value for value in results if value is not None]
    assert len(winners) == (2 if same_content else 1)
    assert target.read_bytes() == winners[0]
    assert list(tmp_path.iterdir()) == [target]
