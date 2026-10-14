"""Real-process failures across publication, governance, backup and retry.

Workers synchronize with pipes and bounded waits rather than scheduling sleeps.
Termination tests model process death, not machine power loss or disk failure.
"""

import multiprocessing
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

from football_analytics.services.governance import require_approved_model
from football_analytics.storage.repository import PublicationConflictError, Repository

TIMEOUT = 15


def _publish(path, channel, phase, number=0):
    repo = Repository(path)
    approval = repo.get("model_release", "m")
    expected = [
        ("model_release", "m", approval["record_id"]),
        ("model", "m", approval["payload"]["model_record_id"]),
        ("model_metrics", "m", approval["payload"]["metrics_record_id"]),
    ]
    if phase == "before_commit":
        connection = repo.connection

        def pause():
            channel.send("uncommitted")
            channel.recv()
            return 0

        @contextmanager
        def instrumented():
            with connection() as db:
                db.create_function("pause_publication", 0, pause)
                yield db

        repo.connection = instrumented
        with repo.connection() as db:
            db.execute(
                "CREATE TRIGGER pause_guard BEFORE INSERT ON records "
                "WHEN NEW.kind='forecast_batch' BEGIN SELECT pause_publication(); END"
            )
    else:
        channel.send("ready")
        channel.recv()
    now = datetime.now(UTC)
    try:
        result = repo.publish_once(
            "forecast_batch", "daily",
            {"request_id": "request", "winner": number, "status": "completed"},
            [("forecast", f"forecast-{number}", {"winner": number}, now)],
            available_at=now, recorded_at=now, expected_records=expected,
        )
    except PublicationConflictError as exc:
        channel.send(("rejected", str(exc)))
        channel.recv()
        return
    channel.send(("committed", result))
    channel.recv()  # Simulate a worker remaining alive after commit, before acknowledgment.


def _receive(channel):
    assert channel.poll(TIMEOUT), "Worker did not reach the required boundary"
    return channel.recv()


def _stop(process, channel):
    if process.is_alive():
        process.terminate()
    process.join(timeout=TIMEOUT)
    if process.is_alive():
        process.kill()
        process.join(timeout=TIMEOUT)
    channel.close()
    assert not process.is_alive()


@pytest.fixture
def reviewed_repo(tmp_path):
    repo = Repository(tmp_path / "live.sqlite3")
    now = datetime.now(UTC)
    model = repo.put("model", "m", {"bundle": "fixture"}, available_at=now)
    metrics = repo.put("model_metrics", "m", {"evidence": "fixture"}, available_at=now)
    repo.put(
        "model_release", "m",
        {"status": "approved", "model_record_id": model, "metrics_record_id": metrics},
        available_at=now,
    )
    return repo


def _retry(repo):
    now = datetime.now(UTC)
    return repo.publish_once(
        "forecast_batch", "daily",
        {"request_id": "request", "winner": "retry", "status": "completed"},
        [("forecast", "retry", {"winner": "retry"}, now)],
        available_at=now, recorded_at=now,
    )


def _assert_consistent(repo, winner, count):
    assert repo.health()
    forecasts = repo.scan("forecast")
    runs = repo.scan("forecast_batch")
    assert len(forecasts) == len(runs) == count
    if count:
        assert forecasts[0]["payload"]["winner"] == runs[0]["payload"]["winner"] == winner
    assert require_approved_model(repo, "m", as_of=datetime.now(UTC))["status"] == "approved"


def test_precommit_death_and_backup_exclude_partial_publication(reviewed_repo, tmp_path):
    repo = reviewed_repo
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_publish, args=(str(repo.path), child, "before_commit"))
    worker.start()
    child.close()
    try:
        assert _receive(parent) == "uncommitted"
        # A forecast row has been inserted but the audit guard has not committed.
        # WAL readers and an online backup must see neither member of that transaction.
        _assert_consistent(repo, None, 0)
        destination = tmp_path / "during-publication.sqlite3"
        repo.backup(destination)
        _assert_consistent(Repository(destination), None, 0)
    finally:
        _stop(worker, parent)
    # The instrumentation is a persisted trigger; remove it for normal recovery.
    with repo.connection() as db:
        db.execute("DROP TRIGGER pause_guard")
    _assert_consistent(repo, None, 0)
    assert _retry(repo)["winner"] == "retry"
    _assert_consistent(repo, "retry", 1)


def test_postcommit_death_preserves_retry_identity_through_backup(reviewed_repo, tmp_path):
    repo = reviewed_repo
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_publish, args=(str(repo.path), child, "after_commit"))
    worker.start()
    child.close()
    try:
        assert _receive(parent) == "ready"
        parent.send("go")
        event, result = _receive(parent)
        assert event == "committed"
    finally:
        _stop(worker, parent)
    assert _retry(repo) == result
    _assert_consistent(repo, 0, 1)
    destination = tmp_path / "committed.sqlite3"
    repo.backup(destination)
    restored = Repository(destination)
    assert _retry(restored) == result
    _assert_consistent(restored, 0, 1)


def test_competing_processes_publish_one_consistent_run(reviewed_repo):
    repo = reviewed_repo
    context = multiprocessing.get_context("spawn")
    workers = []
    try:
        for number in range(4):
            parent, child = context.Pipe()
            worker = context.Process(
                target=_publish, args=(str(repo.path), child, "race", number)
            )
            worker.start()
            child.close()
            workers.append((worker, parent))
        for _, channel in workers:
            assert _receive(channel) == "ready"
        for _, channel in workers:
            channel.send("go")
        results = [_receive(channel) for _, channel in workers]
        assert all(event == "committed" for event, _ in results)
        assert all(result == results[0][1] for _, result in results)
        winner = results[0][1]["winner"]
        _assert_consistent(repo, winner, 1)
        assert _retry(repo) == results[0][1]
    finally:
        for worker, channel in workers:
            _stop(worker, channel)



def test_cross_process_revocation_blocks_publication_and_survives_restore(reviewed_repo, tmp_path):
    repo = reviewed_repo
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_publish, args=(str(repo.path), child, "revoked"))
    worker.start()
    child.close()
    try:
        assert _receive(parent) == "ready"
        now = datetime.now(UTC)
        repo.put("model_release", "m", {"status": "revoked"}, available_at=now)
        parent.send("go")
        event, reason = _receive(parent)
        assert event == "rejected"
        assert "model_release/m" in reason
    finally:
        _stop(worker, parent)
    destination = tmp_path / "revoked.sqlite3"
    repo.backup(destination)
    for candidate in (repo, Repository(destination)):
        assert candidate.health()
        assert not candidate.scan("forecast")
        assert not candidate.scan("forecast_batch")
        with pytest.raises(ValueError, match="revoked"):
            require_approved_model(candidate, "m", as_of=datetime.now(UTC))
