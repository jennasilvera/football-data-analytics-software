"""Operational batch publication, retry and failure/recovery guarantees."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from football_analytics.api.main import create_app
from football_analytics.cli import main
from football_analytics.config import Settings
from football_analytics.data import load_canonical_catalogs
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.data.serialization import json_safe
from football_analytics.domain import Match, MatchStatus
from football_analytics.services.batches import forecast_batch
from football_analytics.services.forecasting import train_forecast_bundle
from football_analytics.storage.repository import Repository

SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"


@pytest.fixture(scope="module")
def bundle():
    return train_forecast_bundle(
        build_legacy_mens_results_observations(
            pd.read_csv(SAMPLE / "results.csv"), ingested_at=datetime(2020, 6, 1, tzinfo=UTC)
        ),
        load_canonical_catalogs(
            teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
        ),
        cutoff=datetime(2020, 5, 15, tzinfo=UTC),
        min_calibration=5,
    )


@pytest.fixture
def setup(tmp_path, bundle):
    repo = Repository(tmp_path / "db")
    now = datetime.now(UTC)
    earlier = now - timedelta(seconds=5)
    model_id = bundle["bundle_id"]
    model_record = repo.put("model", model_id, bundle, available_at=earlier, recorded_at=earlier)
    evidence = repo.put("model_metrics", model_id, {}, available_at=earlier, recorded_at=earlier)
    repo.put(
        "model_release",
        model_id,
        {
            "status": "approved",
            "model_record_id": model_record,
            "metrics_record_id": evidence,
        },
        available_at=earlier,
        recorded_at=earlier,
    )

    def fixture(match_id, *, days=1, away="BRA", status=MatchStatus.SCHEDULED, recorded=None):
        timestamp = recorded or earlier
        match = Match(
            match_id,
            (now + timedelta(days=days)).date(),
            "ARG",
            away,
            "friendly",
            True,
            status=status,
        )
        return repo.put(
            "match",
            match_id,
            {"match": json_safe(match)},
            available_at=timestamp,
            recorded_at=timestamp,
        )

    fixture("future")
    options = dict(
        run_id="daily-1",
        model_id=model_id,
        from_date=now.date(),
        through_date=(now + timedelta(days=2)).date(),
        now=now,
        max_model_age_days=5000,
    )
    return repo, options, fixture


def test_batch_audits_skips_snapshot_lineage_and_idempotent_retry(setup, tmp_path):
    repo, args, fixture = setup
    fixture("today", days=0)
    fixture("completed", status=MatchStatus.COMPLETED)
    fixture("cancelled", status=MatchStatus.CANCELLED)
    fixture("outside", days=5)
    fixture("not-yet-known", recorded=args["now"] + timedelta(seconds=1))
    report = forecast_batch(repo, **args)
    assert report["status"] == "completed"
    assert report["counts"] == {"forecasted": 1, "skipped": 3, "failed": 0, "withheld": 0}
    assert len(repo.scan("forecast")) == 1
    assert set(report["lineage"]) == {"model_record_id", "metrics_record_id", "approval_record_id"}
    assert all(row["match_record_id"] for row in report["fixtures"])
    assert not repo.scan("forecast_batch", as_of=args["now"])
    repo.put(
        "model_release",
        args["model_id"],
        {"status": "revoked"},
        available_at=datetime.now(UTC),
        recorded_at=datetime.now(UTC),
    )
    # Retry is historical retrieval, not fresh prediction or approval.
    assert forecast_batch(repo, **args) == report
    assert len(repo.scan("forecast")) == 1
    with pytest.raises(ValueError, match="different request"):
        forecast_batch(repo, **{**args, "method": "class_frequency_v1"})
    backup = tmp_path / "backup"
    repo.backup(backup)
    assert forecast_batch(Repository(backup), **args) == report


def test_one_bad_fixture_withholds_entire_batch_and_retains_failure(setup):
    repo, args, fixture = setup
    fixture("unknown", away="UNKNOWN")
    report = forecast_batch(repo, **args)
    assert report["status"] == "failed"
    assert report["counts"] == {"forecasted": 0, "skipped": 0, "failed": 1, "withheld": 1}
    assert not repo.scan("forecast")
    assert all("forecast_id" not in row for row in report["fixtures"])
    assert repo.get("forecast_batch", args["run_id"])["payload"] == report
    fixture("unknown", recorded=datetime.now(UTC))
    assert forecast_batch(repo, **args) == report
    retry = forecast_batch(repo, **{**args, "run_id": "daily-2", "now": datetime.now(UTC)})
    assert retry["status"] == "completed"
    assert retry["counts"]["forecasted"] == 2


def test_missing_review_stale_models_and_bad_methods_are_durable_failures(setup):
    repo, args, _ = setup
    for run_id, change in (
        ("stale", {"max_model_age_days": 1}),
        ("method", {"method": "does-not-exist"}),
        ("missing-model", {"model_id": "missing"}),
    ):
        report = forecast_batch(repo, **{**args, "run_id": run_id, **change})
        assert report["status"] == "failed"
        assert report["errors"][0]["stage"] == "model_review"
        assert repo.get("forecast_batch", run_id)["payload"] == report
    assert not repo.scan("forecast")


def test_publication_is_withheld_if_fixture_window_closes_during_run(setup, monkeypatch):
    from football_analytics.services import batches

    repo, args, _ = setup

    class LateClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return args["now"] + timedelta(days=3)

    monkeypatch.setattr(batches, "datetime", LateClock)
    report = forecast_batch(repo, **args)
    assert report["status"] == "failed"
    assert report["fixtures"][0]["reason"] == "pre_match_window_closed_during_run"
    assert not repo.scan("forecast")


def test_publication_failure_rolls_back_forecasts_and_guard(setup):
    repo, args, _ = setup
    with repo.connection() as db:
        db.execute(
            "CREATE TRIGGER fail_guard BEFORE INSERT ON records "
            "WHEN NEW.kind='forecast_batch' BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        forecast_batch(repo, **args)
    assert not repo.scan("forecast")
    assert not repo.scan("forecast_batch")
    with repo.connection() as db:
        db.execute("DROP TRIGGER fail_guard")
    assert forecast_batch(repo, **args)["status"] == "completed"


def test_concurrent_requests_publish_only_the_winner_and_its_related_records(tmp_path):
    repo = Repository(tmp_path / "concurrency")
    now = datetime.now(UTC)
    barrier = Barrier(2)

    def publish(number):
        barrier.wait()
        return repo.publish_once(
            "forecast_batch",
            "one-run",
            {
                "request_id": "same-request",
                "winner": number,
            },
            [("forecast", str(number), {"winner": number}, now)],
            available_at=now,
            recorded_at=now,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, (1, 2)))
    assert results[0] == results[1]
    assert len(repo.scan("forecast")) == len(repo.scan("forecast_batch")) == 1
    assert repo.scan("forecast")[0]["payload"]["winner"] == results[0]["winner"]
    with pytest.raises(ValueError, match="different request"):
        repo.publish_once(
            "forecast_batch",
            "one-run",
            {"request_id": "different"},
            [],
            available_at=now,
            recorded_at=now,
        )


def test_cli_failure_exit_and_authenticated_audit_routes(setup, capsys):
    repo, args, _ = setup
    cli = [
        "predict-batch",
        "--database",
        str(repo.path),
        "--model-id",
        args["model_id"],
        "--run-id",
        "bad-cli-run",
        "--from-date",
        args["from_date"].isoformat(),
        "--through-date",
        args["through_date"].isoformat(),
        "--max-model-age-days",
        "1",
    ]
    with pytest.raises(SystemExit) as exc:
        main(cli)
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "failed"
    assert "no forecasts published" in output.err
    main(["batch-report", "--database", str(repo.path), "--run-id", "bad-cli-run"])
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    client = TestClient(create_app(Settings(repo.path, "batch-tests-secret-more-than-24")))
    assert client.get("/forecast-batches").status_code == 401
    client.headers["Authorization"] = "Bearer batch-tests-secret-more-than-24"
    response = client.get("/forecast-batches?limit=1")
    assert response.status_code == 200
    assert response.json()["items"][0]["run_id"] == "bad-cli-run"
    assert client.get("/forecast-batches/bad-cli-run").json()["status"] == "failed"
    assert client.get("/forecast-batches/missing").status_code == 404


def test_deadline_checks_after_lock_roll_back_and_preserve_failure_audit(setup, monkeypatch):
    from football_analytics.storage import repository

    repo, args, _ = setup

    class PublicationClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return args["now"] + timedelta(days=3)

    monkeypatch.setattr(repository, "datetime", PublicationClock)
    report = forecast_batch(repo, **args)
    assert report["status"] == "failed"
    assert report["errors"][0]["stage"] == "publication"
    assert report["counts"]["withheld"] == 1
    assert not repo.scan("forecast")
    assert repo.get("forecast_batch", args["run_id"])["payload"] == report


def test_empty_batches_are_successful_auditable_results(setup):
    repo, args, _ = setup
    report = forecast_batch(repo, **{**args, "from_date": args["through_date"]})
    assert report["status"] == "completed"
    assert report["counts"] == {"forecasted": 0, "skipped": 0, "failed": 0, "withheld": 0}
    assert forecast_batch(repo, **{**args, "from_date": args["through_date"]}) == report


def test_publication_checks_integrity_of_a_racing_existing_guard(tmp_path):
    repo = Repository(tmp_path / "db")
    now = datetime.now(UTC)
    repo.publish_once(
        "forecast_batch", "r", {"request_id": "request"}, [], available_at=now, recorded_at=now
    )
    with repo.connection() as db:
        db.execute("UPDATE records SET payload='{}'")
    with pytest.raises(ValueError, match="integrity"):
        repo.publish_once(
            "forecast_batch", "r", {"request_id": "request"}, [], available_at=now, recorded_at=now
        )


def test_cli_storage_error_is_distinct_from_durable_batch_failure(setup, monkeypatch, capsys):
    repo, args, _ = setup

    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(Repository, "publish_once", unavailable)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "predict-batch",
                "--database",
                str(repo.path),
                "--model-id",
                args["model_id"],
                "--run-id",
                "storage-failure",
                "--from-date",
                args["from_date"].isoformat(),
                "--through-date",
                args["through_date"].isoformat(),
                "--max-model-age-days",
                "5000",
            ]
        )
    assert exc.value.code == 2
    assert "database is locked" in capsys.readouterr().err
    assert not repo.scan("forecast_batch")
    assert not repo.scan("forecast")


@pytest.mark.parametrize("kind", ["model", "model_metrics", "model_release", "match"])
def test_changed_publication_inputs_withhold_batch_and_retry_keeps_audit(setup, monkeypatch, kind):
    repo, args, _ = setup
    original_publish = repo.publish_once
    changed = False

    def publish_after_change(*positional, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            entity = "future" if kind == "match" else args["model_id"]
            payload = repo.get(kind, entity)["payload"]
            if kind == "model_release":
                payload["status"] = "revoked"
            elif kind == "match":
                payload["match"]["status"] = "cancelled"
            else:
                payload["revision_note"] = "replaced while computing"
            timestamp = datetime.now(UTC)
            repo.put(kind, entity, payload, available_at=timestamp, recorded_at=timestamp)
        return original_publish(*positional, **kwargs)

    monkeypatch.setattr(repo, "publish_once", publish_after_change)
    report = forecast_batch(repo, **args)
    assert report["status"] == "failed"
    assert report["counts"]["withheld"] == 1
    assert report["counts"]["forecasted"] == 0
    assert report["fixtures"][0]["reason"] == "publication_input_changed"
    assert report["errors"][0]["stage"] == "publication"
    assert kind in report["errors"][0]["reason"]
    assert not repo.scan("forecast")
    assert forecast_batch(repo, **args) == report
    assert repo.get("forecast_batch", args["run_id"])["payload"] == report


def test_retry_returns_winning_publication_even_when_input_revisions_changed(tmp_path):
    repo = Repository(tmp_path / "db")
    now = datetime.now(UTC)
    expected = repo.put("model", "m", {}, available_at=now, recorded_at=now)
    args = dict(
        available_at=now, recorded_at=now, expected_records=[("model", "m", expected)]
    )
    winner = repo.publish_once("forecast_batch", "r", {"request_id": "same"}, [], **args)
    later = datetime.now(UTC)
    repo.put("model", "m", {"changed": True}, available_at=later, recorded_at=later)
    assert repo.publish_once("forecast_batch", "r", {"request_id": "same"}, [], **args) == winner


def test_publication_rechecks_input_after_waiting_for_an_ingestion_lock(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from threading import Event

    from football_analytics.storage.repository import PublicationConflictError

    repo = Repository(tmp_path / "lock-race")
    now = datetime.now(UTC)
    expected = repo.put("model", "m", {}, available_at=now, recorded_at=now)
    connection = repo.connection
    opened = Event()

    @contextmanager
    def observed_connection():
        with connection() as db:
            opened.set()
            yield db

    def publish():
        return repo.publish_once(
            "forecast_batch", "r", {"request_id": "same"},
            [("forecast", "f", {}, now)],
            available_at=now, recorded_at=now,
            expected_records=[("model", "m", expected)],
        )

    # Hold a real write transaction while a second connection tries to publish.
    with ThreadPoolExecutor(max_workers=1) as pool:
        with connection() as writer:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("DELETE FROM records WHERE kind='model'")
            monkeypatch.setattr(repo, "connection", observed_connection)
            pending = pool.submit(publish)
            assert opened.wait(timeout=5)
        # The deletion becomes visible only when the writer releases its lock.
        with pytest.raises(PublicationConflictError, match="model/m"):
            pending.result(timeout=5)
    assert not repo.scan("forecast")
    assert not repo.scan("forecast_batch")


@pytest.mark.parametrize("kind", ["model", "model_metrics", "model_release", "match"])
def test_single_forecast_rejects_replaced_inputs(setup, monkeypatch, kind):
    from football_analytics.services import serving
    from football_analytics.storage.repository import PublicationConflictError

    repo, args, _ = setup
    compute = serving.forecast_match

    def replace_input(*positional, **kwargs):
        result = compute(*positional, **kwargs)
        entity = "future" if kind == "match" else args["model_id"]
        payload = repo.get(kind, entity)["payload"]
        payload["revision_note"] = "concurrent replacement"
        timestamp = datetime.now(UTC)
        repo.put(kind, entity, payload, available_at=timestamp, recorded_at=timestamp)
        return result

    monkeypatch.setattr(serving, "forecast_match", replace_input)
    with pytest.raises(PublicationConflictError, match=kind):
        serving.predict_and_publish(
            repo, model_id=args["model_id"], match_id="future",
            now=args["now"], max_model_age_days=5000,
        )
    assert not repo.scan("forecast")


def test_single_forecast_publication_uses_completion_time_and_snapshot_fixture(setup):
    from football_analytics.services.serving import predict_and_publish

    repo, args, fixture = setup
    # A future revision must not replace the fixture visible at the request snapshot.
    fixture("future", away="UNKNOWN", recorded=args["now"] + timedelta(days=1))
    result = predict_and_publish(
        repo, model_id=args["model_id"], match_id="future",
        now=args["now"], max_model_age_days=5000,
    )
    stored = repo.get("forecast", result["forecast_id"])
    assert stored["payload"] == json_safe(result)
    assert datetime.fromisoformat(stored["available_at"]) > args["now"]
    assert not repo.scan("forecast", as_of=args["now"])


@pytest.mark.parametrize("interface", ["api", "cli"])
def test_single_forecast_interfaces_reject_publication_after_kickoff(
    setup, monkeypatch, capsys, interface
):
    from football_analytics.storage import repository

    repo, args, _ = setup

    class LateClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return args["now"] + timedelta(days=3)

    monkeypatch.setattr(repository, "datetime", LateClock)
    if interface == "api":
        key = "single-forecast-tests-more-than-24"
        client = TestClient(create_app(Settings(repo.path, key, max_model_age_days=5000)))
        response = client.post(
            "/predict", headers={"Authorization": "Bearer " + key},
            json={"model_id": args["model_id"], "match_id": "future"},
        )
        assert response.status_code == 409
        assert "deadline" in response.json()["detail"]
    else:
        with pytest.raises(SystemExit) as exc:
            main([
                "predict", "--database", str(repo.path), "--model-id", args["model_id"],
                "--match-id", "future", "--max-model-age-days", "5000",
            ])
        assert exc.value.code == 2
        assert "deadline" in capsys.readouterr().err
    assert not repo.scan("forecast")


def test_single_forecast_rejects_mismatched_stored_match_identity(setup):
    from football_analytics.services.serving import predict_and_publish

    repo, args, _ = setup
    source = repo.get("match", "future")["payload"]
    source["match"]["match_id"] = "different"
    now = datetime.now(UTC)
    repo.put("match", "future", source, available_at=now, recorded_at=now)
    with pytest.raises(ValueError, match="identity"):
        predict_and_publish(
            repo, model_id=args["model_id"], match_id="future", now=now,
            max_model_age_days=5000,
        )
    assert not repo.scan("forecast")


def test_guarded_put_many_rolls_back_if_deadline_closes_before_commit(tmp_path, monkeypatch):
    from football_analytics.storage import repository

    repo = Repository(tmp_path / "deadline")
    now = datetime.now(UTC)
    calls = 0

    class ClosingClock(datetime):
        @classmethod
        def now(cls, tz=None):
            nonlocal calls
            calls += 1
            return now + timedelta(seconds=2 if calls >= 3 else 0)

    monkeypatch.setattr(repository, "datetime", ClosingClock)
    with pytest.raises(repository.PublicationDeadlineError):
        repo.put_many(
            [("forecast", "a", {}, now), ("forecast", "b", {}, now)],
            recorded_at=now, deadline=now + timedelta(seconds=1),
        )
    assert not repo.scan("forecast")


@pytest.mark.parametrize("mode", ["single", "batch"])
@pytest.mark.parametrize("replacement_status", ["revoked", "approved"])
def test_publication_guards_the_exact_validated_approval(
    setup, monkeypatch, mode, replacement_status
):
    from football_analytics.services.serving import predict_and_publish
    from football_analytics.storage.repository import PublicationConflictError

    repo, args, _ = setup
    original_get = repo.get
    replaced = False

    def racing_get(kind, entity_id, **kwargs):
        nonlocal replaced
        record = original_get(kind, entity_id, **kwargs)
        if kind == "model_release" and not replaced:
            replaced = True
            # A second connection commits after the validated read but before
            # the caller can fetch the approval again for its publication guard.
            writer = Repository(repo.path)
            writer.put(
                kind, entity_id,
                {**record["payload"], "status": replacement_status, "reason": "new decision"},
                available_at=args["now"], recorded_at=args["now"],
            )
        return record

    monkeypatch.setattr(repo, "get", racing_get)
    if mode == "single":
        with pytest.raises(PublicationConflictError, match="model_release"):
            predict_and_publish(
                repo, model_id=args["model_id"], match_id="future", now=args["now"],
                max_model_age_days=5000,
            )
        assert repo.scan("forecast_batch") == []
    else:
        report = forecast_batch(repo, **args)
        assert report["status"] == "failed"
        assert report["counts"]["withheld"] == 1
        assert report["errors"][-1]["stage"] == "publication"
        assert "model_release" in report["errors"][-1]["reason"]
        assert forecast_batch(repo, **args) == report
    assert replaced
    assert repo.scan("forecast") == []
