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
