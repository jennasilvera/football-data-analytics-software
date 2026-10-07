"""End-to-end native platform acceptance with explicitly synthetic, date-shifted data."""

import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from football_analytics.api.main import create_app
from football_analytics.config import Settings
from football_analytics.storage.repository import Repository

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/sample/v2"


def main():
    with tempfile.TemporaryDirectory(prefix="football-platform-") as directory:
        work = Path(directory)
        now = datetime.now(UTC)
        shift = now.date() - datetime(2020, 5, 15).date()
        frame = pd.read_csv(SAMPLE / "results.csv")
        frame["date"] = (pd.to_datetime(frame["date"]) + pd.Timedelta(shift)).dt.strftime(
            "%Y-%m-%d"
        )
        results = work / "results.csv"
        frame.to_csv(results, index=False)
        database = work / "platform.sqlite3"
        shared = [
            "--results",
            str(results),
            "--teams",
            str(SAMPLE / "teams.json"),
            "--competitions",
            str(SAMPLE / "competitions.json"),
            "--source-id",
            "synthetic-platform-acceptance",
            "--assert-senior-mens-a",
        ]

        def command(name, *args, research=False):
            options = [] if research else ["--database", str(database)]
            output = subprocess.check_output(
                [sys.executable, "-m", "football_analytics", name, *options, *args],
                cwd=ROOT,
                text=True,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            )
            return json.loads(output)

        legal = ["--legal-use-notes", "Repository synthetic acceptance data; no performance claim"]
        command("validate-data", *shared, *legal)
        command("ingest-results", *shared, *legal)
        research = command(
            "research",
            *shared,
            "--ingested-at",
            now.isoformat(),
            "--cutoff",
            (datetime(2020, 3, 1, tzinfo=UTC) + shift).isoformat(),
            "--cutoff",
            (datetime(2020, 4, 1, tzinfo=UTC) + shift).isoformat(),
            "--evaluation-days",
            "30",
            "--min-train",
            "12",
            "--model",
            "all",
            "--postprocess-days",
            "21",
            "--min-postprocess",
            "5",
            "--paired-uncertainty",
            "--output",
            str(work / "research"),
            research=True,
        )
        trained = command(
            "train",
            *shared,
            *legal,
            "--cutoff",
            now.isoformat(),
            "--historical-availability-assumed",
            "--min-calibration",
            "5",
        )
        model_id = trained["model_id"]
        command("import-research", "--model-id", model_id, "--input", research["report"])
        command(
            "approve-model",
            "--model-id",
            model_id,
            "--reason",
            "Synthetic integration acceptance only; not production approval",
        )
        fixture = work / "fixtures.csv"
        pd.DataFrame(
            [
                {
                    "date": (now + timedelta(days=2)).date().isoformat(),
                    "home_team": "Argentina",
                    "away_team": "Brazil",
                    "tournament": "International Friendly",
                    "neutral": True,
                }
            ]
        ).to_csv(fixture, index=False)
        fixture_args = shared.copy()
        fixture_args[1] = str(fixture)
        command("ingest-fixtures", *fixture_args, *legal)
        repo = Repository(database)
        match_id = next(
            r["entity_id"]
            for r in repo.scan("match")
            if r["payload"]["match"]["status"] == "scheduled"
        )
        command("build-features", "--match-id", match_id)
        forecast = command("predict", "--model-id", model_id, "--match-id", match_id)
        assert abs(sum(forecast["probabilities"].values()) - 1) < 1e-12
        command("update-ratings")
        report = command("team-report", "--team", "ARG")
        assert report["next_match_outlook"]["match_id"] == match_id
        command("match-report", "--match-id", match_id)
        command("model-card", "--model-id", model_id)
        key = "synthetic-acceptance-key-not-for-deployment"
        with TestClient(create_app(Settings(database, key))) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/teams").status_code == 401
            client.headers["Authorization"] = "Bearer " + key
            assert (
                client.post(
                    "/predict", json={"model_id": model_id, "match_id": match_id}
                ).status_code
                == 200
            )
            assert client.get(f"/models/{model_id}/metrics").status_code == 200
        backup = work / "recovery.sqlite3"
        command("backup", "--output", str(backup))
        restored = Repository(backup)
        assert restored.scan("forecast") == repo.scan("forecast")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "synthetic": True,
                    "models": len(repo.scan("model")),
                    "forecasts": len(repo.scan("forecast")),
                    "recovery_verified": True,
                }
            )
        )


if __name__ == "__main__":
    main()
