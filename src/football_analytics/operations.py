"""Operational CLI commands kept separate from research orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from football_analytics.data import load_canonical_catalogs, normalize_match_batch
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.data.adapters.tabular import (
    TabularMatchColumns,
    TabularSourcePolicy,
    build_match_observations,
)
from football_analytics.data.contracts import LeakageRisk
from football_analytics.data.scope import GenderCategory, TeamLevel
from football_analytics.data.serialization import json_safe, match_from_dict
from football_analytics.domain import MatchStatus
from football_analytics.domain.scores import ScoreBasis
from football_analytics.models.postprocessing import content_id
from football_analytics.services.forecasting import train_forecast_bundle
from football_analytics.services.intelligence import ratings, team_report
from football_analytics.storage.files import publish_immutable
from football_analytics.storage.repository import Repository

COMMANDS = {
    "ingest-results",
    "ingest-fixtures",
    "ingest-rankings",
    "ingest-odds",
    "validate-data",
    "train",
    "predict",
    "predict-batch",
    "batch-report",
    "update-ratings",
    "team-report",
    "match-report",
    "model-card",
    "approve-model",
    "revoke-model",
    "backup",
    "serve",
    "dashboard",
    "build-features",
    "import-research",
    "drift-report",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="football-analytics")
    commands = p.add_subparsers(dest="command", required=True)
    for name in sorted(COMMANDS):
        c = commands.add_parser(name)
        c.add_argument(
            "--database",
            type=Path,
            default=Path(os.getenv("FOOTBALL_DATABASE", "outputs/platform.sqlite3")),
        )
        if name in ("ingest-results", "ingest-fixtures", "validate-data", "train"):
            c.add_argument("--results", type=Path, required=True)
            c.add_argument("--teams", type=Path, required=True)
            c.add_argument("--competitions", type=Path, required=True)
            c.add_argument("--source-id", required=True)
            c.add_argument("--assert-senior-mens-a", action="store_true", required=True)
            c.add_argument(
                "--score-basis", choices=[b.value for b in ScoreBasis], default="unknown"
            )
            c.add_argument("--legal-use-notes", required=True)
        if name == "drift-report":
            c.add_argument("--reference", type=Path, required=True)
            c.add_argument("--current", type=Path, required=True)
        if name == "train":
            c.add_argument("--cutoff", type=datetime.fromisoformat, required=True)
            c.add_argument("--historical-availability-assumed", action="store_true", required=True)
            c.add_argument("--holdout-days", type=int, default=30)
            c.add_argument("--min-calibration", type=int, default=30)
            c.add_argument("--uncalibrated", action="store_true")
            c.add_argument("--feature-groups")
            c.add_argument("--feature-context", type=Path)
        if name in ("ingest-rankings", "ingest-odds", "import-research"):
            c.add_argument("--input", type=Path, required=True)
        if name in (
            "predict", "predict-batch", "approve-model", "revoke-model", "model-card",
            "import-research",
        ):
            c.add_argument("--model-id", required=True)
        if name in ("predict", "match-report", "build-features"):
            c.add_argument("--match-id", required=True)
        if name in ("predict-batch", "batch-report"):
            c.add_argument("--run-id", required=True)
        if name == "predict-batch":
            c.add_argument("--from-date", type=date.fromisoformat, required=True)
            c.add_argument("--through-date", type=date.fromisoformat, required=True)
        if name in ("predict", "predict-batch"):
            c.add_argument("--method", default="ensemble")
            c.add_argument(
                "--max-model-age-days",
                type=int,
                default=int(os.getenv("FOOTBALL_MAX_MODEL_AGE_DAYS", "90")),
            )
        if name in ("approve-model", "revoke-model"):
            c.add_argument("--reason", required=True)
        if name == "team-report":
            c.add_argument("--team", required=True, help="Canonical team ID")
        if name == "backup":
            c.add_argument("--output", type=Path, required=True)
        if name in ("serve", "dashboard"):
            c.add_argument("--host", default="127.0.0.1")
            c.add_argument("--port", type=int, default=8000 if name == "serve" else 8501)
        if name not in ("serve", "dashboard", "backup"):
            c.add_argument("--output", type=Path)
    return p


def run(argv: list[str]) -> None:
    p = parser()
    args = p.parse_args(argv)
    try:
        repo = Repository(args.database)
        now = datetime.now(UTC)
        result = execute(args, repo, now)
        if result is not None:
            encoded = (
                json.dumps(json_safe(result), sort_keys=True, indent=2, allow_nan=False) + "\n"
            ).encode()
            if getattr(args, "output", None) and args.command != "backup":
                publish_immutable(args.output, encoded)
            print(encoded.decode())
            if args.command == "predict-batch" and result.get("status") == "failed":
                p.exit(1, "Batch failed; audit retained and no forecasts published.\n")
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        p.exit(2, f"Operation failed: {exc}\n")


def execute(args, repo: Repository, now: datetime):
    command = args.command
    if command in ("ingest-results", "ingest-fixtures", "validate-data", "train"):
        catalogs = load_canonical_catalogs(
            teams_path=args.teams, competitions_path=args.competitions
        )
        raw = args.results.read_bytes()
        import io

        frame = pd.read_csv(io.BytesIO(raw))
        digest = hashlib.sha256(raw).hexdigest()
        if command == "ingest-fixtures":
            observations = build_match_observations(
                frame,
                columns=TabularMatchColumns(
                    "date",
                    "home_team",
                    "away_team",
                    "tournament",
                    "neutral",
                    kickoff_at="kickoff_at" if "kickoff_at" in frame else None,
                    source_match_id="source_match_id" if "source_match_id" in frame else None,
                ),
                policy=TabularSourcePolicy(
                    args.source_id,
                    GenderCategory.MEN,
                    TeamLevel.SENIOR_A,
                    True,
                    MatchStatus.SCHEDULED,
                    LeakageRisk.SAFE,
                    source_version=digest,
                    legal_use_notes=args.legal_use_notes,
                ),
                ingested_at=now,
            )
        else:
            observations = build_legacy_mens_results_observations(
                frame,
                ingested_at=now,
                source_id=args.source_id,
                source_version=digest,
                legal_use_notes=args.legal_use_notes,
                score_basis=ScoreBasis(args.score_basis),
            )
        normalized = normalize_match_batch(
            observations,
            team_resolver=catalogs.team_resolver(),
            competition_resolver=catalogs.competition_resolver(),
        )
        if normalized.excluded or normalized.quarantined:
            raise ValueError(
                f"Ingestion refused: {len(normalized.excluded)} excluded and "
                f"{len(normalized.quarantined)} quarantined observations"
            )
        if command == "validate-data":
            return {"valid": True, "matches": len(normalized.normalized), "source_sha256": digest}
        if command == "train":
            bundle = train_forecast_bundle(
                observations,
                catalogs,
                cutoff=args.cutoff,
                holdout_days=args.holdout_days,
                min_calibration=args.min_calibration,
                calibrate=not args.uncalibrated,
                feature_groups=tuple(args.feature_groups.split(","))
                if args.feature_groups
                else None,
                feature_context=json.loads(args.feature_context.read_text())
                if args.feature_context
                else None,
            )
            repo.put("model", bundle["bundle_id"], bundle, available_at=now, recorded_at=now)
            repo.put(
                "model_release",
                bundle["bundle_id"],
                {"status": "candidate"},
                available_at=now,
                recorded_at=now,
            )
            return {
                "model_id": bundle["bundle_id"],
                "status": "candidate",
                "training_matches": len(bundle["records"]),
                "calibrated": bool(bundle["transforms"]),
            }
        existing = {
            (
                row["payload"]["match"]["match_date"],
                tuple(
                    sorted(
                        (
                            row["payload"]["match"]["home_team_id"],
                            row["payload"]["match"]["away_team_id"],
                        )
                    )
                ),
                row["payload"]["match"]["competition_id"],
            ): row["entity_id"]
            for row in repo.scan("match")
        }
        for record in normalized.normalized:
            key = (
                record.match.match_date.isoformat(),
                tuple(sorted((record.match.home_team_id, record.match.away_team_id))),
                record.match.competition_id,
            )
            if key in existing and existing[key] != record.match.match_id:
                raise ValueError("Cross-import reversed duplicate fixture.")
        items = [("team", team.team_id, json_safe(team), now) for team in catalogs.teams]
        items.extend(
            ("competition", c.competition_id, json_safe(c), now) for c in catalogs.competitions
        )
        items.extend(("match", r.match.match_id, json_safe(r), now) for r in normalized.normalized)
        repo.put_many(items, recorded_at=now)
        return {
            "ingested": len(normalized.normalized),
            "recorded_at": now.isoformat(),
            "source_sha256": digest,
        }
    if command in ("ingest-rankings", "ingest-odds"):
        from football_analytics.data.serialization import metadata_from_dict
        from football_analytics.features import FifaRankingObservation, MarketSnapshotObservation

        rows = json.loads(args.input.read_text())
        if not isinstance(rows, list):
            raise ValueError("Observation JSON must contain an array.")
        validated = []
        for row in rows:
            data = dict(row)
            basis = data.pop("score_basis", "unknown")
            if command == "ingest-odds" and basis != "regulation_time":
                raise ValueError("Odds require explicit regulation_time settlement basis.")
            data["metadata"] = metadata_from_dict(data["metadata"])
            cls = (
                FifaRankingObservation
                if command == "ingest-rankings"
                else MarketSnapshotObservation
            )
            observation = cls(**data)
            if (
                observation.metadata.available_at is None
                or not observation.metadata.legal_use_notes
            ):
                raise ValueError("Observation needs publication time and legal-use notes.")
            if observation.metadata.available_at > now:
                raise ValueError("Future observation publication time.")
            payload = json_safe(observation)
            if command == "ingest-odds":
                payload["score_basis"] = basis
            validated.append(payload)
        kind = "ranking" if command == "ingest-rankings" else "odds"
        repo.put_many(
            [
                (
                    kind,
                    content_id(kind + "_", row),
                    row,
                    datetime.fromisoformat(row["metadata"]["available_at"]),
                )
                for row in validated
            ],
            recorded_at=now,
        )
        return {"ingested": len(validated), "kind": kind}
    if command == "import-research":
        report = json.loads(args.input.read_text())
        model = repo.get("model", args.model_id, as_of=now)
        release_rows = repo.list("model_release", entity_id=args.model_id, as_of=now)
        metrics_rows = repo.list("model_metrics", entity_id=args.model_id, as_of=now)
        expected: list[tuple[str, str, str | None]] = [
            ("model", args.model_id, model["record_id"]),
            ("model_release", args.model_id,
             release_rows[0]["record_id"] if release_rows else None),
            ("model_metrics", args.model_id,
             metrics_rows[0]["record_id"] if metrics_rows else None),
        ]
        from football_analytics.services.governance import validate_research

        validate_research(report, model["payload"])
        repo.put_many(
            [
                ("model_metrics", args.model_id, report, now),
                ("model_release", args.model_id, {
                    "status": "candidate", "reason": "Research evidence imported; review required",
                }, now),
            ],
            expected_records=expected,
        )
        return {"model_id": args.model_id, "linked_research_runs": len(report["runs"])}
    if command in ("approve-model", "revoke-model"):
        model = repo.get("model", args.model_id, as_of=now)
        if not args.reason.strip():
            raise ValueError("Model review requires a reason.")
        release = {
            "status": "revoked", "reason": args.reason,
            "reviewed_at": now.isoformat(), "automatic_promotion": False,
        }
        if command == "approve-model":
            from football_analytics.services.governance import validate_research

            metrics = repo.get("model_metrics", args.model_id, as_of=now)
            previous = repo.list("model_release", entity_id=args.model_id, as_of=now)
            expected = [
                ("model", args.model_id, model["record_id"]),
                ("model_metrics", args.model_id, metrics["record_id"]),
                ("model_release", args.model_id, previous[0]["record_id"] if previous else None),
            ]
            validate_research(metrics["payload"], model["payload"])
            release.update(
                status="approved", metrics_record_id=metrics["record_id"],
                model_record_id=model["record_id"],
            )
        repo.put_many(
            [("model_release", args.model_id, release, now)],
            expected_records=expected if command == "approve-model" else None,
        )
        return {"model_id": args.model_id, **release}
    if command == "batch-report":
        return repo.get("forecast_batch", args.run_id)["payload"]
    if command == "predict-batch":
        from football_analytics.services.batches import forecast_batch

        return forecast_batch(
            repo, run_id=args.run_id, model_id=args.model_id,
            from_date=args.from_date, through_date=args.through_date, now=now,
            method=args.method, max_model_age_days=args.max_model_age_days,
        )
    if command == "predict":
        from football_analytics.services.serving import predict_and_publish

        return predict_and_publish(
            repo, model_id=args.model_id, match_id=args.match_id, now=now,
            method=args.method, max_model_age_days=args.max_model_age_days,
        )
    if command == "update-ratings":
        result = ratings(repo, now)
        for row in result:
            repo.put("rating", row["team_id"], row, available_at=now, recorded_at=now)
        return result
    if command == "team-report":
        return team_report(repo, args.team, now)
    if command == "match-report":
        return {
            "match": repo.get("match", args.match_id)["payload"],
            "forecasts": [
                r["payload"]
                for r in repo.scan("forecast")
                if r["payload"]["match"]["match_id"] == args.match_id
            ],
        }
    if command == "model-card":
        bundle = repo.get("model", args.model_id, as_of=now)["payload"]
        return {
            "model_id": args.model_id,
            "training_cutoff": bundle["training_cutoff"],
            "intended_use": "senior men's regulation-time research forecasts",
            "non_intended_use": "guaranteed outcomes or unvalidated financial decisions",
            "history_policy": bundle["history_policy"],
            "calibrated": bool(bundle["transforms"]),
            "availability_warning": bundle["availability_warning"],
            "source_ids": sorted({r["metadata"]["source"] for r in bundle["records"]}),
            "limits": [
                "Frozen training-history snapshot",
                "Probability interval coverage unvalidated",
                "Observed-source revisions not reconstructed for retrospective research",
            ],
            "release": repo.get("model_release", args.model_id)["payload"],
        }
    if command == "backup":
        repo.backup(args.output)
        return {"backup": str(args.output), "verified": True}
    if command == "build-features":
        from football_analytics.features import (
            PredictionContext,
            RollingFormFeatureProvider,
            build_feature_vector,
            materialize_feature_vector,
        )
        from football_analytics.services.forecasting import form_policy
        from football_analytics.services.intelligence import available_records

        source = repo.get("match", args.match_id)["payload"]
        provider = RollingFormFeatureProvider(available_records(repo, now))
        vector = build_feature_vector(
            PredictionContext(match_from_dict(source["match"]), now), [provider]
        )
        feature_result = {
            "vector": json_safe(vector),
            "materialized": json_safe(
                materialize_feature_vector(vector, policy=form_policy(provider))
            ),
        }
        identity = content_id("feature_", feature_result)
        repo.put("feature", identity, feature_result, available_at=now, recorded_at=now)
        return {"feature_id": identity, **feature_result}
    if command == "drift-report":
        from football_analytics.evaluation.drift import feature_drift
        from football_analytics.features.materialization import (
            AppliedImputation,
            MaterializedFeatureRow,
        )

        def rows(path):
            sample = json.loads(path.read_text())
            return [
                MaterializedFeatureRow(
                    match_id=r["match_id"],
                    prediction_time_iso=r["prediction_time_iso"],
                    feature_set_id=r["feature_set_id"],
                    imputation_policy_id=r["imputation_policy_id"],
                    columns=tuple((name, value) for name, value in r["columns"]),
                    applied_imputations=tuple(
                        AppliedImputation(**v) for v in r["applied_imputations"]
                    ),
                )
                for r in sample
            ]

        return feature_drift(rows(args.reference), rows(args.current))
    if command == "serve":
        from dataclasses import replace

        import uvicorn

        from football_analytics.api.main import create_app
        from football_analytics.config import Settings
        from football_analytics.logging_config import configure_logging

        configure_logging()
        uvicorn.run(
            create_app(replace(Settings.from_env(), database=args.database)),
            host=args.host,
            port=args.port,
        )
        return None
    if command == "dashboard":
        import subprocess

        subprocess.run(
            [
                "streamlit",
                "run",
                str(Path(__file__).with_name("dashboard.py")),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
                "--server.address",
                args.host,
                "--server.port",
                str(args.port),
            ],
            check=True,
        )
        return None
    raise ValueError("Unsupported operation.")
