"""Snapshot-based, atomic operational forecast batches with durable retry results."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any

from football_analytics.data.contracts import ensure_utc
from football_analytics.data.serialization import match_from_dict
from football_analytics.domain import MatchStatus
from football_analytics.models.postprocessing import content_id
from football_analytics.services.forecasting import forecast_match
from football_analytics.services.governance import require_approved_model
from football_analytics.services.market import attach_market
from football_analytics.storage.repository import PublicationDeadlineError, Repository


def forecast_batch(
    repo: Repository,
    *,
    run_id: str,
    model_id: str,
    from_date: date,
    through_date: date,
    now: datetime,
    method: str = "ensemble",
    max_model_age_days: int = 90,
) -> dict[str, Any]:
    """Publish all eligible forecasts or just a failure audit; never a partial batch."""
    now = ensure_utc(now)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", run_id):
        raise ValueError(
            "Run ID must be 1..128 letters, digits, dots, colons, underscores or hyphens."
        )
    if through_date < from_date or type(max_model_age_days) is not int or max_model_age_days <= 0:
        raise ValueError("Declare ordered UTC dates and a positive model age limit.")
    if not model_id.strip() or not method.strip():
        raise ValueError("Model ID and forecast method must not be blank.")
    request = {
        "schema_version": 1,
        "model_id": model_id,
        "method": method,
        "from_date": from_date.isoformat(),
        "through_date": through_date.isoformat(),
        "max_model_age_days": max_model_age_days,
    }
    request_id = content_id("forecast_batch_request_", request)
    try:
        stored: dict[str, Any] = repo.get("forecast_batch", run_id)["payload"]
    except KeyError:
        pass
    else:
        if stored.get("request_id") != request_id:
            raise ValueError("Run ID already belongs to a different request.")
        return stored
    report: dict[str, Any] = {
        "run_id": run_id,
        "request_id": request_id,
        "request": request,
        "snapshot_at": now.isoformat(),
        "status": "completed",
        "lineage": {},
        "fixtures": [],
        "errors": [],
    }
    forecasts = []
    matches = []
    stage = "model_review"
    try:
        release = require_approved_model(repo, model_id, as_of=now)
        model = repo.get("model", model_id, as_of=now)["payload"]
        report["lineage"] = {
            "model_record_id": release["model_record_id"],
            "metrics_record_id": release["metrics_record_id"],
            "approval_record_id": repo.get("model_release", model_id, as_of=now)["record_id"],
        }
        age = (now - datetime.fromisoformat(model["training_cutoff"])).total_seconds() / 86400
        if age < 0 or age > max_model_age_days:
            raise ValueError("Model history exceeds configured freshness bounds.")
        available_methods = {*model["classifiers"], "independent_poisson_v1"}
        for transform in model["transforms"]:
            if transform["method"] == "convex_ensemble":
                available_methods.add("ensemble")
            elif transform["method"] == "temperature":
                available_methods.add("calibrated_" + transform["members"][0]["spec_id"])
        if method not in available_methods:
            raise ValueError("Requested forecast method is unavailable in this bundle.")
        stage = "fixture_selection"
        selected = [
            row
            for row in repo.scan("match", as_of=now)
            if from_date
            <= date.fromisoformat(row["payload"]["match"]["match_date"])
            <= through_date
        ]
        if len(selected) > 1000:
            raise ValueError("Batch contains more than 1000 fixtures; use a smaller date window.")
        stage = "forecast"
        for source in selected:
            row = {"match_id": source["entity_id"], "match_record_id": source["record_id"]}
            try:
                match = match_from_dict(source["payload"]["match"])
                if match.match_id != source["entity_id"]:
                    raise ValueError("Stored match identity disagrees with its payload.")
                if match.status is not MatchStatus.SCHEDULED:
                    row.update(status="skipped", reason="not_scheduled")
                elif (match.kickoff_at is not None and match.kickoff_at <= now) or (
                    match.kickoff_at is None and match.match_date <= now.date()
                ):
                    row.update(status="skipped", reason="pre_match_window_closed")
                else:
                    forecast = attach_market(
                        repo,
                        forecast_match(
                            model,
                            match,
                            prediction_time=now,
                            method=method,
                        ),
                        now,
                    )
                    forecasts.append(forecast)
                    matches.append(match)
                    row.update(status="forecasted", forecast_id=forecast["forecast_id"])
            except (ValueError, KeyError, TypeError) as exc:
                row.update(status="failed", reason=str(exc), error_type=type(exc).__name__)
            report["fixtures"].append(row)
    except (ValueError, KeyError, TypeError) as exc:
        report["errors"].append(
            {"stage": stage, "reason": str(exc), "error_type": type(exc).__name__}
        )
    finished = max(now, datetime.now(UTC))
    # A batch computed before kickoff must also finish before publishing its forecasts.
    late = {
        m.match_id
        for m in matches
        if (m.kickoff_at is not None and m.kickoff_at <= finished)
        or (m.kickoff_at is None and m.match_date <= finished.date())
    }
    for row in report["fixtures"]:
        if row["match_id"] in late:
            row.update(status="failed", reason="pre_match_window_closed_during_run")
            row.pop("forecast_id", None)
    failed = bool(report["errors"]) or any(r["status"] == "failed" for r in report["fixtures"])
    if failed:
        report["status"] = "failed"
        for row in report["fixtures"]:
            if row["status"] == "forecasted":
                row.update(status="withheld", reason="batch_failed")
                row.pop("forecast_id", None)
        forecasts = []
    report["counts"] = {
        status: sum(r["status"] == status for r in report["fixtures"])
        for status in ("forecasted", "skipped", "failed", "withheld")
    }
    report["completed_at"] = finished.isoformat()
    deadline = (
        min(
            (
                m.kickoff_at or datetime.combine(m.match_date, datetime.min.time(), UTC)
                for m in matches
            ),
            default=None,
        )
        if forecasts
        else None
    )
    try:
        return repo.publish_once(
            "forecast_batch",
            run_id,
            report,
            [("forecast", f["forecast_id"], f, finished) for f in forecasts],
            available_at=finished,
            recorded_at=finished,
            deadline=deadline,
        )
    except PublicationDeadlineError as exc:
        report["status"] = "failed"
        report["errors"].append({"stage": "publication", "reason": str(exc)})
        for row in report["fixtures"]:
            if row["status"] == "forecasted":
                row.update(status="withheld", reason="publication_deadline_closed")
                row.pop("forecast_id", None)
        report["counts"]["withheld"] += report["counts"]["forecasted"]
        report["counts"]["forecasted"] = 0
        finished = max(finished, datetime.now(UTC))
        report["completed_at"] = finished.isoformat()
        return repo.publish_once(
            "forecast_batch",
            run_id,
            report,
            [],
            available_at=finished,
            recorded_at=finished,
        )
