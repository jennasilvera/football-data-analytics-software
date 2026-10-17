"""Shared single-match serving with snapshot inputs and guarded publication."""

from datetime import UTC, datetime
from typing import Any

from football_analytics.data.contracts import ensure_utc
from football_analytics.data.serialization import match_from_dict
from football_analytics.services.forecasting import forecast_match
from football_analytics.services.governance import require_approved_model_record
from football_analytics.services.lineage import attach_operational_lineage
from football_analytics.services.market import attach_market
from football_analytics.storage.repository import Repository


class ServingConflictError(ValueError):
    """The selected model cannot currently serve an operational forecast."""


def predict_and_publish(
    repo: Repository, *, model_id: str, match_id: str, now: datetime,
    method: str = "ensemble", max_model_age_days: int = 90,
) -> dict[str, Any]:
    now = ensure_utc(now)
    model_record = repo.get("model", model_id, as_of=now)
    model = model_record["payload"]
    try:
        approval = require_approved_model_record(repo, model_id, as_of=now)
        release = approval["payload"]
    except ValueError as exc:
        raise ServingConflictError(str(exc)) from exc
    age = (now - datetime.fromisoformat(model["training_cutoff"])).total_seconds() / 86400
    if type(max_model_age_days) is not int or max_model_age_days <= 0 or not (
        0 <= age <= max_model_age_days
    ):
        raise ServingConflictError("Model history exceeds configured freshness bounds.")
    source = repo.get("match", match_id, as_of=now)
    match = match_from_dict(source["payload"]["match"])
    if match.match_id != match_id:
        raise ValueError("Stored match identity disagrees with its payload.")
    forecast = attach_market(
        repo, forecast_match(model, match, prediction_time=now, method=method), now
    )
    forecast = attach_operational_lineage(
        forecast, model_record_id=model_record["record_id"],
        metrics_record_id=release["metrics_record_id"], approval_record_id=approval["record_id"],
        match_record_id=source["record_id"],
    )
    finished = max(now, datetime.now(UTC))
    deadline = match.kickoff_at or datetime.combine(match.match_date, datetime.min.time(), UTC)
    repo.put_many(
        [("forecast", forecast["forecast_id"], forecast, finished)],
        recorded_at=finished,
        deadline=deadline,
        expected_records=[
            ("model", model_id, model_record["record_id"]),
            ("model_metrics", model_id, release["metrics_record_id"]),
            ("model_release", model_id, approval["record_id"]),
            ("match", match_id, source["record_id"]),
        ],
    )
    return forecast
