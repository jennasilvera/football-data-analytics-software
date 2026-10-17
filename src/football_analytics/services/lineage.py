"""Exact operational forecast inputs, preserved across later repository revisions."""

from typing import Any

from football_analytics.models.postprocessing import content_id
from football_analytics.storage.repository import Repository, RepositoryIntegrityError


def attach_operational_lineage(
    forecast: dict[str, Any], *, model_record_id: str, metrics_record_id: str,
    approval_record_id: str, match_record_id: str,
) -> dict[str, Any]:
    body = {key: value for key, value in forecast.items() if key != "forecast_id"}
    body["operational_lineage"] = {
        "schema_version": 1, "model_record_id": model_record_id,
        "metrics_record_id": metrics_record_id, "approval_record_id": approval_record_id,
        "match_record_id": match_record_id,
    }
    return {"forecast_id": content_id("forecast_", body), **body}


def forecast_lineage(repo: Repository, forecast_id: str) -> dict[str, Any]:
    forecast = repo.get("forecast", forecast_id)["payload"]
    lineage = forecast.get("operational_lineage")
    report: dict[str, Any] = {"forecast_id": forecast_id, "records": []}
    if lineage is None:
        return {**report, "status": "unavailable", "reason": "legacy_or_research_forecast"}
    if lineage.get("schema_version") != 1:
        raise RepositoryIntegrityError("Unsupported forecast lineage schema.")
    expected = {
        "model_record_id": ("model", forecast["bundle_id"]),
        "metrics_record_id": ("model_metrics", forecast["bundle_id"]),
        "approval_record_id": ("model_release", forecast["bundle_id"]),
        "match_record_id": ("match", forecast["match"]["match_id"]),
    }
    refs = [(role, lineage.get(role), kind, entity) for role, (kind, entity) in expected.items()]
    market = forecast.get("market_disagreement")
    if market:
        refs.extend(
            ("market_record_id", identity, "odds", None) for identity in market["record_ids"]
        )
    for role, identity, kind, entity in refs:
        try:
            row = repo.get_record(identity)
        except KeyError as exc:
            raise RepositoryIntegrityError("Forecast lineage revision is unavailable.") from exc
        if row["kind"] != kind or (entity is not None and row["entity_id"] != entity):
            raise RepositoryIntegrityError("Forecast lineage revision identity mismatch.")
        report["records"].append({
            "role": role,
            **{key: row[key] for key in (
                "record_id", "kind", "entity_id", "available_at", "recorded_at"
            )},
        })
    return {**report, "status": "resolved",
            "note": "Historical inputs; not current serving approval"}
