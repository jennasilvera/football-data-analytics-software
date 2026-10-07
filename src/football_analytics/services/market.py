"""Operational market enrichment selects only snapshots known at forecast time."""

from datetime import datetime

from football_analytics.data.contracts import LeakageRisk
from football_analytics.data.serialization import metadata_from_dict
from football_analytics.features.market import MarketSnapshotObservation, devig_decimal_odds
from football_analytics.models.postprocessing import content_id
from football_analytics.storage.repository import Repository


def attach_market(repo: Repository, forecast: dict, as_of: datetime) -> dict:
    snapshots: dict[str, tuple[datetime, MarketSnapshotObservation, str]] = {}
    for record in repo.scan("odds", as_of=as_of):
        row = dict(record["payload"])
        if row.pop("score_basis", "unknown") != "regulation_time":
            continue
        row["metadata"] = metadata_from_dict(row["metadata"])
        snapshot = MarketSnapshotObservation(**row)
        if snapshot.match_id != forecast["match"]["match_id"]:
            continue
        if snapshot.metadata.leakage_risk is not LeakageRisk.SAFE:
            continue
        available = snapshot.metadata.available_at
        if available is None or available > as_of:
            continue
        previous = snapshots.get(snapshot.market_source_id)
        if previous is None or previous[0] < available:
            snapshots[snapshot.market_source_id] = (available, snapshot, record["record_id"])
    if not snapshots:
        return forecast
    values = [devig_decimal_odds(item[1]).probabilities.as_tuple() for item in snapshots.values()]
    keys = ("home_win", "draw", "away_win")
    fair = {key: sum(v[i] for v in values) / len(values) for i, key in enumerate(keys)}
    body = {k: v for k, v in forecast.items() if k != "forecast_id"}
    body["market_disagreement"] = {
        "market_probabilities": fair,
        "model_minus_market": {k: forecast["probabilities"][k] - fair[k] for k in keys},
        "bookmakers": sorted(snapshots),
        "record_ids": sorted(item[2] for item in snapshots.values()),
        "note": "Proportional margin removal; equal bookmaker weights; no staking recommendation",
    }
    return {"forecast_id": content_id("forecast_", body), **body}
