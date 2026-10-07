"""Explicit JSON codecs; no dynamic imports or executable deserialization."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from typing import Any

from football_analytics.data.contracts import LeakageRisk, SourceMetadata
from football_analytics.data.normalization import CanonicalMatchRecord
from football_analytics.domain import Match, MatchStatus
from football_analytics.domain.scores import ScoreBasis


def json_safe(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            default=lambda x: x.isoformat() if isinstance(x, (datetime, date)) else asdict(x),
            allow_nan=False,
        )
    )


def match_from_dict(payload: dict[str, Any]) -> Match:
    p = dict(payload)
    p["match_date"] = date.fromisoformat(p["match_date"])
    p["kickoff_at"] = datetime.fromisoformat(p["kickoff_at"]) if p.get("kickoff_at") else None
    p["status"] = MatchStatus(p.get("status", "scheduled"))
    return Match(**p)


def record_from_dict(payload: dict[str, Any]) -> CanonicalMatchRecord:
    p = dict(payload)
    metadata = dict(p.pop("metadata"))
    for key in ("available_at", "ingested_at", "event_time"):
        metadata[key] = datetime.fromisoformat(metadata[key]) if metadata.get(key) else None
    metadata["leakage_risk"] = LeakageRisk(metadata["leakage_risk"])
    p["score_basis"] = ScoreBasis(p["score_basis"])
    return CanonicalMatchRecord(
        **{**p, "match": match_from_dict(p["match"]), "metadata": SourceMetadata(**metadata)}
    )


def metadata_from_dict(payload: dict[str, Any]) -> SourceMetadata:
    metadata = dict(payload)
    for key in ("available_at", "ingested_at", "event_time"):
        metadata[key] = datetime.fromisoformat(metadata[key]) if metadata.get(key) else None
    metadata["leakage_risk"] = LeakageRisk(metadata["leakage_risk"])
    return SourceMetadata(**metadata)


def catalogs_from_dict(payload: dict[str, Any]):
    from football_analytics.data.catalogs import CanonicalCatalogs
    from football_analytics.domain import Competition, CompetitionKind, Confederation, Team

    teams = tuple(
        Team(
            **{
                **r,
                "confederation": Confederation(r["confederation"])
                if r.get("confederation")
                else None,
            }
        )
        for r in payload["teams"]
    )
    competitions = tuple(
        Competition(
            **{
                **r,
                "kind": CompetitionKind(r["kind"]),
                "confederation": Confederation(r["confederation"])
                if r.get("confederation")
                else None,
            }
        )
        for r in payload["competitions"]
    )
    return CanonicalCatalogs(
        teams, payload["team_aliases"], competitions, payload["competition_aliases"]
    )
