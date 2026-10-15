"""Reviewable canonicalization reports for data onboarding."""

from typing import Any

from football_analytics.data.batch import BatchNormalizationReport
from football_analytics.models.postprocessing import content_id


def validation_report(
    batch: BatchNormalizationReport, *, source_id: str, source_sha256: str,
    catalog_id: str, data_kind: str,
) -> dict[str, Any]:
    issues = [
        {"decision": row.decision.value, "source": row.source,
         "source_match_id": row.source_match_id, "reasons": list(row.reasons)}
        for row in (*batch.excluded, *batch.quarantined)
    ]
    body = {
        "schema_version": 1,
        "valid": batch.total_rows > 0 and not issues,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "catalog_id": catalog_id,
        "data_kind": data_kind,
        "matches": len(batch.normalized),
        "counts": {"total": batch.total_rows, "normalized": len(batch.normalized),
                   "excluded": len(batch.excluded), "quarantined": len(batch.quarantined)},
        "issues": issues,
        "batch_errors": ["empty_batch"] if batch.total_rows == 0 else [],
        "normalized_matches": [
            {"source_match_id": row.source_match_id, "match_id": row.match.match_id,
             "home_team_id": row.match.home_team_id, "away_team_id": row.match.away_team_id,
             "competition_id": row.match.competition_id,
             "match_date": row.match.match_date.isoformat()}
            for row in batch.normalized
        ],
        "scope": "input_batch_only; no cross-import or model-readiness certification",
    }
    return {"report_id": content_id("data_validation_", body), **body}
