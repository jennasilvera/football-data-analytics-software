from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_analytics.data.batch import BatchNormalizationReport
from football_analytics.data.normalization import CanonicalMatchRecord


NORMALIZED_COLUMNS = [
    "match_id",
    "match_date",
    "kickoff_at",
    "time_precision",
    "home_team_id",
    "away_team_id",
    "competition_id",
    "neutral",
    "status",
    "home_score",
    "away_score",
    "source",
    "source_match_id",
    "available_at",
    "ingested_at",
    "leakage_risk",
    "home_resolution_method",
    "away_resolution_method",
    "competition_resolution_method",
]

DISPOSITION_COLUMNS = [
    "source",
    "source_match_id",
    "decision",
    "reasons",
]


def normalization_frames(
    report: BatchNormalizationReport,
) -> dict[str, pd.DataFrame]:
    """Convert a batch report into stable tabular artifacts."""

    normalized = pd.DataFrame(
        [_normalized_row(record) for record in report.normalized],
        columns=NORMALIZED_COLUMNS,
    )
    excluded = pd.DataFrame(
        [_disposition_row(result) for result in report.excluded],
        columns=DISPOSITION_COLUMNS,
    )
    quarantined = pd.DataFrame(
        [_disposition_row(result) for result in report.quarantined],
        columns=DISPOSITION_COLUMNS,
    )

    return {
        "normalized": normalized,
        "excluded": excluded,
        "quarantined": quarantined,
    }


def save_normalization_artifacts(
    report: BatchNormalizationReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write normalized, excluded, and quarantine CSV artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    frames = normalization_frames(report)
    paths = {
        "normalized": destination / "normalized_matches.csv",
        "excluded": destination / "excluded_matches.csv",
        "quarantined": destination / "quarantined_matches.csv",
    }

    for name, path in paths.items():
        frames[name].to_csv(path, index=False)

    return paths


def _normalized_row(record: CanonicalMatchRecord) -> dict[str, object]:
    match = record.match
    metadata = record.metadata

    return {
        "match_id": match.match_id,
        "match_date": match.match_date.isoformat(),
        "kickoff_at": match.kickoff_at.isoformat()
        if match.kickoff_at is not None
        else None,
        "time_precision": match.time_precision.value,
        "home_team_id": match.home_team_id,
        "away_team_id": match.away_team_id,
        "competition_id": match.competition_id,
        "neutral": match.neutral,
        "status": match.status.value,
        "home_score": record.home_score,
        "away_score": record.away_score,
        "source": metadata.source,
        "source_match_id": record.source_match_id,
        "available_at": metadata.available_at.isoformat()
        if metadata.available_at is not None
        else None,
        "ingested_at": metadata.ingested_at.isoformat(),
        "leakage_risk": metadata.leakage_risk.value,
        "home_resolution_method": record.home_resolution_method,
        "away_resolution_method": record.away_resolution_method,
        "competition_resolution_method": record.competition_resolution_method,
    }


def _disposition_row(result: object) -> dict[str, object]:
    return {
        "source": result.source,
        "source_match_id": result.source_match_id,
        "decision": result.decision.value,
        "reasons": "|".join(result.reasons),
    }
