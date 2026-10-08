"""Audited confederation slices using membership published by each forecast cutoff."""

from dataclasses import asdict
from datetime import date, datetime

from football_analytics.data.confederations import MembershipHistory
from football_analytics.evaluation.diagnostics import EvaluationDiagnostics
from football_analytics.evaluation.metrics import ScoredPrediction, evaluate_predictions
from football_analytics.models.postprocessing import content_id


def confederation_diagnostics(
    diagnostics: EvaluationDiagnostics, history: MembershipHistory
) -> dict:
    assignments = []
    groups: dict[tuple[str, str], list] = {}
    for row in diagnostics.matches:
        home, home_release = history.resolve(
            row.home_team_id,
            date.fromisoformat(row.match_date),
            datetime.fromisoformat(row.prediction_time_iso),
        )
        away, away_release = history.resolve(
            row.away_team_id,
            date.fromisoformat(row.match_date),
            datetime.fromisoformat(row.prediction_time_iso),
        )
        home_key, away_key = home.value if home else "unknown", away.value if away else "unknown"
        assignments.append(
            {
                "match_id": row.match_id,
                "home": home_key,
                "away": away_key,
                "home_release_id": home_release,
                "away_release_id": away_release,
            }
        )
        for key in (
            ("home_confederation", home_key),
            ("away_confederation", away_key),
            ("confederation_pair", home_key + "/" + away_key),
        ):
            groups.setdefault(key, []).append(row)
    slices = []
    for (dimension, label), matches in sorted(groups.items()):
        slices.append(
            {
                "dimension": dimension,
                "key": label,
                "match_ids": [m.match_id for m in matches],
                "metrics": asdict(
                    evaluate_predictions(
                        [ScoredPrediction(m.actual, m.probabilities) for m in matches]
                    )
                ),
                "below_minimum": len(matches) < diagnostics.min_sample,
            }
        )
    body = {
        "schema_version": 1,
        "diagnostic_id": diagnostics.diagnostic_id,
        "membership_dataset_id": history.dataset_id,
        "membership_releases": [asdict(r) for r in history.releases],
        "min_sample": diagnostics.min_sample,
        "assignments": assignments,
        "slices": slices,
        "coverage": {
            "matches": len(assignments),
            "both_known": sum(
                r["home"] != "unknown" and r["away"] != "unknown" for r in assignments
            ),
        },
        "policy": "complete_timeline_latest_published_at_prediction_v1",
        "note": "Unknown memberships retain their matches; every dimension partitions the sample.",
    }
    from football_analytics.data.serialization import json_safe

    body = json_safe(body)
    return {"report_id": content_id("confederation_diagnostics_", body), **body}
