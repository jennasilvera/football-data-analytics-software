from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from football_analytics.data.adapters import (
    build_legacy_mens_results_observations,
)
from football_analytics.domain import MatchStatus


def test_legacy_results_adapter_preserves_current_schema_semantics() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2022-12-18"],
            "home_team": ["Argentina"],
            "away_team": ["France"],
            "home_score": [3],
            "away_score": [3],
            "tournament": ["FIFA World Cup"],
            "city": ["Lusail"],
            "country": ["Qatar"],
            "neutral": [True],
        }
    )

    observations = build_legacy_mens_results_observations(
        frame,
        ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
        source_version="legacy-baseline",
    )

    observation = observations[0]
    assert observation.match_date.isoformat() == "2022-12-18"
    assert observation.kickoff_at is None
    assert observation.home_score == 3
    assert observation.away_score == 3
    assert observation.status is MatchStatus.COMPLETED
    assert observation.metadata.available_at is None
