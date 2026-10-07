from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pandas as pd

from football_analytics.data.adapters.tabular import (
    TabularMatchColumns,
    TabularSourcePolicy,
    build_match_observations,
)
from football_analytics.data.contracts import LeakageRisk
from football_analytics.data.observations import MatchObservation
from football_analytics.data.scope import GenderCategory, TeamLevel
from football_analytics.domain import MatchStatus
from football_analytics.domain.scores import ScoreBasis

LEGACY_RESULTS_COLUMNS = TabularMatchColumns(
    match_date="date",
    home_team="home_team",
    away_team="away_team",
    competition="tournament",
    neutral="neutral",
    home_score="home_score",
    away_score="away_score",
)


def build_legacy_mens_results_observations(
    frame: pd.DataFrame,
    *,
    ingested_at: datetime,
    source_id: str = "legacy_mens_international_results",
    source_version: str | None = None,
    legal_use_notes: str | None = None,
    score_basis: ScoreBasis = ScoreBasis.UNKNOWN,
) -> list[MatchObservation]:
    """Adapt the current legacy historical-results schema into V2 observations."""

    policy = TabularSourcePolicy(
        source_id=source_id,
        score_basis=score_basis,
        gender=GenderCategory.MEN,
        team_level=TeamLevel.SENIOR_A,
        official=True,
        default_status=MatchStatus.COMPLETED,
        leakage_risk=LeakageRisk.POST_MATCH_ONLY,
        source_version=source_version,
        legal_use_notes=legal_use_notes,
    )

    return build_match_observations(
        frame,
        columns=replace(
            LEGACY_RESULTS_COLUMNS,
            score_basis="score_basis" if "score_basis" in frame.columns else None,
        ),
        policy=policy,
        ingested_at=ingested_at,
    )
