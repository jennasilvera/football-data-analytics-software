from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from football_analytics.data import LeakageRisk, SourceMetadata
from football_analytics.domain import Match, MatchStatus
from football_analytics.features import (
    FeatureMissingReason,
    FeatureStatus,
    FifaRankingFeatureProvider,
    FifaRankingObservation,
    PredictionContext,
    build_feature_vector,
)


def _target() -> Match:
    return Match(
        match_id="target",
        match_date=date(2026, 6, 10),
        kickoff_at=datetime(2026, 6, 10, 20, 0, tzinfo=UTC),
        home_team_id="ARG",
        away_team_id="FRA",
        competition_id="friendly",
        neutral=True,
        status=MatchStatus.SCHEDULED,
    )


def _ranking(
    *,
    team_id: str,
    rank: int,
    points: float | None,
    available_at: datetime | None,
    record_id: str,
    leakage_risk: LeakageRisk = LeakageRisk.SAFE,
) -> FifaRankingObservation:
    return FifaRankingObservation(
        team_id=team_id,
        rank=rank,
        points=points,
        metadata=SourceMetadata(
            source="fifa_ranking_archive",
            source_record_id=record_id,
            available_at=available_at,
            ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
            leakage_risk=leakage_risk,
        ),
    )


def test_fifa_provider_uses_latest_safe_observation_available_at_cutoff() -> None:
    cutoff = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    observations = [
        _ranking(
            team_id="ARG",
            rank=2,
            points=1860.0,
            available_at=cutoff - timedelta(days=30),
            record_id="arg-old",
        ),
        _ranking(
            team_id="ARG",
            rank=1,
            points=1885.0,
            available_at=cutoff - timedelta(days=2),
            record_id="arg-current",
        ),
        _ranking(
            team_id="ARG",
            rank=3,
            points=1800.0,
            available_at=cutoff + timedelta(days=1),
            record_id="arg-future",
        ),
        _ranking(
            team_id="FRA",
            rank=4,
            points=1770.0,
            available_at=cutoff - timedelta(days=2),
            record_id="fra-current",
        ),
    ]
    context = PredictionContext(match=_target(), prediction_time=cutoff)

    vector = build_feature_vector(
        context,
        [FifaRankingFeatureProvider(observations)],
    )
    values = {value.definition.name: value for value in vector.values}

    assert values["fifa.home.rank"].value == pytest.approx(1.0)
    assert values["fifa.away.rank"].value == pytest.approx(4.0)
    assert values["fifa.rank_diff_home_minus_away"].value == pytest.approx(-3.0)
    assert values["fifa.points_diff_home_minus_away"].value == pytest.approx(115.0)
    assert values["fifa.home.rank"].lineage.source_record_ids == ("arg-current",)


def test_fifa_provider_rejects_unknown_or_unsafe_availability() -> None:
    cutoff = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    observations = [
        _ranking(
            team_id="ARG",
            rank=1,
            points=1885.0,
            available_at=None,
            record_id="arg-unknown-availability",
        ),
        _ranking(
            team_id="FRA",
            rank=4,
            points=1770.0,
            available_at=cutoff - timedelta(days=2),
            record_id="fra-review",
            leakage_risk=LeakageRisk.REVIEW,
        ),
    ]
    context = PredictionContext(match=_target(), prediction_time=cutoff)

    vector = build_feature_vector(
        context,
        [FifaRankingFeatureProvider(observations)],
    )
    values = {value.definition.name: value for value in vector.values}

    assert values["fifa.home.rank"].status is FeatureStatus.MISSING
    assert (
        values["fifa.home.rank"].missing_reason
        is FeatureMissingReason.TEMPORAL_INTEGRITY
    )
    assert values["fifa.away.rank"].status is FeatureStatus.MISSING


def test_fifa_provider_can_enforce_staleness_policy() -> None:
    cutoff = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    observations = [
        _ranking(
            team_id="ARG",
            rank=1,
            points=1885.0,
            available_at=cutoff - timedelta(days=60),
            record_id="arg-stale",
        )
    ]
    context = PredictionContext(match=_target(), prediction_time=cutoff)

    vector = build_feature_vector(
        context,
        [FifaRankingFeatureProvider(observations, max_age_days=45)],
    )
    values = {value.definition.name: value for value in vector.values}

    assert values["fifa.home.rank"].status is FeatureStatus.MISSING
    assert values["fifa.home.rank"].missing_reason is FeatureMissingReason.STALE


def test_fifa_points_can_be_missing_while_rank_remains_observed() -> None:
    cutoff = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    observations = [
        _ranking(
            team_id="ARG",
            rank=1,
            points=None,
            available_at=cutoff - timedelta(days=2),
            record_id="arg-rank-only",
        )
    ]
    context = PredictionContext(match=_target(), prediction_time=cutoff)

    vector = build_feature_vector(
        context,
        [FifaRankingFeatureProvider(observations)],
    )
    values = {value.definition.name: value for value in vector.values}

    assert values["fifa.home.rank"].status is FeatureStatus.OBSERVED
    assert values["fifa.home.points"].status is FeatureStatus.MISSING
