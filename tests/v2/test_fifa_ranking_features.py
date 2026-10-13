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


def test_ranking_trends_adjacent_publications_and_missing_points():
    from football_analytics.features.fifa_ranking import FifaRankingTrendProvider

    cutoff = datetime(2026, 6, 1, tzinfo=UTC)
    rows = [
        _ranking(team_id="ARG", rank=30, points=1300, available_at=cutoff-timedelta(days=60),
                 record_id="older"),
        _ranking(team_id="ARG", rank=26, points=None, available_at=cutoff-timedelta(days=30),
                 record_id="previous"),
        _ranking(team_id="ARG", rank=25, points=1400, available_at=cutoff,
                 record_id="latest"),
        _ranking(team_id="ARG", rank=1, points=2000, available_at=cutoff+timedelta(days=1),
                 record_id="future"),
    ]
    context = PredictionContext(_target(), cutoff)
    provider = FifaRankingTrendProvider(rows)
    values = {v.definition.name: v for v in provider.compute(context)}
    assert values["fifa_trend.home.rank_bucket"].value == 2
    assert values["fifa_trend.home.rank_improvement"].value == 1
    assert values["fifa_trend.home.publication_gap_days"].value == 30
    assert values["fifa_trend.home.points_change"].value is None
    assert values["fifa_trend.home.points_change"].lineage.source_record_ids == (
        "latest", "previous"
    )
    assert values["fifa_trend.away.rank_bucket"].value is None
    assert all(v.value is None for v in FifaRankingTrendProvider(
        rows[:-2], max_age_days=10
    ).compute(context))
    # Adding this group never changes the established ranking-only schema.
    assert len(FifaRankingFeatureProvider(rows).definitions()) == 6


@pytest.mark.parametrize("rank,bucket", [(1, 1), (10, 1), (11, 2), (25, 2), (26, 3),
                                          (50, 3), (51, 4), (100, 4), (101, 5)])
def test_ranking_bucket_boundaries(rank, bucket):
    from football_analytics.features.fifa_ranking import FifaRankingTrendProvider

    cutoff = datetime(2026, 6, 1, tzinfo=UTC)
    row = _ranking(team_id="ARG", rank=rank, points=1400, available_at=cutoff, record_id="r")
    values = FifaRankingTrendProvider([row]).compute(PredictionContext(_target(), cutoff))
    assert values[0].value == bucket
    assert values[1].missing_reason is FeatureMissingReason.NO_HISTORY


@pytest.mark.parametrize("rank,points", [(1.5, 10), (True, 10), (1, float("nan")),
                                         (1, float("inf"))])
def test_ranking_observation_rejects_invalid_numeric_inputs(rank, points):
    with pytest.raises(ValueError):
        _ranking(team_id="ARG", rank=rank, points=points, available_at=None, record_id="r")


def test_ranking_trends_do_not_mix_provider_methodologies():
    from dataclasses import replace

    from football_analytics.features.fifa_ranking import FifaRankingTrendProvider

    cutoff = datetime(2026, 6, 1, tzinfo=UTC)
    old = _ranking(team_id="ARG", rank=20, points=1400,
                   available_at=cutoff-timedelta(days=30), record_id="old")
    new = _ranking(team_id="ARG", rank=10, points=1600, available_at=cutoff, record_id="new")
    new = replace(new, metadata=replace(new.metadata, source="new-methodology"))
    values = FifaRankingTrendProvider([old, new]).compute(PredictionContext(_target(), cutoff))
    assert values[0].value == 1
    assert values[1].value is None
    assert values[2].value is None


def test_ranking_points_momentum_uses_adjacent_safe_records():
    from football_analytics.features.fifa_ranking import FifaRankingTrendProvider

    cutoff = datetime(2026, 6, 1, tzinfo=UTC)
    rows = [
        _ranking(team_id="ARG", rank=20, points=1400,
                 available_at=cutoff-timedelta(days=30), record_id="old"),
        _ranking(team_id="ARG", rank=10, points=1450, available_at=cutoff, record_id="new"),
        _ranking(team_id="ARG", rank=1, points=2000,
                 available_at=cutoff-timedelta(days=1), record_id="unsafe",
                 leakage_risk=LeakageRisk.POST_MATCH_ONLY),
    ]
    values = FifaRankingTrendProvider(rows).compute(PredictionContext(_target(), cutoff))
    assert values[1].value == 10
    assert values[2].value == 50
    assert values[2].lineage.source_record_ids == ("new", "old")
