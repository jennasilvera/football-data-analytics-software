from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from football_analytics.data import CanonicalMatchRecord, LeakageRisk, SourceMetadata
from football_analytics.domain import Match, MatchStatus
from football_analytics.domain.scores import ScoreBasis
from football_analytics.features import (
    FeatureStatus,
    LegacyEloFeatureProvider,
    PredictionContext,
    build_feature_vector,
)
from football_analytics.ratings import LegacyEloRatingEngine, replay_completed_matches


def _completed_record(
    *,
    match_id: str,
    match_date: date,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
) -> CanonicalMatchRecord:
    return CanonicalMatchRecord(
        score_basis=ScoreBasis.REGULATION_TIME,
        match=Match(
            match_id=match_id,
            match_date=match_date,
            home_team_id=home_team,
            away_team_id=away_team,
            competition_id="friendly",
            neutral=True,
            status=MatchStatus.COMPLETED,
        ),
        metadata=SourceMetadata(
            source="legacy",
            ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
            leakage_risk=LeakageRisk.POST_MATCH_ONLY,
        ),
        source_match_id=f"legacy-{match_id}",
        source_home_team_name=home_team,
        source_away_team_name=away_team,
        source_competition_name="International Friendly",
        home_score=home_score,
        away_score=away_score,
    )


def _target_match() -> Match:
    return Match(
        match_id="arg-fra-target",
        match_date=date(2026, 6, 3),
        kickoff_at=datetime(2026, 6, 3, 20, 0, tzinfo=UTC),
        home_team_id="ARG",
        away_team_id="FRA",
        competition_id="friendly",
        neutral=True,
        status=MatchStatus.SCHEDULED,
    )


def _snapshots():
    replay = replay_completed_matches(
        [
            _completed_record(
                match_id="arg-bra-prior",
                match_date=date(2026, 5, 30),
                home_team="ARG",
                away_team="BRA",
                home_score=2,
                away_score=0,
            )
        ],
        engine=LegacyEloRatingEngine(),
        competition_names={"friendly": "International Friendly"},
    )
    return replay.snapshots


def test_rating_provider_uses_only_snapshots_before_prediction_cutoff() -> None:
    context = PredictionContext(
        match=_target_match(),
        prediction_time=datetime(2026, 5, 29, 12, 0, tzinfo=UTC),
    )

    vector = build_feature_vector(
        context,
        [LegacyEloFeatureProvider(_snapshots())],
    )
    values = {value.definition.name: value for value in vector.values}

    assert values["elo.home_rating"].value == pytest.approx(1500.0)
    assert values["elo.home_rating"].status is FeatureStatus.IMPUTED
    assert values["elo.away_rating"].status is FeatureStatus.IMPUTED
    assert values["elo.rating_diff_home_minus_away"].status is FeatureStatus.IMPUTED


def test_rating_provider_uses_prior_snapshot_after_it_is_temporally_eligible() -> None:
    context = PredictionContext(
        match=_target_match(),
        prediction_time=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )

    vector = build_feature_vector(
        context,
        [LegacyEloFeatureProvider(_snapshots())],
    )
    values = {value.definition.name: value for value in vector.values}

    assert values["elo.home_rating"].value > 1500.0
    assert values["elo.home_rating"].status is FeatureStatus.OBSERVED
    assert values["elo.home_rating"].lineage.artifact_ids == ("arg-bra-prior",)

    assert values["elo.away_rating"].value == pytest.approx(1500.0)
    assert values["elo.away_rating"].status is FeatureStatus.IMPUTED

    assert values["elo.rating_diff_home_minus_away"].status is FeatureStatus.IMPUTED
    assert values["elo.expected_home_score"].status is FeatureStatus.IMPUTED


def test_rating_default_is_explicit_imputation_not_observed_data() -> None:
    context = PredictionContext(
        match=_target_match(),
        prediction_time=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )

    vector = build_feature_vector(
        context,
        [LegacyEloFeatureProvider([])],
    )

    assert all(value.status is FeatureStatus.IMPUTED for value in vector.values)
    assert all(value.value is not None for value in vector.values)
    assert all(
        value.imputation_method is not None
        for value in vector.values
    )


def test_feature_set_id_is_deterministic() -> None:
    context = PredictionContext(
        match=_target_match(),
        prediction_time=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )
    provider = LegacyEloFeatureProvider(_snapshots())

    first = build_feature_vector(context, [provider])
    second = build_feature_vector(context, [provider])

    assert first.feature_set_id == second.feature_set_id
    assert all(
        value.definition.version == "legacy_elo_v1"
        for value in first.values
    )
