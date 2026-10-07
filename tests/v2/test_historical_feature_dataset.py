from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from football_analytics.data import CanonicalMatchRecord, LeakageRisk, SourceMetadata
from football_analytics.domain import Match, MatchOutcome, MatchStatus
from football_analytics.features import (
    FeatureDefinition,
    FeatureLineage,
    FeatureStatus,
    FeatureValue,
    HistoricalFeatureLeakageError,
    PredictionCutoffPolicy,
    ResultEligibilityBasis,
    RollingFormFeatureProvider,
    build_historical_feature_dataset,
)


def _record(
    *,
    match_id: str,
    match_date: date,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    kickoff_at: datetime | None = None,
) -> CanonicalMatchRecord:
    return CanonicalMatchRecord(
        match=Match(
            match_id=match_id,
            match_date=match_date,
            kickoff_at=kickoff_at,
            home_team_id=home,
            away_team_id=away,
            competition_id="friendly",
            neutral=True,
            status=MatchStatus.COMPLETED,
        ),
        metadata=SourceMetadata(
            source="history",
            ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
            leakage_risk=LeakageRisk.POST_MATCH_ONLY,
        ),
        source_match_id=f"source-{match_id}",
        source_home_team_name=home,
        source_away_team_name=away,
        source_competition_name="International Friendly",
        home_score=home_score,
        away_score=away_score,
    )


def test_date_only_dataset_uses_prior_day_cutoff_and_excludes_target_result() -> None:
    records = [
        _record(
            match_id="arg-bra",
            match_date=date(2026, 1, 1),
            home="ARG",
            away="BRA",
            home_score=2,
            away_score=0,
        ),
        _record(
            match_id="arg-fra",
            match_date=date(2026, 1, 10),
            home="ARG",
            away="FRA",
            home_score=1,
            away_score=1,
        ),
    ]

    dataset = build_historical_feature_dataset(
        records,
        providers=[RollingFormFeatureProvider(records)],
    )

    first, second = dataset.examples
    assert first.target is MatchOutcome.HOME_WIN
    assert second.target is MatchOutcome.DRAW
    assert dataset.result_eligibility_policy_id == "completed_result_eligibility_v1"
    assert first.target_available_at == datetime(2026, 1, 2, tzinfo=UTC)
    assert (
        first.target_availability_basis
        is ResultEligibilityBasis.CONSERVATIVE_NEXT_UTC_DAY
    )
    assert second.prediction_time == datetime(
        2026,
        1,
        9,
        23,
        59,
        59,
        999999,
        tzinfo=UTC,
    )

    second_values = {
        value.definition.name: value
        for value in second.vector.values
    }
    form_value = second_values["form.home.points_per_match_5"]
    assert form_value.value == pytest.approx(3.0)
    assert form_value.status is FeatureStatus.OBSERVED
    assert form_value.lineage.source_record_ids == ("source-arg-bra",)
    assert "source-arg-fra" not in form_value.lineage.source_record_ids


def test_exact_kickoff_policy_supports_explicit_forecast_horizon() -> None:
    kickoff = datetime(2026, 6, 10, 20, 0, tzinfo=UTC)
    record = _record(
        match_id="arg-fra",
        match_date=date(2026, 6, 10),
        kickoff_at=kickoff,
        home="ARG",
        away="FRA",
        home_score=2,
        away_score=1,
    )
    policy = PredictionCutoffPolicy(
        policy_id="twenty_four_hours_before_v1",
        exact_kickoff_lead=timedelta(hours=24),
    )

    dataset = build_historical_feature_dataset(
        [record],
        providers=[RollingFormFeatureProvider([record])],
        cutoff_policy=policy,
    )

    assert dataset.cutoff_policy_id == "twenty_four_hours_before_v1"
    assert dataset.examples[0].prediction_time == kickoff - timedelta(hours=24)


class _FutureLineageProvider:
    provider_id = "future_lineage_test"
    _definition = FeatureDefinition(
        name="test.future_lineage",
        version="v1",
        group="test",
        description="Feature intentionally carrying future lineage.",
    )

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return (self._definition,)

    def compute(self, context):
        return (
            FeatureValue(
                definition=self._definition,
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=1.0,
                lineage=FeatureLineage(
                    source_record_ids=("source-future-match",),
                ),
            ),
        )


def test_historical_dataset_rejects_future_match_lineage() -> None:
    records = [
        _record(
            match_id="target-match",
            match_date=date(2026, 1, 1),
            home="ARG",
            away="BRA",
            home_score=1,
            away_score=0,
        ),
        _record(
            match_id="future-match",
            match_date=date(2026, 1, 2),
            home="FRA",
            away="GER",
            home_score=1,
            away_score=1,
        ),
    ]

    with pytest.raises(HistoricalFeatureLeakageError, match="not eligible before the cutoff"):
        build_historical_feature_dataset(
            records,
            providers=[_FutureLineageProvider()],
        )


class _TargetArtifactProvider:
    provider_id = "target_artifact_test"
    _definition = FeatureDefinition(
        name="test.target_artifact",
        version="v1",
        group="test",
        description="Feature intentionally carrying target artifact lineage.",
    )

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return (self._definition,)

    def compute(self, context):
        return (
            FeatureValue(
                definition=self._definition,
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=1.0,
                lineage=FeatureLineage(
                    artifact_ids=(context.match.match_id,),
                ),
            ),
        )


def test_historical_dataset_rejects_target_match_artifact_lineage() -> None:
    record = _record(
        match_id="target-match",
        match_date=date(2026, 1, 1),
        home="ARG",
        away="BRA",
        home_score=1,
        away_score=0,
    )

    with pytest.raises(HistoricalFeatureLeakageError, match="not eligible before the cutoff"):
        build_historical_feature_dataset(
            [record],
            providers=[_TargetArtifactProvider()],
        )


def test_historical_dataset_requires_completed_scored_matches() -> None:
    with pytest.raises(ValueError, match="No completed scored matches"):
        build_historical_feature_dataset(
            [],
            providers=[],
        )
