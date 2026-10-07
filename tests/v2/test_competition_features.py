from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from football_analytics.data import LeakageRisk, SourceMetadata
from football_analytics.domain import (
    Competition,
    CompetitionKind,
    CompetitionStage,
    Confederation,
    Match,
)
from football_analytics.features import (
    CompetitionContextFeatureProvider,
    FeatureMissingReason,
    FeatureStatus,
    MatchCompetitionContextObservation,
    PredictionContext,
    build_feature_vector,
)

CUTOFF = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _metadata(
    record_id: str,
    *,
    available_at: datetime,
    leakage_risk: LeakageRisk = LeakageRisk.SAFE,
) -> SourceMetadata:
    return SourceMetadata(
        source="competition_feed",
        source_record_id=record_id,
        available_at=available_at,
        ingested_at=max(available_at, CUTOFF - timedelta(days=30)),
        leakage_risk=leakage_risk,
    )


def _match(
    *,
    competition_id: str = "uefa-nations-league",
) -> Match:
    return Match(
        match_id="fra-esp",
        match_date=date(2026, 6, 3),
        kickoff_at=datetime(2026, 6, 3, 20, 0, tzinfo=UTC),
        home_team_id="FRA",
        away_team_id="ESP",
        competition_id=competition_id,
        neutral=True,
    )


def _competition() -> Competition:
    return Competition(
        competition_id="uefa-nations-league",
        name="UEFA Nations League",
        kind=CompetitionKind.NATIONS_LEAGUE,
        confederation=Confederation.UEFA,
    )


def test_competition_kind_uses_canonical_enum_not_name_heuristics() -> None:
    provider = CompetitionContextFeatureProvider(
        [
            Competition(
                competition_id="odd-name",
                name="Contains World Cup Friendly Words",
                kind=CompetitionKind.NATIONS_LEAGUE,
                confederation=Confederation.UEFA,
            )
        ]
    )
    context = PredictionContext(
        match=_match(competition_id="odd-name"),
        prediction_time=CUTOFF,
    )

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["competition.kind.nations_league"].value == 1.0
    assert values["competition.kind.world_cup"].value == 0.0
    assert values["competition.kind.friendly"].value == 0.0


def test_competition_stage_and_leg_use_latest_cutoff_safe_observation() -> None:
    provider = CompetitionContextFeatureProvider(
        [_competition()],
        match_context=[
            MatchCompetitionContextObservation(
                match_id="fra-esp",
                stage=CompetitionStage.PLAYOFF,
                leg_number=1,
                metadata=_metadata(
                    "context-before",
                    available_at=CUTOFF - timedelta(days=1),
                ),
            ),
            MatchCompetitionContextObservation(
                match_id="fra-esp",
                stage=CompetitionStage.FINAL,
                leg_number=None,
                metadata=_metadata(
                    "context-future",
                    available_at=CUTOFF + timedelta(hours=1),
                ),
            ),
        ],
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["competition.stage.playoff"].value == 1.0
    assert values["competition.stage.final"].value == 0.0
    assert values["competition.leg_number"].value == 1.0
    assert values["competition.leg_number"].lineage.source_record_ids == (
        "context-before",
    )


def test_future_only_stage_context_is_missing_not_leaked() -> None:
    provider = CompetitionContextFeatureProvider(
        [_competition()],
        match_context=[
            MatchCompetitionContextObservation(
                match_id="fra-esp",
                stage=CompetitionStage.FINAL,
                metadata=_metadata(
                    "future",
                    available_at=CUTOFF + timedelta(hours=1),
                ),
            )
        ],
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["competition.stage.final"].status is FeatureStatus.MISSING
    assert (
        values["competition.stage.final"].missing_reason
        is FeatureMissingReason.TEMPORAL_INTEGRITY
    )


def test_unknown_competition_produces_missing_kind_features() -> None:
    provider = CompetitionContextFeatureProvider([_competition()])
    context = PredictionContext(
        match=_match(competition_id="unregistered"),
        prediction_time=CUTOFF,
    )

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["competition.kind.world_cup"].status is FeatureStatus.MISSING
    assert (
        values["competition.kind.world_cup"].missing_reason
        is FeatureMissingReason.UPSTREAM_UNAVAILABLE
    )


def test_missing_leg_number_is_explicitly_not_applicable() -> None:
    provider = CompetitionContextFeatureProvider(
        [_competition()],
        match_context=[
            MatchCompetitionContextObservation(
                match_id="fra-esp",
                stage=CompetitionStage.GROUP_OR_LEAGUE,
                metadata=_metadata(
                    "group-stage",
                    available_at=CUTOFF - timedelta(days=1),
                ),
            )
        ],
    )
    context = PredictionContext(match=_match(), prediction_time=CUTOFF)

    vector = build_feature_vector(context, [provider])
    values = {value.definition.name: value for value in vector.values}

    assert values["competition.leg_number"].status is FeatureStatus.MISSING
    assert (
        values["competition.leg_number"].missing_reason
        is FeatureMissingReason.NOT_APPLICABLE
    )
