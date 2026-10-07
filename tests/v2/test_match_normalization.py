from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from football_analytics.data import (
    CompetitionEntityResolver,
    GenderCategory,
    LeakageRisk,
    MatchObservation,
    NormalizationDecision,
    SourceMetadata,
    TeamEntityResolver,
    TeamLevel,
    normalize_match_observation,
)
from football_analytics.domain import (
    Competition,
    CompetitionKind,
    MatchStatus,
    MatchTimePrecision,
    Team,
)


MATCH_DATE = date(2026, 11, 14)
KICKOFF = datetime(2026, 11, 14, 20, 0, tzinfo=UTC)


def _metadata() -> SourceMetadata:
    return SourceMetadata(
        source="fixture_feed",
        source_record_id="source-row-42",
        source_version="2026-11-14T17:00:00Z",
        event_time=KICKOFF,
        available_at=KICKOFF - timedelta(hours=3),
        ingested_at=KICKOFF - timedelta(hours=2, minutes=55),
        leakage_risk=LeakageRisk.SAFE,
    )


def _team_resolver() -> TeamEntityResolver:
    return TeamEntityResolver(
        [
            Team(team_id="ARG", name="Argentina", fifa_code="ARG"),
            Team(team_id="BRA", name="Brazil", fifa_code="BRA"),
        ],
        aliases={"Brasil": "BRA"},
    )


def _competition_resolver() -> CompetitionEntityResolver:
    return CompetitionEntityResolver(
        [
            Competition(
                competition_id="friendly",
                name="International Friendly",
                kind=CompetitionKind.FRIENDLY,
            )
        ],
        aliases={"Friendly": "friendly"},
    )


def _observation(
    *,
    away_team_name: str = "Brasil",
    competition_name: str = "Friendly",
    gender: GenderCategory = GenderCategory.MEN,
    team_level: TeamLevel = TeamLevel.SENIOR_A,
    official: bool | None = True,
) -> MatchObservation:
    return MatchObservation(
        source_match_id="feed-123",
        match_date=MATCH_DATE,
        kickoff_at=KICKOFF,
        home_team_name="Argentina",
        away_team_name=away_team_name,
        competition_name=competition_name,
        neutral=True,
        status=MatchStatus.SCHEDULED,
        gender=gender,
        team_level=team_level,
        official=official,
        metadata=_metadata(),
    )


def test_match_normalization_resolves_canonical_entities_and_preserves_lineage() -> None:
    result = normalize_match_observation(
        _observation(),
        team_resolver=_team_resolver(),
        competition_resolver=_competition_resolver(),
    )

    assert result.decision is NormalizationDecision.NORMALIZED
    assert result.record is not None
    assert result.record.match.home_team_id == "ARG"
    assert result.record.match.away_team_id == "BRA"
    assert result.record.match.competition_id == "friendly"
    assert result.record.match.time_precision is MatchTimePrecision.EXACT_KICKOFF
    assert result.record.away_resolution_method == "alias"
    assert result.record.source_match_id == "feed-123"
    assert result.record.metadata.source == "fixture_feed"


def test_match_id_is_independent_of_source_record_id_and_kickoff_precision() -> None:
    first = _observation()
    second = MatchObservation(
        source_match_id="different-provider-id",
        match_date=first.match_date,
        kickoff_at=None,
        home_team_name=first.home_team_name,
        away_team_name=first.away_team_name,
        competition_name=first.competition_name,
        neutral=first.neutral,
        status=first.status,
        gender=first.gender,
        team_level=first.team_level,
        official=first.official,
        metadata=SourceMetadata(
            source="other_feed",
            available_at=first.metadata.available_at,
            ingested_at=first.metadata.ingested_at,
        ),
    )

    first_result = normalize_match_observation(
        first,
        team_resolver=_team_resolver(),
        competition_resolver=_competition_resolver(),
    )
    second_result = normalize_match_observation(
        second,
        team_resolver=_team_resolver(),
        competition_resolver=_competition_resolver(),
    )

    assert first_result.record is not None
    assert second_result.record is not None
    assert first_result.record.match.match_id == second_result.record.match.match_id
    assert second_result.record.match.time_precision is MatchTimePrecision.DATE_ONLY


def test_unresolved_team_is_quarantined_instead_of_receiving_fallback_identity() -> None:
    result = normalize_match_observation(
        _observation(away_team_name="Atlantis"),
        team_resolver=_team_resolver(),
        competition_resolver=_competition_resolver(),
    )

    assert result.decision is NormalizationDecision.QUARANTINED
    assert result.record is None
    assert result.reasons == ("unresolved_away_team:Atlantis",)


def test_unknown_competition_is_quarantined() -> None:
    result = normalize_match_observation(
        _observation(competition_name="Mystery Cup"),
        team_resolver=_team_resolver(),
        competition_resolver=_competition_resolver(),
    )

    assert result.decision is NormalizationDecision.QUARANTINED
    assert result.reasons == ("unresolved_competition:Mystery Cup",)


def test_out_of_scope_match_is_excluded_before_entity_resolution() -> None:
    result = normalize_match_observation(
        _observation(
            away_team_name="Unknown U23 Team",
            team_level=TeamLevel.OLYMPIC_U23,
        ),
        team_resolver=_team_resolver(),
        competition_resolver=_competition_resolver(),
    )

    assert result.decision is NormalizationDecision.EXCLUDED
    assert result.reasons == ("team_level_out_of_scope:olympic_u23",)


def test_unknown_scope_metadata_is_quarantined_for_review() -> None:
    result = normalize_match_observation(
        _observation(gender=GenderCategory.UNKNOWN),
        team_resolver=_team_resolver(),
        competition_resolver=_competition_resolver(),
    )

    assert result.decision is NormalizationDecision.QUARANTINED
    assert result.reasons == ("gender_unknown",)


def test_completed_observation_requires_scores() -> None:
    with pytest.raises(ValueError, match="require final scores"):
        MatchObservation(
            source_match_id="feed-123",
            match_date=MATCH_DATE,
            kickoff_at=KICKOFF,
            home_team_name="Argentina",
            away_team_name="Brazil",
            competition_name="Friendly",
            neutral=True,
            status=MatchStatus.COMPLETED,
            gender=GenderCategory.MEN,
            team_level=TeamLevel.SENIOR_A,
            official=True,
            metadata=_metadata(),
        )


def test_completed_historical_observation_can_be_date_only() -> None:
    observation = MatchObservation(
        source_match_id="history-1",
        match_date=date(2001, 9, 5),
        kickoff_at=None,
        home_team_name="Argentina",
        away_team_name="Brazil",
        competition_name="Friendly",
        neutral=False,
        status=MatchStatus.COMPLETED,
        gender=GenderCategory.MEN,
        team_level=TeamLevel.SENIOR_A,
        official=True,
        metadata=SourceMetadata(
            source="historical_results",
            available_at=datetime(2001, 9, 6, tzinfo=UTC),
            ingested_at=datetime(2026, 10, 7, tzinfo=UTC),
            leakage_risk=LeakageRisk.POST_MATCH_ONLY,
        ),
        home_score=2,
        away_score=1,
    )

    result = normalize_match_observation(
        observation,
        team_resolver=_team_resolver(),
        competition_resolver=_competition_resolver(),
    )

    assert result.record is not None
    assert result.record.match.kickoff_at is None
    assert result.record.match.time_precision is MatchTimePrecision.DATE_ONLY
