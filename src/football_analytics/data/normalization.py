from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from football_analytics.data.contracts import SourceMetadata
from football_analytics.data.entity_resolution import (
    CompetitionEntityResolver,
    ResolutionStatus,
    TeamEntityResolver,
)
from football_analytics.data.observations import MatchObservation
from football_analytics.data.scope import (
    ScopeDecision,
    assess_senior_mens_a_scope,
)
from football_analytics.domain import Match


class NormalizationDecision(StrEnum):
    """Disposition of a source observation during canonicalization."""

    NORMALIZED = "normalized"
    EXCLUDED = "excluded"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class CanonicalMatchRecord:
    """Canonical match plus source/result fields required for downstream lineage."""

    match: Match
    metadata: SourceMetadata
    source_match_id: str
    source_home_team_name: str
    source_away_team_name: str
    source_competition_name: str
    home_score: int | None = None
    away_score: int | None = None
    home_resolution_method: str | None = None
    away_resolution_method: str | None = None
    competition_resolution_method: str | None = None


@dataclass(frozen=True, slots=True)
class MatchNormalizationResult:
    decision: NormalizationDecision
    reasons: tuple[str, ...]
    record: CanonicalMatchRecord | None = None


def normalize_match_observation(
    observation: MatchObservation,
    *,
    team_resolver: TeamEntityResolver,
    competition_resolver: CompetitionEntityResolver,
) -> MatchNormalizationResult:
    """Normalize a source observation without guessing unresolved identities."""

    scope = assess_senior_mens_a_scope(
        gender=observation.gender,
        team_level=observation.team_level,
        official=observation.official,
    )

    if scope.decision is ScopeDecision.EXCLUDE:
        return MatchNormalizationResult(
            decision=NormalizationDecision.EXCLUDED,
            reasons=(scope.reason,),
        )

    if scope.decision is ScopeDecision.REVIEW:
        return MatchNormalizationResult(
            decision=NormalizationDecision.QUARANTINED,
            reasons=(scope.reason,),
        )

    home = team_resolver.resolve(observation.home_team_name)
    away = team_resolver.resolve(observation.away_team_name)
    competition = competition_resolver.resolve(observation.competition_name)

    unresolved_reasons: list[str] = []

    if home.status is ResolutionStatus.UNRESOLVED:
        unresolved_reasons.append(
            f"unresolved_home_team:{observation.home_team_name}"
        )

    if away.status is ResolutionStatus.UNRESOLVED:
        unresolved_reasons.append(
            f"unresolved_away_team:{observation.away_team_name}"
        )

    if competition.status is ResolutionStatus.UNRESOLVED:
        unresolved_reasons.append(
            f"unresolved_competition:{observation.competition_name}"
        )

    if unresolved_reasons:
        return MatchNormalizationResult(
            decision=NormalizationDecision.QUARANTINED,
            reasons=tuple(unresolved_reasons),
        )

    assert home.team is not None
    assert away.team is not None
    assert competition.competition is not None

    if not competition.competition.senior_mens_a_international:
        return MatchNormalizationResult(
            decision=NormalizationDecision.EXCLUDED,
            reasons=("competition_out_of_scope",),
        )

    match = Match(
        match_id=build_canonical_match_id(
            kickoff_at=observation.kickoff_at.isoformat(),
            home_team_id=home.team.team_id,
            away_team_id=away.team.team_id,
            competition_id=competition.competition.competition_id,
        ),
        kickoff_at=observation.kickoff_at,
        home_team_id=home.team.team_id,
        away_team_id=away.team.team_id,
        competition_id=competition.competition.competition_id,
        neutral=observation.neutral,
        status=observation.status,
    )

    return MatchNormalizationResult(
        decision=NormalizationDecision.NORMALIZED,
        reasons=("canonicalized",),
        record=CanonicalMatchRecord(
            match=match,
            metadata=observation.metadata,
            source_match_id=observation.source_match_id,
            source_home_team_name=observation.home_team_name,
            source_away_team_name=observation.away_team_name,
            source_competition_name=observation.competition_name,
            home_score=observation.home_score,
            away_score=observation.away_score,
            home_resolution_method=home.matched_by,
            away_resolution_method=away.matched_by,
            competition_resolution_method=competition.matched_by,
        ),
    )


def build_canonical_match_id(
    *,
    kickoff_at: str,
    home_team_id: str,
    away_team_id: str,
    competition_id: str,
) -> str:
    """Build a deterministic ID from canonical match identity fields."""

    key = "|".join(
        [
            kickoff_at.strip(),
            home_team_id.strip(),
            away_team_id.strip(),
            competition_id.strip(),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return f"match_{digest}"
