from __future__ import annotations

from dataclasses import dataclass

from football_analytics.data.entity_resolution import (
    CompetitionEntityResolver,
    TeamEntityResolver,
)
from football_analytics.data.normalization import (
    CanonicalMatchRecord,
    MatchNormalizationResult,
    NormalizationDecision,
    normalize_match_observation,
)
from football_analytics.data.observations import MatchObservation


@dataclass(frozen=True, slots=True)
class BatchNormalizationReport:
    """Partitioned result of normalizing one source batch."""

    normalized: tuple[CanonicalMatchRecord, ...]
    excluded: tuple[MatchNormalizationResult, ...]
    quarantined: tuple[MatchNormalizationResult, ...]

    @property
    def total_rows(self) -> int:
        return len(self.normalized) + len(self.excluded) + len(self.quarantined)


def normalize_match_batch(
    observations: list[MatchObservation],
    *,
    team_resolver: TeamEntityResolver,
    competition_resolver: CompetitionEntityResolver,
) -> BatchNormalizationReport:
    """Normalize a batch and quarantine duplicate or reversed canonical fixtures."""

    normalized: list[CanonicalMatchRecord] = []
    excluded: list[MatchNormalizationResult] = []
    quarantined: list[MatchNormalizationResult] = []
    seen_identity: dict[tuple[str, tuple[str, str], str], CanonicalMatchRecord] = {}

    for observation in observations:
        result = normalize_match_observation(
            observation,
            team_resolver=team_resolver,
            competition_resolver=competition_resolver,
        )

        if result.decision is NormalizationDecision.EXCLUDED:
            excluded.append(result)
            continue

        if result.decision is NormalizationDecision.QUARANTINED:
            quarantined.append(result)
            continue

        assert result.record is not None
        record = result.record
        team_ids = sorted(
            [record.match.home_team_id, record.match.away_team_id]
        )
        team_pair = (team_ids[0], team_ids[1])
        identity = (
            record.match.match_date.isoformat(),
            team_pair,
            record.match.competition_id,
        )

        prior = seen_identity.get(identity)
        if prior is not None:
            quarantined.append(
                MatchNormalizationResult(
                    decision=NormalizationDecision.QUARANTINED,
                    reasons=(
                        "duplicate_or_reversed_fixture:"
                        f"{prior.source_match_id}",
                    ),
                    source=observation.metadata.source,
                    source_match_id=observation.source_match_id,
                )
            )
            continue

        seen_identity[identity] = record
        normalized.append(record)

    return BatchNormalizationReport(
        normalized=tuple(normalized),
        excluded=tuple(excluded),
        quarantined=tuple(quarantined),
    )
