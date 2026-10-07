"""Canonical data contracts, entity resolution, and temporal integrity helpers."""

from football_analytics.data.contracts import (
    LeakageRisk,
    PointInTimeRecord,
    SourceMetadata,
    TemporalIntegrityError,
    assert_pre_match_available,
)
from football_analytics.data.entity_resolution import (
    CompetitionEntityResolver,
    CompetitionResolution,
    ResolutionStatus,
    TeamEntityResolver,
    TeamResolution,
)
from football_analytics.data.normalization import (
    CanonicalMatchRecord,
    MatchNormalizationResult,
    NormalizationDecision,
    normalize_match_observation,
)
from football_analytics.data.observations import MatchObservation
from football_analytics.data.scope import (
    GenderCategory,
    ScopeAssessment,
    ScopeDecision,
    TeamLevel,
    assess_senior_mens_a_scope,
)
from football_analytics.data.sources import DataSourceDefinition, SourceRegistry

__all__ = [
    "CanonicalMatchRecord",
    "CompetitionEntityResolver",
    "CompetitionResolution",
    "DataSourceDefinition",
    "GenderCategory",
    "LeakageRisk",
    "MatchNormalizationResult",
    "MatchObservation",
    "NormalizationDecision",
    "PointInTimeRecord",
    "ResolutionStatus",
    "ScopeAssessment",
    "ScopeDecision",
    "SourceMetadata",
    "SourceRegistry",
    "TeamEntityResolver",
    "TeamLevel",
    "TeamResolution",
    "TemporalIntegrityError",
    "assert_pre_match_available",
    "assess_senior_mens_a_scope",
    "normalize_match_observation",
]
