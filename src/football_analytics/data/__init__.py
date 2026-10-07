"""Canonical data contracts, entity resolution, and temporal integrity helpers."""

from football_analytics.data.artifacts import (
    normalization_frames,
    save_normalization_artifacts,
)
from football_analytics.data.batch import (
    BatchNormalizationReport,
    normalize_match_batch,
)
from football_analytics.data.catalogs import CanonicalCatalogs, load_canonical_catalogs
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
from football_analytics.data.snapshots import DataSnapshot, dataframe_snapshot
from football_analytics.data.sources import DataSourceDefinition, SourceRegistry

__all__ = [
    "BatchNormalizationReport",
    "CanonicalCatalogs",
    "CanonicalMatchRecord",
    "CompetitionEntityResolver",
    "CompetitionResolution",
    "DataSnapshot",
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
    "dataframe_snapshot",
    "load_canonical_catalogs",
    "normalization_frames",
    "normalize_match_batch",
    "normalize_match_observation",
    "save_normalization_artifacts",
]
