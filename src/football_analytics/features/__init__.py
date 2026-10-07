"""Versioned point-in-time feature contracts and providers."""

from football_analytics.features.artifacts import (
    FeatureVectorArtifact,
    build_feature_vector_artifact,
    load_feature_vector_artifact,
    save_feature_vector_artifact,
)
from football_analytics.features.availability import (
    PlayerAvailabilityObservation,
    PlayerAvailabilityStatus,
    SquadAvailabilityFeatureProvider,
)
from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureProvider,
    FeatureStatus,
    FeatureValue,
    FeatureVector,
    PredictionContext,
    build_feature_vector,
    feature_set_id_for_definitions,
)
from football_analytics.features.dataset import (
    DEFAULT_HISTORICAL_CUTOFF_POLICY,
    HistoricalFeatureDataset,
    HistoricalFeatureExample,
    HistoricalFeatureLeakageError,
    PredictionCutoffPolicy,
    build_historical_feature_dataset,
)
from football_analytics.features.fifa_ranking import (
    FifaRankingFeatureProvider,
    FifaRankingObservation,
)
from football_analytics.features.form import RollingFormFeatureProvider
from football_analytics.features.materialization import (
    AppliedImputation,
    ConstantImputationRule,
    ImputationPolicy,
    MaterializedFeatureRow,
    MissingFeaturePolicyError,
    materialize_feature_vector,
)
from football_analytics.features.rating import LegacyEloFeatureProvider
from football_analytics.features.schedule import ScheduleRestFeatureProvider
from football_analytics.features.travel import (
    TeamReferenceLocationObservation,
    TravelContextFeatureProvider,
    VenueLocationObservation,
)

__all__ = [
    "AppliedImputation",
    "ConstantImputationRule",
    "DEFAULT_HISTORICAL_CUTOFF_POLICY",
    "FeatureDefinition",
    "FeatureLineage",
    "FeatureMissingReason",
    "FeatureProvider",
    "FeatureStatus",
    "FeatureValue",
    "FeatureVector",
    "FeatureVectorArtifact",
    "FifaRankingFeatureProvider",
    "FifaRankingObservation",
    "HistoricalFeatureDataset",
    "HistoricalFeatureExample",
    "HistoricalFeatureLeakageError",
    "ImputationPolicy",
    "LegacyEloFeatureProvider",
    "MaterializedFeatureRow",
    "MissingFeaturePolicyError",
    "PlayerAvailabilityObservation",
    "PlayerAvailabilityStatus",
    "PredictionContext",
    "PredictionCutoffPolicy",
    "RollingFormFeatureProvider",
    "ScheduleRestFeatureProvider",
    "SquadAvailabilityFeatureProvider",
    "TeamReferenceLocationObservation",
    "TravelContextFeatureProvider",
    "VenueLocationObservation",
    "build_feature_vector",
    "build_feature_vector_artifact",
    "build_historical_feature_dataset",
    "feature_set_id_for_definitions",
    "load_feature_vector_artifact",
    "materialize_feature_vector",
    "save_feature_vector_artifact",
]
