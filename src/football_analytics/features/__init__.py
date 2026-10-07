"""Versioned point-in-time feature contracts and providers."""

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
)
from football_analytics.features.fifa_ranking import (
    FifaRankingFeatureProvider,
    FifaRankingObservation,
)
from football_analytics.features.form import RollingFormFeatureProvider
from football_analytics.features.rating import LegacyEloFeatureProvider
from football_analytics.features.schedule import ScheduleRestFeatureProvider

__all__ = [
    "FeatureDefinition",
    "FeatureLineage",
    "FeatureMissingReason",
    "FeatureProvider",
    "FeatureStatus",
    "FeatureValue",
    "FeatureVector",
    "FifaRankingFeatureProvider",
    "FifaRankingObservation",
    "LegacyEloFeatureProvider",
    "PredictionContext",
    "RollingFormFeatureProvider",
    "ScheduleRestFeatureProvider",
    "build_feature_vector",
]
