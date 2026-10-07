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
from football_analytics.features.rating import LegacyEloFeatureProvider

__all__ = [
    "FeatureDefinition",
    "FeatureLineage",
    "FeatureMissingReason",
    "FeatureProvider",
    "FeatureStatus",
    "FeatureValue",
    "FeatureVector",
    "LegacyEloFeatureProvider",
    "PredictionContext",
    "build_feature_vector",
]
