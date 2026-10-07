from __future__ import annotations

from football_analytics.features import (
    CompetitionContextFeatureProvider,
    MarketSnapshotFeatureProvider,
    SquadAvailabilityFeatureProvider,
    TravelContextFeatureProvider,
)


def test_feature_package_exports_integrated_context_providers() -> None:
    assert CompetitionContextFeatureProvider.provider_id == "competition_context_features"
    assert MarketSnapshotFeatureProvider.provider_id == "market_snapshot_features"
    assert SquadAvailabilityFeatureProvider.provider_id == "squad_availability_features"
    assert TravelContextFeatureProvider.provider_id == "travel_context_features"
