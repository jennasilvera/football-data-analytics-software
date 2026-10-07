from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from football_analytics.data.contracts import LeakageRisk, SourceMetadata
from football_analytics.evaluation import OutcomeProbabilities
from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    PredictionContext,
)

MARKET_SNAPSHOT_FEATURE_VERSION = "market_snapshot_v1"


@dataclass(frozen=True, slots=True)
class MarketSnapshotObservation:
    """One point-in-time three-way market snapshot for a canonical match."""

    match_id: str
    market_source_id: str
    home_decimal_odds: float
    draw_decimal_odds: float
    away_decimal_odds: float
    metadata: SourceMetadata

    def __post_init__(self) -> None:
        match_id = self.match_id.strip()
        market_source_id = self.market_source_id.strip()

        if not match_id:
            raise ValueError("match_id must not be blank.")
        if not market_source_id:
            raise ValueError("market_source_id must not be blank.")

        odds = (
            float(self.home_decimal_odds),
            float(self.draw_decimal_odds),
            float(self.away_decimal_odds),
        )
        if any(not math.isfinite(value) or value <= 1.0 for value in odds):
            raise ValueError("Decimal odds must be finite and greater than 1.0.")

        object.__setattr__(self, "match_id", match_id)
        object.__setattr__(self, "market_source_id", market_source_id)
        object.__setattr__(self, "home_decimal_odds", odds[0])
        object.__setattr__(self, "draw_decimal_odds", odds[1])
        object.__setattr__(self, "away_decimal_odds", odds[2])


@dataclass(frozen=True, slots=True)
class DeviggedMarket:
    """Fair three-way probabilities plus the source market overround."""

    probabilities: OutcomeProbabilities
    overround: float


@dataclass(frozen=True, slots=True)
class _ResolvedMarket:
    latest: tuple[MarketSnapshotObservation, ...]
    opening: tuple[MarketSnapshotObservation, ...]
    missing_reason: FeatureMissingReason | None


class MarketSnapshotFeatureProvider:
    """Build cutoff-safe de-vigged market consensus and movement features."""

    provider_id = "market_snapshot_features"

    def __init__(
        self,
        observations: Sequence[MarketSnapshotObservation],
        *,
        version: str = MARKET_SNAPSHOT_FEATURE_VERSION,
    ) -> None:
        version = version.strip()
        if not version:
            raise ValueError("version must not be blank.")

        self._version = version
        self._by_match: dict[str, list[MarketSnapshotObservation]] = {}
        seen_timestamp: set[tuple[str, str, datetime]] = set()

        for observation in observations:
            available_at = observation.metadata.available_at
            if available_at is not None:
                key = (
                    observation.match_id,
                    observation.market_source_id,
                    available_at,
                )
                if key in seen_timestamp:
                    raise ValueError(
                        "Duplicate market snapshot timestamp for "
                        f"{observation.match_id}/{observation.market_source_id}: "
                        f"{available_at.isoformat()}"
                    )
                seen_timestamp.add(key)

            self._by_match.setdefault(
                observation.match_id,
                [],
            ).append(observation)

        for match_observations in self._by_match.values():
            match_observations.sort(key=_snapshot_sort_key)

        self._definitions = _definitions(version=version)

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        resolved = self._resolve(context)

        if not resolved.latest:
            return self._missing_values(
                reason=(
                    resolved.missing_reason
                    or FeatureMissingReason.UPSTREAM_UNAVAILABLE
                ),
                context=context,
            )

        latest_markets = tuple(devig_decimal_odds(snapshot) for snapshot in resolved.latest)
        opening_markets = tuple(devig_decimal_odds(snapshot) for snapshot in resolved.opening)

        latest_consensus = _mean_probabilities(
            tuple(market.probabilities for market in latest_markets)
        )
        opening_consensus = _mean_probabilities(
            tuple(market.probabilities for market in opening_markets)
        )
        average_overround = (
            sum(market.overround for market in latest_markets)
            / len(latest_markets)
        )

        latest_lineage = _lineage(resolved.latest)
        movement_lineage = _lineage(
            (*resolved.opening, *resolved.latest)
        )

        latest_available_times = tuple(
            snapshot.metadata.available_at
            for snapshot in resolved.latest
            if snapshot.metadata.available_at is not None
        )
        assert latest_available_times
        oldest_age_hours = max(
            (context.prediction_time - available_at).total_seconds() / 3600.0
            for available_at in latest_available_times
        )

        latest_values = {
            "home": latest_consensus.home_win,
            "draw": latest_consensus.draw,
            "away": latest_consensus.away_win,
        }
        opening_values = {
            "home": opening_consensus.home_win,
            "draw": opening_consensus.draw,
            "away": opening_consensus.away_win,
        }

        values: list[FeatureValue] = [
            FeatureValue(
                definition=self._definition("market.consensus.source_count"),
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=float(len(resolved.latest)),
                lineage=latest_lineage,
            ),
            FeatureValue(
                definition=self._definition("market.consensus.average_overround"),
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=average_overround,
                lineage=latest_lineage,
            ),
            FeatureValue(
                definition=self._definition(
                    "market.consensus.oldest_source_snapshot_age_hours"
                ),
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=oldest_age_hours,
                lineage=latest_lineage,
            ),
        ]

        for outcome in ("home", "draw", "away"):
            values.append(
                FeatureValue(
                    definition=self._definition(
                        f"market.consensus.{outcome}_fair_probability"
                    ),
                    status=FeatureStatus.OBSERVED,
                    as_of=context.prediction_time,
                    value=latest_values[outcome],
                    lineage=latest_lineage,
                )
            )
            values.append(
                FeatureValue(
                    definition=self._definition(
                        f"market.consensus.{outcome}_fair_probability_move_from_open"
                    ),
                    status=FeatureStatus.OBSERVED,
                    as_of=context.prediction_time,
                    value=latest_values[outcome] - opening_values[outcome],
                    lineage=movement_lineage,
                )
            )

        return tuple(values)

    def _resolve(
        self,
        context: PredictionContext,
    ) -> _ResolvedMarket:
        observations = self._by_match.get(context.match.match_id, [])
        eligible = [
            observation
            for observation in observations
            if observation.metadata.available_at is not None
            and observation.metadata.available_at <= context.prediction_time
            and observation.metadata.leakage_risk is LeakageRisk.SAFE
        ]

        if not eligible:
            return _ResolvedMarket(
                latest=(),
                opening=(),
                missing_reason=(
                    FeatureMissingReason.TEMPORAL_INTEGRITY
                    if observations
                    else FeatureMissingReason.UPSTREAM_UNAVAILABLE
                ),
            )

        by_source: dict[str, list[MarketSnapshotObservation]] = {}
        for observation in eligible:
            by_source.setdefault(
                observation.market_source_id,
                [],
            ).append(observation)

        latest: list[MarketSnapshotObservation] = []
        opening: list[MarketSnapshotObservation] = []

        for source_observations in by_source.values():
            source_observations.sort(key=_snapshot_sort_key)
            opening.append(source_observations[0])
            latest.append(source_observations[-1])

        latest.sort(key=lambda observation: observation.market_source_id)
        opening.sort(key=lambda observation: observation.market_source_id)

        return _ResolvedMarket(
            latest=tuple(latest),
            opening=tuple(opening),
            missing_reason=None,
        )

    def _missing_values(
        self,
        *,
        reason: FeatureMissingReason,
        context: PredictionContext,
    ) -> tuple[FeatureValue, ...]:
        return tuple(
            FeatureValue(
                definition=definition,
                status=FeatureStatus.MISSING,
                as_of=context.prediction_time,
                missing_reason=reason,
            )
            for definition in self._definitions
        )

    def _definition(self, name: str) -> FeatureDefinition:
        for definition in self._definitions:
            if definition.name == name:
                return definition
        raise KeyError(name)


def devig_decimal_odds(
    snapshot: MarketSnapshotObservation,
) -> DeviggedMarket:
    """Convert decimal 1X2 odds into normalized fair probabilities."""

    raw = (
        1.0 / snapshot.home_decimal_odds,
        1.0 / snapshot.draw_decimal_odds,
        1.0 / snapshot.away_decimal_odds,
    )
    implied_total = sum(raw)
    probabilities = OutcomeProbabilities(
        home_win=raw[0] / implied_total,
        draw=raw[1] / implied_total,
        away_win=raw[2] / implied_total,
    )
    return DeviggedMarket(
        probabilities=probabilities,
        overround=implied_total - 1.0,
    )


def _mean_probabilities(
    probabilities: tuple[OutcomeProbabilities, ...],
) -> OutcomeProbabilities:
    if not probabilities:
        raise ValueError("At least one probability distribution is required.")

    count = len(probabilities)
    return OutcomeProbabilities(
        home_win=sum(item.home_win for item in probabilities) / count,
        draw=sum(item.draw for item in probabilities) / count,
        away_win=sum(item.away_win for item in probabilities) / count,
    )


def _lineage(
    observations: Sequence[MarketSnapshotObservation],
) -> FeatureLineage:
    return FeatureLineage(
        source_record_ids=tuple(
            sorted(
                {
                    observation.metadata.source_record_id
                    for observation in observations
                    if observation.metadata.source_record_id is not None
                }
            )
        )
    )


def _snapshot_sort_key(
    observation: MarketSnapshotObservation,
) -> tuple[object, ...]:
    available_at = observation.metadata.available_at
    return (
        available_at is None,
        available_at.isoformat() if available_at is not None else "",
        observation.market_source_id,
        observation.metadata.source_record_id or "",
    )


def _definitions(
    *,
    version: str,
) -> tuple[FeatureDefinition, ...]:
    specs = [
        (
            "market.consensus.source_count",
            "Number of market sources represented in the cutoff-safe consensus.",
        ),
        (
            "market.consensus.average_overround",
            "Mean raw implied-probability excess across latest source snapshots.",
        ),
        (
            "market.consensus.oldest_source_snapshot_age_hours",
            "Age in hours of the stalest latest source snapshot in the consensus.",
        ),
    ]

    for outcome in ("home", "draw", "away"):
        specs.extend(
            [
                (
                    f"market.consensus.{outcome}_fair_probability",
                    (
                        "Mean de-vigged latest market probability for "
                        f"{outcome}."
                    ),
                ),
                (
                    (
                        "market.consensus."
                        f"{outcome}_fair_probability_move_from_open"
                    ),
                    (
                        "Consensus de-vigged probability movement from each "
                        f"source's opening snapshot for {outcome}."
                    ),
                ),
            ]
        )

    return tuple(
        FeatureDefinition(
            name=name,
            version=version,
            group="market_context",
            description=description,
        )
        for name, description in specs
    )
