from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from football_analytics.data.contracts import LeakageRisk, SourceMetadata
from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    PredictionContext,
)

FIFA_RANKING_FEATURE_VERSION = "fifa_ranking_v1"


@dataclass(frozen=True, slots=True)
class FifaRankingObservation:
    """One canonical FIFA ranking observation with point-in-time metadata."""

    team_id: str
    rank: int
    points: float | None
    metadata: SourceMetadata

    def __post_init__(self) -> None:
        team_id = self.team_id.strip()
        if not team_id:
            raise ValueError("team_id must not be blank.")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer.")
        if self.points is not None and (not math.isfinite(self.points) or self.points < 0):
            raise ValueError("points must be finite and nonnegative.")
        object.__setattr__(self, "team_id", team_id)


@dataclass(frozen=True, slots=True)
class _ResolvedRanking:
    observation: FifaRankingObservation | None
    missing_reason: FeatureMissingReason | None


class FifaRankingFeatureProvider:
    """Build point-in-time FIFA ranking features from timestamped observations."""

    provider_id = "fifa_ranking_features"

    def __init__(
        self,
        observations: Sequence[FifaRankingObservation],
        *,
        version: str = FIFA_RANKING_FEATURE_VERSION,
        max_age_days: int | None = None,
    ) -> None:
        if max_age_days is not None and max_age_days <= 0:
            raise ValueError("max_age_days must be positive when supplied.")

        self._version = version.strip()
        if not self._version:
            raise ValueError("version must not be blank.")

        self._max_age_days = max_age_days
        self._by_team: dict[str, list[FifaRankingObservation]] = {}
        seen_timestamp: set[tuple[str, object]] = set()

        for observation in observations:
            available_at = observation.metadata.available_at

            if available_at is not None:
                key = (observation.team_id, available_at)
                if key in seen_timestamp:
                    raise ValueError(
                        "Duplicate FIFA ranking availability timestamp for "
                        f"{observation.team_id}: {available_at.isoformat()}"
                    )
                seen_timestamp.add(key)

            self._by_team.setdefault(observation.team_id, []).append(observation)

        for team_observations in self._by_team.values():
            team_observations.sort(
                key=lambda observation: (
                    observation.metadata.available_at is None,
                    (
                        observation.metadata.available_at.isoformat()
                        if observation.metadata.available_at is not None
                        else ""
                    ),
                    observation.metadata.source_record_id or "",
                )
            )

        self._definitions = _definitions(version=self._version)

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        home = self._resolve(
            context.match.home_team_id,
            context=context,
        )
        away = self._resolve(
            context.match.away_team_id,
            context=context,
        )

        home_rank = self._rank_value(
            name="fifa.home.rank",
            resolved=home,
            context=context,
        )
        away_rank = self._rank_value(
            name="fifa.away.rank",
            resolved=away,
            context=context,
        )
        home_points = self._points_value(
            name="fifa.home.points",
            resolved=home,
            context=context,
        )
        away_points = self._points_value(
            name="fifa.away.points",
            resolved=away,
            context=context,
        )

        return (
            home_rank,
            away_rank,
            _difference(
                definition=self._definition(
                    "fifa.rank_diff_home_minus_away"
                ),
                home=home_rank,
                away=away_rank,
                context=context,
            ),
            home_points,
            away_points,
            _difference(
                definition=self._definition(
                    "fifa.points_diff_home_minus_away"
                ),
                home=home_points,
                away=away_points,
                context=context,
            ),
        )

    def _resolve(
        self,
        team_id: str,
        *,
        context: PredictionContext,
    ) -> _ResolvedRanking:
        all_observations = self._by_team.get(team_id, [])
        eligible = [
            observation
            for observation in all_observations
            if observation.metadata.available_at is not None
            and observation.metadata.available_at <= context.prediction_time
            and observation.metadata.leakage_risk is LeakageRisk.SAFE
        ]

        if not eligible:
            reason = (
                FeatureMissingReason.TEMPORAL_INTEGRITY
                if all_observations
                else FeatureMissingReason.UPSTREAM_UNAVAILABLE
            )
            return _ResolvedRanking(
                observation=None,
                missing_reason=reason,
            )

        observation = eligible[-1]

        if self._max_age_days is not None:
            assert observation.metadata.available_at is not None
            max_age = timedelta(days=self._max_age_days)
            if context.prediction_time - observation.metadata.available_at > max_age:
                return _ResolvedRanking(
                    observation=None,
                    missing_reason=FeatureMissingReason.STALE,
                )

        return _ResolvedRanking(
            observation=observation,
            missing_reason=None,
        )

    def _rank_value(
        self,
        *,
        name: str,
        resolved: _ResolvedRanking,
        context: PredictionContext,
    ) -> FeatureValue:
        if resolved.observation is None:
            return _missing_value(
                definition=self._definition(name),
                reason=resolved.missing_reason
                or FeatureMissingReason.UPSTREAM_UNAVAILABLE,
                context=context,
            )

        return FeatureValue(
            definition=self._definition(name),
            status=FeatureStatus.OBSERVED,
            as_of=context.prediction_time,
            value=float(resolved.observation.rank),
            lineage=_ranking_lineage(resolved.observation),
        )

    def _points_value(
        self,
        *,
        name: str,
        resolved: _ResolvedRanking,
        context: PredictionContext,
    ) -> FeatureValue:
        if resolved.observation is None:
            return _missing_value(
                definition=self._definition(name),
                reason=resolved.missing_reason
                or FeatureMissingReason.UPSTREAM_UNAVAILABLE,
                context=context,
            )

        if resolved.observation.points is None:
            return FeatureValue(
                definition=self._definition(name),
                status=FeatureStatus.MISSING,
                as_of=context.prediction_time,
                missing_reason=FeatureMissingReason.UPSTREAM_UNAVAILABLE,
                lineage=_ranking_lineage(resolved.observation),
            )

        return FeatureValue(
            definition=self._definition(name),
            status=FeatureStatus.OBSERVED,
            as_of=context.prediction_time,
            value=float(resolved.observation.points),
            lineage=_ranking_lineage(resolved.observation),
        )

    def _definition(self, name: str) -> FeatureDefinition:
        for definition in self._definitions:
            if definition.name == name:
                return definition
        raise KeyError(name)


def _ranking_lineage(
    observation: FifaRankingObservation,
) -> FeatureLineage:
    source_record_id = observation.metadata.source_record_id
    return FeatureLineage(
        source_record_ids=(source_record_id,)
        if source_record_id is not None
        else ()
    )


def _missing_value(
    *,
    definition: FeatureDefinition,
    reason: FeatureMissingReason,
    context: PredictionContext,
) -> FeatureValue:
    return FeatureValue(
        definition=definition,
        status=FeatureStatus.MISSING,
        as_of=context.prediction_time,
        missing_reason=reason,
    )


def _difference(
    *,
    definition: FeatureDefinition,
    home: FeatureValue,
    away: FeatureValue,
    context: PredictionContext,
) -> FeatureValue:
    lineage = FeatureLineage(
        source_record_ids=tuple(
            sorted(
                set(
                    home.lineage.source_record_ids
                    + away.lineage.source_record_ids
                )
            )
        )
    )

    if home.value is None or away.value is None:
        reason = home.missing_reason or away.missing_reason
        return FeatureValue(
            definition=definition,
            status=FeatureStatus.MISSING,
            as_of=context.prediction_time,
            missing_reason=reason or FeatureMissingReason.UPSTREAM_UNAVAILABLE,
            lineage=lineage,
        )

    return FeatureValue(
        definition=definition,
        status=FeatureStatus.OBSERVED,
        as_of=context.prediction_time,
        value=float(home.value - away.value),
        lineage=lineage,
    )


def _definitions(
    *,
    version: str,
) -> tuple[FeatureDefinition, ...]:
    specs = [
        ("fifa.home.rank", "Latest home-team FIFA rank at the prediction cutoff."),
        ("fifa.away.rank", "Latest away-team FIFA rank at the prediction cutoff."),
        (
            "fifa.rank_diff_home_minus_away",
            "Home FIFA rank minus away FIFA rank.",
        ),
        (
            "fifa.home.points",
            "Latest home-team FIFA ranking points at the prediction cutoff.",
        ),
        (
            "fifa.away.points",
            "Latest away-team FIFA ranking points at the prediction cutoff.",
        ),
        (
            "fifa.points_diff_home_minus_away",
            "Home FIFA ranking points minus away FIFA ranking points.",
        ),
    ]

    return tuple(
        FeatureDefinition(
            name=name,
            version=version,
            group="team_strength",
            description=description,
        )
        for name, description in specs
    )


class FifaRankingTrendProvider(FifaRankingFeatureProvider):
    """Opt-in adjacent-publication changes and declared ordinal rank buckets.

    Points comparisons require the same source; providers must split source IDs
    when their points methodology changes. No older non-null points are substituted.
    """

    provider_id = "fifa_ranking_trends_v1"

    def __init__(self, observations: Sequence[FifaRankingObservation], **kwargs) -> None:
        super().__init__(observations, **kwargs)
        descriptions = {
            "rank_bucket": "Ordinal bucket: 1=1..10, 2=11..25, 3=26..50, 4=51..100, 5=101+.",
            "rank_improvement": "Previous rank minus latest rank; positive means improvement.",
            "points_change": "Latest points minus previous points from adjacent publications.",
            "publication_gap_days": "Days between adjacent eligible ranking publications.",
        }
        self._definitions = tuple(
            FeatureDefinition(
                f"fifa_trend.{side}.{metric}", self.provider_id, "team_strength", description
            )
            for side in ("home", "away")
            for metric, description in descriptions.items()
        )

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        result = []
        for side, team_id in (
            ("home", context.match.home_team_id), ("away", context.match.away_team_id)
        ):
            resolved = self._resolve(team_id, context=context)
            latest = resolved.observation
            previous = None
            if latest is not None:
                assert latest.metadata.available_at is not None
                candidates = [
                    row for row in self._by_team[team_id]
                    if row.metadata.available_at is not None
                    and row.metadata.available_at < latest.metadata.available_at
                    and row.metadata.leakage_risk is LeakageRisk.SAFE
                ]
                previous = candidates[-1] if candidates else None
                if previous and previous.metadata.source != latest.metadata.source:
                    previous = None
            for metric in (
                "rank_bucket", "rank_improvement", "points_change", "publication_gap_days"
            ):
                value: float | None = None
                used = [latest] if latest else []
                reason = resolved.missing_reason or FeatureMissingReason.NO_HISTORY
                if latest and metric == "rank_bucket":
                    value = 1 + sum(latest.rank > bound for bound in (10, 25, 50, 100))
                elif latest and previous:
                    used.append(previous)
                    if metric == "rank_improvement":
                        value = previous.rank - latest.rank
                    elif metric == "points_change":
                        if latest.points is not None and previous.points is not None:
                            value = latest.points - previous.points
                        else:
                            reason = FeatureMissingReason.UPSTREAM_UNAVAILABLE
                    else:
                        assert latest.metadata.available_at is not None
                        assert previous.metadata.available_at is not None
                        value = (
                            latest.metadata.available_at - previous.metadata.available_at
                        ).total_seconds() / 86400
                result.append(FeatureValue(
                    self._definition(f"fifa_trend.{side}.{metric}"),
                    FeatureStatus.OBSERVED if value is not None else FeatureStatus.MISSING,
                    context.prediction_time,
                    float(value) if value is not None else None,
                    missing_reason=None if value is not None else reason,
                    lineage=FeatureLineage(source_record_ids=tuple(sorted({
                        row.metadata.source_record_id for row in used
                        if row.metadata.source_record_id is not None
                    }))),
                ))
        return tuple(result)
