from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from football_analytics.data.contracts import LeakageRisk, SourceMetadata
from football_analytics.domain import Competition, CompetitionKind, CompetitionStage
from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    PredictionContext,
)

COMPETITION_CONTEXT_FEATURE_VERSION = "competition_context_v1"


@dataclass(frozen=True, slots=True)
class MatchCompetitionContextObservation:
    """Point-in-time stage metadata for one canonical match."""

    match_id: str
    stage: CompetitionStage
    metadata: SourceMetadata
    leg_number: int | None = None

    def __post_init__(self) -> None:
        match_id = self.match_id.strip()
        if not match_id:
            raise ValueError("match_id must not be blank.")

        if self.leg_number is not None and self.leg_number <= 0:
            raise ValueError("leg_number must be positive when supplied.")

        object.__setattr__(self, "match_id", match_id)


@dataclass(frozen=True, slots=True)
class _ResolvedStage:
    observation: MatchCompetitionContextObservation | None
    missing_reason: FeatureMissingReason | None


class CompetitionContextFeatureProvider:
    """Build canonical competition-kind and cutoff-safe match-stage features."""

    provider_id = "competition_context_features"

    def __init__(
        self,
        competitions: Sequence[Competition],
        *,
        match_context: Sequence[MatchCompetitionContextObservation] = (),
        version: str = COMPETITION_CONTEXT_FEATURE_VERSION,
    ) -> None:
        version = version.strip()
        if not version:
            raise ValueError("version must not be blank.")

        self._version = version
        self._competitions: dict[str, Competition] = {}
        for competition in competitions:
            if competition.competition_id in self._competitions:
                raise ValueError(
                    f"Duplicate competition_id: {competition.competition_id}"
                )
            self._competitions[competition.competition_id] = competition

        self._context_by_match: dict[
            str,
            list[MatchCompetitionContextObservation],
        ] = {}
        seen_availability: set[tuple[str, object]] = set()

        for observation in match_context:
            available_at = observation.metadata.available_at
            if available_at is not None:
                key = (observation.match_id, available_at)
                if key in seen_availability:
                    raise ValueError(
                        "Duplicate competition-context availability timestamp for "
                        f"{observation.match_id}: {available_at.isoformat()}"
                    )
                seen_availability.add(key)

            self._context_by_match.setdefault(
                observation.match_id,
                [],
            ).append(observation)

        for observations in self._context_by_match.values():
            observations.sort(key=_observation_sort_key)

        self._definitions = _definitions(version=version)

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        competition = self._competitions.get(context.match.competition_id)
        stage = self._resolve_stage(context)

        return (
            *self._competition_kind_values(
                competition=competition,
                context=context,
            ),
            self._confederation_scope_value(
                competition=competition,
                context=context,
            ),
            *self._stage_values(
                resolved=stage,
                context=context,
            ),
            self._leg_number_value(
                resolved=stage,
                context=context,
            ),
        )

    def _competition_kind_values(
        self,
        *,
        competition: Competition | None,
        context: PredictionContext,
    ) -> tuple[FeatureValue, ...]:
        if competition is None:
            return tuple(
                FeatureValue(
                    definition=self._definition(
                        f"competition.kind.{kind.value}"
                    ),
                    status=FeatureStatus.MISSING,
                    as_of=context.prediction_time,
                    missing_reason=FeatureMissingReason.UPSTREAM_UNAVAILABLE,
                )
                for kind in CompetitionKind
            )

        return tuple(
            FeatureValue(
                definition=self._definition(
                    f"competition.kind.{kind.value}"
                ),
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=1.0 if competition.kind is kind else 0.0,
            )
            for kind in CompetitionKind
        )

    def _confederation_scope_value(
        self,
        *,
        competition: Competition | None,
        context: PredictionContext,
    ) -> FeatureValue:
        definition = self._definition(
            "competition.confederation_specific"
        )

        if competition is None:
            return FeatureValue(
                definition=definition,
                status=FeatureStatus.MISSING,
                as_of=context.prediction_time,
                missing_reason=FeatureMissingReason.UPSTREAM_UNAVAILABLE,
            )

        return FeatureValue(
            definition=definition,
            status=FeatureStatus.OBSERVED,
            as_of=context.prediction_time,
            value=1.0 if competition.confederation is not None else 0.0,
        )

    def _resolve_stage(
        self,
        context: PredictionContext,
    ) -> _ResolvedStage:
        observations = self._context_by_match.get(
            context.match.match_id,
            [],
        )
        eligible = [
            observation
            for observation in observations
            if observation.metadata.available_at is not None
            and observation.metadata.available_at <= context.prediction_time
            and observation.metadata.leakage_risk is LeakageRisk.SAFE
        ]

        if not eligible:
            return _ResolvedStage(
                observation=None,
                missing_reason=(
                    FeatureMissingReason.TEMPORAL_INTEGRITY
                    if observations
                    else FeatureMissingReason.UPSTREAM_UNAVAILABLE
                ),
            )

        return _ResolvedStage(
            observation=eligible[-1],
            missing_reason=None,
        )

    def _stage_values(
        self,
        *,
        resolved: _ResolvedStage,
        context: PredictionContext,
    ) -> tuple[FeatureValue, ...]:
        if resolved.observation is None:
            return tuple(
                FeatureValue(
                    definition=self._definition(
                        f"competition.stage.{stage.value}"
                    ),
                    status=FeatureStatus.MISSING,
                    as_of=context.prediction_time,
                    missing_reason=(
                        resolved.missing_reason
                        or FeatureMissingReason.UPSTREAM_UNAVAILABLE
                    ),
                )
                for stage in CompetitionStage
            )

        lineage = _lineage(resolved.observation)

        return tuple(
            FeatureValue(
                definition=self._definition(
                    f"competition.stage.{stage.value}"
                ),
                status=FeatureStatus.OBSERVED,
                as_of=context.prediction_time,
                value=1.0 if resolved.observation.stage is stage else 0.0,
                lineage=lineage,
            )
            for stage in CompetitionStage
        )

    def _leg_number_value(
        self,
        *,
        resolved: _ResolvedStage,
        context: PredictionContext,
    ) -> FeatureValue:
        definition = self._definition("competition.leg_number")

        if resolved.observation is None:
            return FeatureValue(
                definition=definition,
                status=FeatureStatus.MISSING,
                as_of=context.prediction_time,
                missing_reason=(
                    resolved.missing_reason
                    or FeatureMissingReason.UPSTREAM_UNAVAILABLE
                ),
            )

        if resolved.observation.leg_number is None:
            return FeatureValue(
                definition=definition,
                status=FeatureStatus.MISSING,
                as_of=context.prediction_time,
                missing_reason=FeatureMissingReason.NOT_APPLICABLE,
                lineage=_lineage(resolved.observation),
            )

        return FeatureValue(
            definition=definition,
            status=FeatureStatus.OBSERVED,
            as_of=context.prediction_time,
            value=float(resolved.observation.leg_number),
            lineage=_lineage(resolved.observation),
        )

    def _definition(self, name: str) -> FeatureDefinition:
        for definition in self._definitions:
            if definition.name == name:
                return definition
        raise KeyError(name)


def _lineage(
    observation: MatchCompetitionContextObservation,
) -> FeatureLineage:
    source_record_id = observation.metadata.source_record_id
    return FeatureLineage(
        source_record_ids=(source_record_id,)
        if source_record_id is not None
        else ()
    )


def _observation_sort_key(
    observation: MatchCompetitionContextObservation,
) -> tuple[object, ...]:
    available_at = observation.metadata.available_at
    return (
        available_at is None,
        available_at.isoformat() if available_at is not None else "",
        observation.metadata.source_record_id or "",
    )


def _definitions(
    *,
    version: str,
) -> tuple[FeatureDefinition, ...]:
    definitions: list[FeatureDefinition] = []

    for kind in CompetitionKind:
        definitions.append(
            FeatureDefinition(
                name=f"competition.kind.{kind.value}",
                version=version,
                group="competition_context",
                description=(
                    "One-hot canonical competition kind: "
                    f"{kind.value}."
                ),
            )
        )

    definitions.append(
        FeatureDefinition(
            name="competition.confederation_specific",
            version=version,
            group="competition_context",
            description=(
                "Whether the canonical competition is scoped to one "
                "confederation."
            ),
        )
    )

    for stage in CompetitionStage:
        definitions.append(
            FeatureDefinition(
                name=f"competition.stage.{stage.value}",
                version=version,
                group="competition_context",
                description=(
                    "One-hot point-in-time match stage: "
                    f"{stage.value}."
                ),
            )
        )

    definitions.append(
        FeatureDefinition(
            name="competition.leg_number",
            version=version,
            group="competition_context",
            description=(
                "Explicit leg number for multi-leg competition context when "
                "the source provides one."
            ),
        )
    )

    return tuple(definitions)
