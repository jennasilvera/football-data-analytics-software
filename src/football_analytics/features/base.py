from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from football_analytics.data.contracts import TemporalIntegrityError, ensure_utc
from football_analytics.domain import Match


class FeatureStatus(StrEnum):
    """How a model feature value was obtained."""

    OBSERVED = "observed"
    MISSING = "missing"
    IMPUTED = "imputed"


class FeatureMissingReason(StrEnum):
    """Why a feature is unavailable or required imputation."""

    NO_HISTORY = "no_history"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    STALE = "stale"
    TEMPORAL_INTEGRITY = "temporal_integrity"
    TEMPORAL_PRECISION = "temporal_precision"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Stable versioned definition of one model feature."""

    name: str
    version: str
    group: str
    description: str

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "group", "description"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must not be blank.")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class FeatureLineage:
    """Lineage references for a derived feature value."""

    source_record_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """One explicit feature value, including missingness and lineage."""

    definition: FeatureDefinition
    status: FeatureStatus
    as_of: datetime
    value: float | None = None
    missing_reason: FeatureMissingReason | None = None
    imputation_method: str | None = None
    lineage: FeatureLineage = FeatureLineage()

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", ensure_utc(self.as_of, "as_of"))

        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError("Feature values must be finite.")

        if self.status is FeatureStatus.OBSERVED:
            if self.value is None:
                raise ValueError("Observed features require a value.")
            if self.missing_reason is not None or self.imputation_method is not None:
                raise ValueError(
                    "Observed features cannot have missingness or imputation metadata."
                )
            return

        if self.status is FeatureStatus.MISSING:
            if self.value is not None:
                raise ValueError("Missing features must not contain a value.")
            if self.missing_reason is None:
                raise ValueError("Missing features require missing_reason.")
            if self.imputation_method is not None:
                raise ValueError("Missing features cannot have imputation_method.")
            return

        if self.status is FeatureStatus.IMPUTED:
            if self.value is None:
                raise ValueError("Imputed features require a value.")
            if self.missing_reason is None:
                raise ValueError("Imputed features require the original missing reason.")
            if not self.imputation_method or not self.imputation_method.strip():
                raise ValueError("Imputed features require imputation_method.")
            return

        raise ValueError(f"Unsupported feature status: {self.status}")


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Point-in-time context for building features for one canonical match."""

    match: Match
    prediction_time: datetime

    def __post_init__(self) -> None:
        prediction_time = ensure_utc(self.prediction_time, "prediction_time")
        object.__setattr__(self, "prediction_time", prediction_time)

        if self.match.kickoff_at is not None:
            if prediction_time > self.match.kickoff_at:
                raise TemporalIntegrityError(
                    "prediction_time cannot be after exact kickoff_at."
                )
            return

        if prediction_time.date() >= self.match.match_date:
            raise TemporalIntegrityError(
                "Date-only matches require a prediction cutoff before match_date."
            )


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Versioned feature vector produced for one point-in-time match context."""

    match_id: str
    prediction_time: datetime
    feature_set_id: str
    values: tuple[FeatureValue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prediction_time",
            ensure_utc(self.prediction_time, "prediction_time"),
        )

        names = [value.definition.name for value in self.values]
        if len(names) != len(set(names)):
            raise ValueError("Feature vector contains duplicate feature names.")


class FeatureProvider(Protocol):
    """Contract implemented by replaceable point-in-time feature providers."""

    provider_id: str

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        """Return every feature definition produced by the provider."""

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        """Compute feature values for one point-in-time context."""


def build_feature_vector(
    context: PredictionContext,
    providers: Sequence[FeatureProvider],
) -> FeatureVector:
    """Build and validate a deterministic feature vector from providers."""

    definitions: list[FeatureDefinition] = []
    values: list[FeatureValue] = []
    seen_names: set[str] = set()

    for provider in providers:
        declared = provider.definitions()
        declared_by_name = {definition.name: definition for definition in declared}

        if len(declared_by_name) != len(declared):
            raise ValueError(
                f"Provider {provider.provider_id} declares duplicate feature names."
            )

        overlap = seen_names.intersection(declared_by_name)
        if overlap:
            raise ValueError(
                "Feature names must be globally unique; duplicates: "
                f"{sorted(overlap)}"
            )

        computed = provider.compute(context)
        computed_by_name = {
            feature_value.definition.name: feature_value
            for feature_value in computed
        }

        if set(computed_by_name) != set(declared_by_name):
            raise ValueError(
                f"Provider {provider.provider_id} computed features that do not "
                "match its declared definitions."
            )

        for name, feature_value in computed_by_name.items():
            if feature_value.definition != declared_by_name[name]:
                raise ValueError(
                    f"Provider {provider.provider_id} changed definition for {name}."
                )

            if feature_value.as_of != context.prediction_time:
                raise TemporalIntegrityError(
                    f"Feature {name} does not use the prediction-time cutoff."
                )

        definitions.extend(declared)
        values.extend(computed)
        seen_names.update(declared_by_name)

    ordered_values = tuple(
        sorted(values, key=lambda feature_value: feature_value.definition.name)
    )

    return FeatureVector(
        match_id=context.match.match_id,
        prediction_time=context.prediction_time,
        feature_set_id=_feature_set_id(definitions),
        values=ordered_values,
    )


def _feature_set_id(definitions: Sequence[FeatureDefinition]) -> str:
    tokens = sorted(
        f"{definition.name}@{definition.version}"
        for definition in definitions
    )
    digest = hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()[:20]
    return f"features_{digest}"
