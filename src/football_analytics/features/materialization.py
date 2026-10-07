from __future__ import annotations

import math
from dataclasses import dataclass

from football_analytics.features.base import (
    FeatureStatus,
    FeatureVector,
)


class MissingFeaturePolicyError(ValueError):
    """Raised when a missing feature has no explicit model-input policy."""


@dataclass(frozen=True, slots=True)
class ConstantImputationRule:
    """Explicit constant imputation rule for one named feature."""

    feature_name: str
    value: float
    method: str

    def __post_init__(self) -> None:
        feature_name = self.feature_name.strip()
        method = self.method.strip()

        if not feature_name:
            raise ValueError("feature_name must not be blank.")
        if not method:
            raise ValueError("method must not be blank.")
        if not math.isfinite(float(self.value)):
            raise ValueError("Imputation value must be finite.")

        object.__setattr__(self, "feature_name", feature_name)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True, slots=True)
class ImputationPolicy:
    """Versioned rules used to make feature vectors model-ready."""

    policy_id: str
    rules: tuple[ConstantImputationRule, ...]
    include_status_indicators: bool = True

    def __post_init__(self) -> None:
        policy_id = self.policy_id.strip()
        if not policy_id:
            raise ValueError("policy_id must not be blank.")
        object.__setattr__(self, "policy_id", policy_id)

        names = [rule.feature_name for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("Imputation policy contains duplicate feature rules.")

    def rule_for(self, feature_name: str) -> ConstantImputationRule | None:
        for rule in self.rules:
            if rule.feature_name == feature_name:
                return rule
        return None


@dataclass(frozen=True, slots=True)
class AppliedImputation:
    """Audit record for a missing feature materialized with a policy rule."""

    feature_name: str
    value: float
    method: str


@dataclass(frozen=True, slots=True)
class MaterializedFeatureRow:
    """Immutable numeric feature row ready for a model interface."""

    match_id: str
    prediction_time_iso: str
    feature_set_id: str
    imputation_policy_id: str
    columns: tuple[tuple[str, float], ...]
    applied_imputations: tuple[AppliedImputation, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(self.columns)


def materialize_feature_vector(
    vector: FeatureVector,
    *,
    policy: ImputationPolicy,
) -> MaterializedFeatureRow:
    """Convert an auditable feature vector into a complete numeric model row."""

    columns: list[tuple[str, float]] = []
    applied: list[AppliedImputation] = []

    for feature in sorted(
        vector.values,
        key=lambda value: value.definition.name,
    ):
        name = feature.definition.name
        original_missing = feature.status is FeatureStatus.MISSING
        original_imputed = feature.status is FeatureStatus.IMPUTED

        if feature.value is None:
            rule = policy.rule_for(name)
            if rule is None:
                raise MissingFeaturePolicyError(
                    f"Missing feature {name!r} has no explicit imputation rule."
                )
            numeric_value = rule.value
            applied.append(
                AppliedImputation(
                    feature_name=name,
                    value=rule.value,
                    method=rule.method,
                )
            )
            materialized_imputed = True
        else:
            numeric_value = float(feature.value)
            materialized_imputed = original_imputed

        columns.append((name, numeric_value))

        if policy.include_status_indicators:
            columns.extend(
                [
                    (f"{name}__is_missing", float(original_missing)),
                    (f"{name}__is_imputed", float(materialized_imputed)),
                ]
            )

    names = [name for name, _ in columns]
    if len(names) != len(set(names)):
        raise ValueError("Materialized feature columns are not unique.")

    return MaterializedFeatureRow(
        match_id=vector.match_id,
        prediction_time_iso=vector.prediction_time.isoformat(),
        feature_set_id=vector.feature_set_id,
        imputation_policy_id=policy.policy_id,
        columns=tuple(columns),
        applied_imputations=tuple(applied),
    )
