from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from football_analytics.domain import MatchOutcome
from football_analytics.features.dataset import (
    HistoricalFeatureDataset,
    HistoricalFeatureExample,
)
from football_analytics.features.materialization import (
    ImputationPolicy,
    MaterializedFeatureRow,
    materialize_feature_vector,
)

MODEL_DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ModelExample:
    """One supervised, fully materialized point-in-time model example."""

    match_id: str
    target: MatchOutcome
    row: MaterializedFeatureRow

    def __post_init__(self) -> None:
        if self.row.match_id != self.match_id:
            raise ValueError("Model example match_id does not match materialized row.")


@dataclass(frozen=True, slots=True)
class ModelDataset:
    """Immutable content-addressed dataset consumed by model implementations."""

    dataset_id: str
    feature_set_id: str
    cutoff_policy_id: str
    imputation_policy_id: str
    column_names: tuple[str, ...]
    examples: tuple[ModelExample, ...]
    schema_version: int = MODEL_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be blank.")
        if not self.feature_set_id.strip():
            raise ValueError("feature_set_id must not be blank.")
        if not self.cutoff_policy_id.strip():
            raise ValueError("cutoff_policy_id must not be blank.")
        if not self.imputation_policy_id.strip():
            raise ValueError("imputation_policy_id must not be blank.")
        if not self.column_names:
            raise ValueError("Model dataset must contain at least one feature column.")
        if not self.examples:
            raise ValueError("Model dataset must contain at least one example.")

        match_ids = [example.match_id for example in self.examples]
        if len(match_ids) != len(set(match_ids)):
            raise ValueError("Model dataset contains duplicate match IDs.")

        for example in self.examples:
            row = example.row
            if row.feature_set_id != self.feature_set_id:
                raise ValueError("Model dataset contains mixed feature-set IDs.")
            if row.imputation_policy_id != self.imputation_policy_id:
                raise ValueError("Model dataset contains mixed imputation-policy IDs.")
            if tuple(name for name, _ in row.columns) != self.column_names:
                raise ValueError("Model dataset contains inconsistent feature columns.")

    @property
    def targets(self) -> tuple[MatchOutcome, ...]:
        return tuple(example.target for example in self.examples)

    def feature_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return model values in the dataset's canonical column order."""

        return tuple(
            tuple(value for _, value in example.row.columns)
            for example in self.examples
        )


def build_model_dataset(
    dataset: HistoricalFeatureDataset,
    *,
    imputation_policy: ImputationPolicy,
) -> ModelDataset:
    """Materialize and content-address one historical feature dataset."""

    return build_model_dataset_from_examples(
        feature_set_id=dataset.feature_set_id,
        cutoff_policy_id=dataset.cutoff_policy_id,
        examples=dataset.examples,
        imputation_policy=imputation_policy,
    )


def build_model_dataset_from_examples(
    *,
    feature_set_id: str,
    cutoff_policy_id: str,
    examples: Sequence[HistoricalFeatureExample],
    imputation_policy: ImputationPolicy,
) -> ModelDataset:
    """Materialize a declared subset of historical examples.

    This function is used by temporal backtests so train and evaluation datasets
    receive their own immutable content identities.
    """

    if not examples:
        raise ValueError("At least one historical example is required.")

    ordered_examples = tuple(
        sorted(
            examples,
            key=lambda example: (
                example.prediction_time,
                example.match_id,
            ),
        )
    )

    materialized = tuple(
        ModelExample(
            match_id=example.match_id,
            target=example.target,
            row=materialize_feature_vector(
                example.vector,
                policy=imputation_policy,
            ),
        )
        for example in ordered_examples
    )

    first_columns = tuple(name for name, _ in materialized[0].row.columns)
    for example in materialized[1:]:
        columns = tuple(name for name, _ in example.row.columns)
        if columns != first_columns:
            raise ValueError(
                "Historical examples materialized to inconsistent feature columns."
            )

    dataset_id = _dataset_id(
        feature_set_id=feature_set_id,
        cutoff_policy_id=cutoff_policy_id,
        imputation_policy=imputation_policy,
        column_names=first_columns,
        examples=materialized,
    )

    return ModelDataset(
        dataset_id=dataset_id,
        feature_set_id=feature_set_id,
        cutoff_policy_id=cutoff_policy_id,
        imputation_policy_id=imputation_policy.policy_id,
        column_names=first_columns,
        examples=materialized,
    )


def _dataset_id(
    *,
    feature_set_id: str,
    cutoff_policy_id: str,
    imputation_policy: ImputationPolicy,
    column_names: tuple[str, ...],
    examples: tuple[ModelExample, ...],
) -> str:
    payload = {
        "schema_version": MODEL_DATASET_SCHEMA_VERSION,
        "feature_set_id": feature_set_id,
        "cutoff_policy_id": cutoff_policy_id,
        "imputation_policy": {
            "policy_id": imputation_policy.policy_id,
            "include_status_indicators": imputation_policy.include_status_indicators,
            "rules": [
                {
                    "feature_name": rule.feature_name,
                    "value": rule.value,
                    "method": rule.method,
                }
                for rule in sorted(
                    imputation_policy.rules,
                    key=lambda rule: rule.feature_name,
                )
            ],
        },
        "column_names": list(column_names),
        "examples": [
            {
                "match_id": example.match_id,
                "prediction_time": example.row.prediction_time_iso,
                "target": example.target.value,
                "values": [value for _, value in example.row.columns],
                "applied_imputations": [
                    {
                        "feature_name": applied.feature_name,
                        "value": applied.value,
                        "method": applied.method,
                    }
                    for applied in example.row.applied_imputations
                ],
            }
            for example in examples
        ],
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"model_dataset_{digest}"
