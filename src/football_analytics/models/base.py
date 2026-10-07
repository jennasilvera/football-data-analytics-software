from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.features import MaterializedFeatureRow
from football_analytics.models.dataset import ModelDataset

type ParameterValue = bool | int | float | str


class ModelFamily(StrEnum):
    """Supported native V2 probabilistic model families."""

    LOGISTIC_REGRESSION = "logistic_regression"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
    CLASS_FREQUENCY = "class_frequency"
    POISSON = "poisson"
    TEMPERATURE_SCALING = "temperature_scaling"
    CONVEX_ENSEMBLE = "convex_ensemble"


@dataclass(frozen=True, slots=True)
class ModelTrainingSpec:
    """Versioned, deterministic declaration of a model training configuration."""

    spec_id: str
    family: ModelFamily
    model_version: str
    random_seed: int = 42
    parameters: tuple[tuple[str, ParameterValue], ...] = ()

    def __post_init__(self) -> None:
        spec_id = self.spec_id.strip()
        model_version = self.model_version.strip()

        if not spec_id:
            raise ValueError("spec_id must not be blank.")
        if not model_version:
            raise ValueError("model_version must not be blank.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")

        parameter_names = [name.strip() for name, _ in self.parameters]
        if any(not name for name in parameter_names):
            raise ValueError("Model parameter names must not be blank.")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Model training spec contains duplicate parameter names.")

        for name, value in self.parameters:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Model parameter {name!r} must be finite.")

        canonical_parameters = tuple(
            sorted(
                ((name.strip(), value) for name, value in self.parameters),
                key=lambda item: item[0],
            )
        )

        object.__setattr__(self, "spec_id", spec_id)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "parameters", canonical_parameters)

    def parameter_dict(self) -> dict[str, ParameterValue]:
        return dict(self.parameters)


@dataclass(frozen=True, slots=True)
class ModelTrainingMetadata:
    """Auditable metadata for one deterministic fitted-model run."""

    training_run_id: str
    model_id: str
    spec_id: str
    family: ModelFamily
    model_version: str
    dataset_id: str
    feature_set_id: str
    cutoff_policy_id: str
    result_eligibility_policy_id: str
    imputation_policy_id: str
    n_examples: int
    feature_names: tuple[str, ...]
    random_seed: int
    parameters: tuple[tuple[str, ParameterValue], ...]


class ProbabilisticModel(Protocol):
    """Minimal inference contract shared by native V2 probabilistic models."""

    @property
    def model_id(self) -> str: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def feature_set_id(self) -> str: ...

    @property
    def imputation_policy_id(self) -> str: ...

    def predict_row(self, row: MaterializedFeatureRow) -> OutcomeProbabilities:
        """Predict one three-way outcome probability distribution."""


@dataclass(frozen=True, slots=True)
class ModelTrainingResult:
    """Fitted model paired with complete training identity metadata."""

    model: ProbabilisticModel
    metadata: ModelTrainingMetadata

    def __post_init__(self) -> None:
        if self.model.model_id != self.metadata.model_id:
            raise ValueError("Fitted model ID does not match training metadata.")


def training_metadata_for(
    dataset: ModelDataset,
    spec: ModelTrainingSpec,
) -> ModelTrainingMetadata:
    """Build deterministic identities for one model/dataset training declaration."""

    payload = {
        "spec_id": spec.spec_id,
        "family": spec.family.value,
        "model_version": spec.model_version,
        "random_seed": spec.random_seed,
        "parameters": list(spec.parameters),
        "dataset_id": dataset.dataset_id,
        "feature_names": list(dataset.column_names),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return ModelTrainingMetadata(
        training_run_id=f"training_run_{digest}",
        model_id=f"model_{digest}",
        spec_id=spec.spec_id,
        family=spec.family,
        model_version=spec.model_version,
        dataset_id=dataset.dataset_id,
        feature_set_id=dataset.feature_set_id,
        cutoff_policy_id=dataset.cutoff_policy_id,
        result_eligibility_policy_id=dataset.result_eligibility_policy_id,
        imputation_policy_id=dataset.imputation_policy_id,
        n_examples=len(dataset.examples),
        feature_names=dataset.column_names,
        random_seed=spec.random_seed,
        parameters=spec.parameters,
    )
