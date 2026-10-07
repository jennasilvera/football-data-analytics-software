"""Training-only class-frequency benchmark, including sparse-class folds."""

from __future__ import annotations

import math
from dataclasses import dataclass

from football_analytics.domain import MatchOutcome
from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.features.materialization import MaterializedFeatureRow
from football_analytics.models.base import (
    ModelFamily,
    ModelTrainingResult,
    ModelTrainingSpec,
    training_metadata_for,
)
from football_analytics.models.dataset import ModelDataset


def class_frequency_spec(*, smoothing: float = 1.0) -> ModelTrainingSpec:
    if not math.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be finite and non-negative.")
    return ModelTrainingSpec(
        "class_frequency_v1",
        ModelFamily.CLASS_FREQUENCY,
        "v1",
        parameters=(("smoothing", float(smoothing)),),
    )


@dataclass(frozen=True, slots=True)
class ClassFrequencyModel:
    model_id: str
    feature_names: tuple[str, ...]
    feature_set_id: str
    imputation_policy_id: str
    probabilities: OutcomeProbabilities

    def predict_row(self, row: MaterializedFeatureRow) -> OutcomeProbabilities:
        if (
            row.feature_set_id != self.feature_set_id
            or row.imputation_policy_id != self.imputation_policy_id
            or tuple(name for name, _ in row.columns) != self.feature_names
        ):
            raise ValueError("Class-frequency model input contract mismatch.")
        return self.probabilities


def train_class_frequency(dataset: ModelDataset, spec: ModelTrainingSpec) -> ModelTrainingResult:
    parameters = spec.parameter_dict()
    if spec.family is not ModelFamily.CLASS_FREQUENCY or set(parameters) != {"smoothing"}:
        raise ValueError("Expected a class-frequency specification with smoothing.")
    smoothing = float(parameters["smoothing"])
    class_frequency_spec(smoothing=smoothing)
    denominator = len(dataset.examples) + 3 * smoothing
    probabilities = OutcomeProbabilities(
        *(
            (dataset.targets.count(outcome) + smoothing) / denominator
            for outcome in (MatchOutcome.HOME_WIN, MatchOutcome.DRAW, MatchOutcome.AWAY_WIN)
        )
    )
    metadata = training_metadata_for(dataset, spec)
    return ModelTrainingResult(
        ClassFrequencyModel(
            metadata.model_id,
            dataset.column_names,
            dataset.feature_set_id,
            dataset.imputation_policy_id,
            probabilities,
        ),
        metadata,
    )
