"""Native V2 model datasets, training contracts, and implementations."""

from football_analytics.models.base import (
    ModelFamily,
    ModelTrainingMetadata,
    ModelTrainingResult,
    ModelTrainingSpec,
    ProbabilisticModel,
    training_metadata_for,
)
from football_analytics.models.dataset import (
    MODEL_DATASET_SCHEMA_VERSION,
    ModelDataset,
    ModelExample,
    build_model_dataset,
    build_model_dataset_from_examples,
)
from football_analytics.models.sklearn_models import (
    ModelInputMismatchError,
    SklearnOutcomeModel,
    hist_gradient_boosting_spec,
    logistic_regression_spec,
    train_sklearn_model,
)

__all__ = [
    "MODEL_DATASET_SCHEMA_VERSION",
    "ModelDataset",
    "ModelExample",
    "ModelFamily",
    "ModelInputMismatchError",
    "ModelTrainingMetadata",
    "ModelTrainingResult",
    "ModelTrainingSpec",
    "ProbabilisticModel",
    "SklearnOutcomeModel",
    "build_model_dataset",
    "build_model_dataset_from_examples",
    "hist_gradient_boosting_spec",
    "logistic_regression_spec",
    "train_sklearn_model",
    "training_metadata_for",
]
