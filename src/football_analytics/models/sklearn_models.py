from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_analytics.domain import MatchOutcome
from football_analytics.evaluation.probabilities import OutcomeProbabilities
from football_analytics.features import MaterializedFeatureRow
from football_analytics.models.base import (
    ModelFamily,
    ModelTrainingResult,
    ModelTrainingSpec,
    ParameterValue,
    training_metadata_for,
)
from football_analytics.models.dataset import ModelDataset


class ModelInputMismatchError(ValueError):
    """Raised when inference inputs do not match a fitted model contract."""


@dataclass(slots=True)
class SklearnOutcomeModel:
    """Fitted scikit-learn estimator exposed through the native V2 contract."""

    model_id: str
    feature_names: tuple[str, ...]
    feature_set_id: str
    imputation_policy_id: str
    estimator: Any

    def predict_row(self, row: MaterializedFeatureRow) -> OutcomeProbabilities:
        if row.feature_set_id != self.feature_set_id:
            raise ModelInputMismatchError(
                "Feature-set ID does not match the fitted model."
            )
        if row.imputation_policy_id != self.imputation_policy_id:
            raise ModelInputMismatchError(
                "Imputation-policy ID does not match the fitted model."
            )

        row_names = tuple(name for name, _ in row.columns)
        if row_names != self.feature_names:
            raise ModelInputMismatchError(
                "Feature columns or ordering do not match the fitted model."
            )

        values = [[value for _, value in row.columns]]
        raw_probabilities = self.estimator.predict_proba(values)[0]
        classes = tuple(str(value) for value in self.estimator.classes_)

        probability_by_class = {
            class_name: float(raw_probabilities[index])
            for index, class_name in enumerate(classes)
        }

        expected_classes = {outcome.value for outcome in MatchOutcome}
        if set(probability_by_class) != expected_classes:
            raise RuntimeError(
                "Fitted estimator does not expose the canonical three outcome classes."
            )

        return OutcomeProbabilities(
            home_win=probability_by_class[MatchOutcome.HOME_WIN.value],
            draw=probability_by_class[MatchOutcome.DRAW.value],
            away_win=probability_by_class[MatchOutcome.AWAY_WIN.value],
        )


def logistic_regression_spec(
    *,
    spec_id: str = "logistic_regression_v1",
    model_version: str = "v1",
    random_seed: int = 42,
    c: float = 1.0,
    max_iter: int = 1_000,
) -> ModelTrainingSpec:
    """Return the default V2 logistic-regression training declaration."""

    return ModelTrainingSpec(
        spec_id=spec_id,
        family=ModelFamily.LOGISTIC_REGRESSION,
        model_version=model_version,
        random_seed=random_seed,
        parameters=(
            ("C", c),
            ("max_iter", max_iter),
        ),
    )


def hist_gradient_boosting_spec(
    *,
    spec_id: str = "hist_gradient_boosting_v1",
    model_version: str = "v1",
    random_seed: int = 42,
    learning_rate: float = 0.05,
    max_iter: int = 250,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 0.05,
) -> ModelTrainingSpec:
    """Return the default V2 histogram-gradient-boosting declaration."""

    return ModelTrainingSpec(
        spec_id=spec_id,
        family=ModelFamily.HIST_GRADIENT_BOOSTING,
        model_version=model_version,
        random_seed=random_seed,
        parameters=(
            ("l2_regularization", l2_regularization),
            ("learning_rate", learning_rate),
            ("max_iter", max_iter),
            ("max_leaf_nodes", max_leaf_nodes),
        ),
    )


def train_sklearn_model(
    dataset: ModelDataset,
    spec: ModelTrainingSpec,
) -> ModelTrainingResult:
    """Fit a native V2 sklearn model against an immutable model dataset."""

    _validate_training_targets(dataset)
    estimator = _build_estimator(spec)

    x_train = [list(row) for row in dataset.feature_matrix()]
    y_train = [target.value for target in dataset.targets]

    estimator.fit(x_train, y_train)

    metadata = training_metadata_for(dataset, spec)
    model = SklearnOutcomeModel(
        model_id=metadata.model_id,
        feature_names=dataset.column_names,
        feature_set_id=dataset.feature_set_id,
        imputation_policy_id=dataset.imputation_policy_id,
        estimator=estimator,
    )

    return ModelTrainingResult(
        model=model,
        metadata=metadata,
    )


def _validate_training_targets(dataset: ModelDataset) -> None:
    observed = set(dataset.targets)
    required = set(MatchOutcome)
    missing = required - observed

    if missing:
        missing_values = sorted(outcome.value for outcome in missing)
        raise ValueError(
            "Training dataset is missing canonical outcome classes: "
            f"{missing_values}"
        )


def _build_estimator(spec: ModelTrainingSpec) -> Any:
    parameters = spec.parameter_dict()

    if spec.family is ModelFamily.LOGISTIC_REGRESSION:
        _reject_unknown_parameters(
            parameters,
            allowed={"C", "max_iter"},
            family=spec.family,
        )
        c = _positive_float(parameters, "C", 1.0)
        max_iter = _positive_int(parameters, "max_iter", 1_000)

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c,
                        max_iter=max_iter,
                        random_state=spec.random_seed,
                    ),
                ),
            ]
        )

    if spec.family is ModelFamily.HIST_GRADIENT_BOOSTING:
        _reject_unknown_parameters(
            parameters,
            allowed={
                "learning_rate",
                "max_iter",
                "max_leaf_nodes",
                "l2_regularization",
            },
            family=spec.family,
        )

        return HistGradientBoostingClassifier(
            learning_rate=_positive_float(parameters, "learning_rate", 0.05),
            max_iter=_positive_int(parameters, "max_iter", 250),
            max_leaf_nodes=_positive_int(parameters, "max_leaf_nodes", 31),
            l2_regularization=_nonnegative_float(
                parameters,
                "l2_regularization",
                0.05,
            ),
            random_state=spec.random_seed,
        )

    raise ValueError(f"Unsupported model family: {spec.family}")


def _reject_unknown_parameters(
    parameters: dict[str, ParameterValue],
    *,
    allowed: set[str],
    family: ModelFamily,
) -> None:
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(
            f"Unsupported {family.value} parameters: {sorted(unknown)}"
        )


def _positive_float(
    parameters: dict[str, ParameterValue],
    name: str,
    default: float,
) -> float:
    value = float(parameters.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _nonnegative_float(
    parameters: dict[str, ParameterValue],
    name: str,
    default: float,
) -> float:
    value = float(parameters.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _positive_int(
    parameters: dict[str, ParameterValue],
    name: str,
    default: int,
) -> int:
    raw = parameters.get(name, default)
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be an integer.")

    value = int(raw)
    if value <= 0 or float(raw) != float(value):
        raise ValueError(f"{name} must be a positive integer.")
    return value
