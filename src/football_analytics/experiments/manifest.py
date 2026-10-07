from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from football_analytics.domain.scores import REGULATION_TARGET_POLICY_ID
from football_analytics.evaluation.backtest import TemporalBacktestResult
from football_analytics.evaluation.calibration import (
    CalibrationReport,
    build_calibration_report,
)
from football_analytics.evaluation.metrics import (
    EvaluationMetrics,
    ScoredPrediction,
)

EXPERIMENT_MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Immutable, self-describing record of one evaluated model experiment."""

    experiment_id: str
    backtest_run_id: str
    calibration_report_id: str | None
    model_spec_id: str
    model_family: str
    model_version: str
    model_random_seed: int
    model_parameters: tuple[tuple[str, bool | int | float | str], ...]
    split_policy_id: str
    feature_set_id: str
    cutoff_policy_id: str
    result_eligibility_policy_id: str
    imputation_policy_id: str
    prediction_count: int
    aggregate_metrics: EvaluationMetrics
    fold_ids: tuple[str, ...]
    training_run_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    train_dataset_ids: tuple[str, ...]
    evaluation_dataset_ids: tuple[str, ...]
    calibration_n_bins: int | None = None
    macro_expected_calibration_error: float | None = None
    code_revision: str | None = None
    target_policy_id: str = REGULATION_TARGET_POLICY_ID
    schema_version: int = EXPERIMENT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.target_policy_id != REGULATION_TARGET_POLICY_ID:
            raise ValueError("Unsupported score target policy.")
        if self.schema_version != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unsupported experiment manifest schema version.")

        required_ids = {
            "backtest_run_id": self.backtest_run_id,
            "model_spec_id": self.model_spec_id,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "split_policy_id": self.split_policy_id,
            "feature_set_id": self.feature_set_id,
            "cutoff_policy_id": self.cutoff_policy_id,
            "result_eligibility_policy_id": self.result_eligibility_policy_id,
            "imputation_policy_id": self.imputation_policy_id,
        }
        for field_name, value in required_ids.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank.")

        if self.model_random_seed < 0:
            raise ValueError("model_random_seed must be non-negative.")

        parameter_names = [name.strip() for name, _ in self.model_parameters]
        if any(not name for name in parameter_names):
            raise ValueError("Model parameter names must not be blank.")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Model parameter names must be unique.")

        canonical_parameters = tuple(
            sorted(
                (
                    (name.strip(), value)
                    for name, value in self.model_parameters
                ),
                key=lambda item: item[0],
            )
        )
        object.__setattr__(self, "model_parameters", canonical_parameters)

        if self.prediction_count <= 0:
            raise ValueError("prediction_count must be positive.")
        if self.aggregate_metrics.n_predictions != self.prediction_count:
            raise ValueError(
                "Aggregate metric count must match experiment prediction_count."
            )

        fold_count = len(self.fold_ids)
        if fold_count == 0:
            raise ValueError("Experiment manifest requires at least one fold.")
        if len(set(self.fold_ids)) != fold_count:
            raise ValueError("Experiment fold_ids must be unique.")

        linked_fold_sequences = (
            self.training_run_ids,
            self.model_ids,
            self.train_dataset_ids,
            self.evaluation_dataset_ids,
        )
        if any(len(values) != fold_count for values in linked_fold_sequences):
            raise ValueError(
                "Fold-linked experiment identifiers must have equal lengths."
            )
        if any(
            not value.strip()
            for values in (self.fold_ids, *linked_fold_sequences)
            for value in values
        ):
            raise ValueError(
                "Fold-linked experiment identifiers must not be blank."
            )

        has_calibration = self.calibration_report_id is not None
        if (
            self.calibration_report_id is not None
            and not self.calibration_report_id.strip()
        ):
            raise ValueError("calibration_report_id must not be blank.")

        calibration_fields = (
            self.calibration_n_bins,
            self.macro_expected_calibration_error,
        )
        if has_calibration != all(value is not None for value in calibration_fields):
            raise ValueError(
                "Calibration metadata must be entirely present or entirely absent."
            )
        if self.calibration_n_bins is not None and self.calibration_n_bins <= 0:
            raise ValueError("calibration_n_bins must be positive.")
        if (
            self.macro_expected_calibration_error is not None
            and self.macro_expected_calibration_error < 0.0
        ):
            raise ValueError(
                "macro_expected_calibration_error must be non-negative."
            )

        if self.code_revision is not None:
            code_revision = self.code_revision.strip()
            if not code_revision:
                raise ValueError("code_revision must not be blank when provided.")
            object.__setattr__(self, "code_revision", code_revision)

        expected_id = _experiment_id(self._identity_payload())
        if self.experiment_id != expected_id:
            raise ValueError(
                "experiment_id does not match the manifest's deterministic identity."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe canonical manifest representation."""

        return {
            "experiment_id": self.experiment_id,
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentManifest:
        """Load and validate a manifest from a JSON-compatible dictionary."""

        if payload.get("schema_version") != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unsupported experiment manifest schema version.")
        if payload.get("target_policy_id") != REGULATION_TARGET_POLICY_ID:
            raise ValueError("Unsupported score target policy.")
        metrics_payload = payload["aggregate_metrics"]
        if not isinstance(metrics_payload, dict):
            raise ValueError("aggregate_metrics must be a JSON object.")

        metrics = EvaluationMetrics(
            n_predictions=int(metrics_payload["n_predictions"]),
            accuracy=float(metrics_payload["accuracy"]),
            log_loss=float(metrics_payload["log_loss"]),
            multiclass_brier_score=float(
                metrics_payload["multiclass_brier_score"]
            ),
            ranked_probability_score=float(
                metrics_payload["ranked_probability_score"]
            ),
        )

        return cls(
            schema_version=int(payload["schema_version"]),
            target_policy_id=str(payload["target_policy_id"]),
            experiment_id=str(payload["experiment_id"]),
            backtest_run_id=str(payload["backtest_run_id"]),
            calibration_report_id=_optional_str(
                payload.get("calibration_report_id")
            ),
            model_spec_id=str(payload["model_spec_id"]),
            model_family=str(payload["model_family"]),
            model_version=str(payload["model_version"]),
            model_random_seed=int(payload["model_random_seed"]),
            model_parameters=tuple(
                (str(name), value)
                for name, value in payload["model_parameters"]
            ),
            split_policy_id=str(payload["split_policy_id"]),
            feature_set_id=str(payload["feature_set_id"]),
            cutoff_policy_id=str(payload["cutoff_policy_id"]),
            result_eligibility_policy_id=str(
                payload["result_eligibility_policy_id"]
            ),
            imputation_policy_id=str(payload["imputation_policy_id"]),
            prediction_count=int(payload["prediction_count"]),
            aggregate_metrics=metrics,
            fold_ids=tuple(str(value) for value in payload["fold_ids"]),
            training_run_ids=tuple(
                str(value) for value in payload["training_run_ids"]
            ),
            model_ids=tuple(str(value) for value in payload["model_ids"]),
            train_dataset_ids=tuple(
                str(value) for value in payload["train_dataset_ids"]
            ),
            evaluation_dataset_ids=tuple(
                str(value) for value in payload["evaluation_dataset_ids"]
            ),
            calibration_n_bins=(
                int(payload["calibration_n_bins"])
                if payload.get("calibration_n_bins") is not None
                else None
            ),
            macro_expected_calibration_error=(
                float(payload["macro_expected_calibration_error"])
                if payload.get("macro_expected_calibration_error") is not None
                else None
            ),
            code_revision=_optional_str(payload.get("code_revision")),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return _identity_payload(
            schema_version=self.schema_version,
            backtest_run_id=self.backtest_run_id,
            calibration_report_id=self.calibration_report_id,
            model_spec_id=self.model_spec_id,
            model_family=self.model_family,
            model_version=self.model_version,
            model_random_seed=self.model_random_seed,
            model_parameters=self.model_parameters,
            split_policy_id=self.split_policy_id,
            feature_set_id=self.feature_set_id,
            cutoff_policy_id=self.cutoff_policy_id,
            result_eligibility_policy_id=self.result_eligibility_policy_id,
            imputation_policy_id=self.imputation_policy_id,
            prediction_count=self.prediction_count,
            aggregate_metrics=self.aggregate_metrics,
            fold_ids=self.fold_ids,
            training_run_ids=self.training_run_ids,
            model_ids=self.model_ids,
            train_dataset_ids=self.train_dataset_ids,
            evaluation_dataset_ids=self.evaluation_dataset_ids,
            calibration_n_bins=self.calibration_n_bins,
            macro_expected_calibration_error=self.macro_expected_calibration_error,
            code_revision=self.code_revision,
        )


def build_experiment_manifest(
    backtest: TemporalBacktestResult,
    *,
    calibration: CalibrationReport | None = None,
    code_revision: str | None = None,
) -> ExperimentManifest:
    """Build one deterministic experiment manifest from evaluated artifacts."""

    if calibration is not None:
        if calibration.n_predictions != backtest.prediction_count:
            raise ValueError(
                "Calibration prediction count must match the backtest prediction count."
            )

        scored = [
            ScoredPrediction(
                actual=prediction.actual,
                probabilities=prediction.probabilities,
            )
            for fold in backtest.folds
            for prediction in fold.predictions
        ]
        expected_calibration = build_calibration_report(
            scored,
            n_bins=calibration.n_bins,
        )
        if calibration.report_id != expected_calibration.report_id:
            raise ValueError(
                "Calibration report does not match the backtest prediction sample."
            )

    calibration_report_id = (
        calibration.report_id if calibration is not None else None
    )
    calibration_n_bins = calibration.n_bins if calibration is not None else None
    macro_ece = (
        calibration.macro_expected_calibration_error
        if calibration is not None
        else None
    )
    normalized_revision = code_revision.strip() if code_revision is not None else None
    fold_ids = tuple(fold.fold_id for fold in backtest.folds)
    training_run_ids = tuple(fold.training_run_id for fold in backtest.folds)
    model_ids = tuple(fold.model_id for fold in backtest.folds)
    train_dataset_ids = tuple(fold.train_dataset_id for fold in backtest.folds)
    evaluation_dataset_ids = tuple(
        fold.evaluation_dataset_id for fold in backtest.folds
    )

    identity = _identity_payload(
        schema_version=EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        backtest_run_id=backtest.backtest_run_id,
        calibration_report_id=calibration_report_id,
        model_spec_id=backtest.model_spec_id,
        model_family=backtest.model_family,
        model_version=backtest.model_version,
        model_random_seed=backtest.model_random_seed,
        model_parameters=backtest.model_parameters,
        split_policy_id=backtest.split_policy_id,
        feature_set_id=backtest.feature_set_id,
        cutoff_policy_id=backtest.cutoff_policy_id,
        result_eligibility_policy_id=backtest.result_eligibility_policy_id,
        imputation_policy_id=backtest.imputation_policy_id,
        prediction_count=backtest.prediction_count,
        aggregate_metrics=backtest.aggregate_metrics,
        fold_ids=fold_ids,
        training_run_ids=training_run_ids,
        model_ids=model_ids,
        train_dataset_ids=train_dataset_ids,
        evaluation_dataset_ids=evaluation_dataset_ids,
        calibration_n_bins=calibration_n_bins,
        macro_expected_calibration_error=macro_ece,
        code_revision=normalized_revision,
    )

    if backtest.target_policy_id != REGULATION_TARGET_POLICY_ID:
        raise ValueError("Unsupported score target policy.")
    return ExperimentManifest(
        experiment_id=_experiment_id(identity),
        backtest_run_id=backtest.backtest_run_id,
        calibration_report_id=calibration_report_id,
        model_spec_id=backtest.model_spec_id,
        model_family=backtest.model_family,
        model_version=backtest.model_version,
        model_random_seed=backtest.model_random_seed,
        model_parameters=backtest.model_parameters,
        split_policy_id=backtest.split_policy_id,
        feature_set_id=backtest.feature_set_id,
        cutoff_policy_id=backtest.cutoff_policy_id,
        result_eligibility_policy_id=backtest.result_eligibility_policy_id,
        imputation_policy_id=backtest.imputation_policy_id,
        prediction_count=backtest.prediction_count,
        aggregate_metrics=backtest.aggregate_metrics,
        fold_ids=fold_ids,
        training_run_ids=training_run_ids,
        model_ids=model_ids,
        train_dataset_ids=train_dataset_ids,
        evaluation_dataset_ids=evaluation_dataset_ids,
        calibration_n_bins=calibration_n_bins,
        macro_expected_calibration_error=macro_ece,
        code_revision=normalized_revision,
    )


def _identity_payload(
    *,
    schema_version: int,
    backtest_run_id: str,
    calibration_report_id: str | None,
    model_spec_id: str,
    model_family: str,
    model_version: str,
    model_random_seed: int,
    model_parameters: tuple[tuple[str, bool | int | float | str], ...],
    split_policy_id: str,
    feature_set_id: str,
    cutoff_policy_id: str,
    result_eligibility_policy_id: str,
    imputation_policy_id: str,
    prediction_count: int,
    aggregate_metrics: EvaluationMetrics,
    fold_ids: tuple[str, ...],
    training_run_ids: tuple[str, ...],
    model_ids: tuple[str, ...],
    train_dataset_ids: tuple[str, ...],
    evaluation_dataset_ids: tuple[str, ...],
    calibration_n_bins: int | None,
    macro_expected_calibration_error: float | None,
    code_revision: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "target_policy_id": REGULATION_TARGET_POLICY_ID,
        "backtest_run_id": backtest_run_id,
        "calibration_report_id": calibration_report_id,
        "model_spec_id": model_spec_id,
        "model_family": model_family,
        "model_version": model_version,
        "model_random_seed": model_random_seed,
        "model_parameters": [list(item) for item in model_parameters],
        "split_policy_id": split_policy_id,
        "feature_set_id": feature_set_id,
        "cutoff_policy_id": cutoff_policy_id,
        "result_eligibility_policy_id": result_eligibility_policy_id,
        "imputation_policy_id": imputation_policy_id,
        "prediction_count": prediction_count,
        "aggregate_metrics": {
            "n_predictions": aggregate_metrics.n_predictions,
            "accuracy": aggregate_metrics.accuracy,
            "log_loss": aggregate_metrics.log_loss,
            "multiclass_brier_score": aggregate_metrics.multiclass_brier_score,
            "ranked_probability_score": aggregate_metrics.ranked_probability_score,
        },
        "fold_ids": list(fold_ids),
        "training_run_ids": list(training_run_ids),
        "model_ids": list(model_ids),
        "train_dataset_ids": list(train_dataset_ids),
        "evaluation_dataset_ids": list(evaluation_dataset_ids),
        "calibration_n_bins": calibration_n_bins,
        "macro_expected_calibration_error": macro_expected_calibration_error,
        "code_revision": code_revision,
    }


def _experiment_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"experiment_{digest}"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
