"""Chronological inner holdout fitting with untouched outer evaluation labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta

from football_analytics.data import CanonicalCatalogs, MatchObservation
from football_analytics.evaluation import (
    ExpandingWindowPolicy,
    RollingWindowPolicy,
    ScoredPrediction,
    TemporalBacktestResult,
    build_calibration_report,
)
from football_analytics.evaluation.backtest import BacktestFoldResult, _backtest_run_id
from football_analytics.evaluation.comparison import ModelComparison, compare_backtests
from football_analytics.evaluation.metrics import evaluate_predictions
from football_analytics.experiments import build_experiment_manifest
from football_analytics.features.history import DEFAULT_RESULT_ELIGIBILITY_POLICY
from football_analytics.models import ModelFamily, ModelTrainingSpec
from football_analytics.models.postprocessing import (
    BaseFitEvidence,
    HeldOutPrediction,
    ProbabilityTransform,
    content_id,
    fit_probability_transform,
)
from football_analytics.services.research import ResearchResult, run_research_comparison


@dataclass(frozen=True, slots=True)
class NestedHoldoutPolicy:
    holdout_window: timedelta
    min_examples: int = 30
    regularization: float = 0.01

    def __post_init__(self) -> None:
        import math

        if self.holdout_window <= timedelta(0):
            raise ValueError("Inner holdout window must be positive.")
        if type(self.min_examples) is not int or self.min_examples < 1:
            raise ValueError("Inner minimum must be a positive integer.")
        if not math.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("Ensemble regularization must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class NestedFoldAudit:
    outer_cutoff: datetime
    inner_cutoff: datetime
    inner_backtests: tuple[TemporalBacktestResult, ...]
    unavailable_match_ids: tuple[str, ...]
    transforms: tuple[ProbabilityTransform, ...]
    outer_base_model_ids: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "outer_cutoff": self.outer_cutoff.isoformat(),
            "inner_cutoff": self.inner_cutoff.isoformat(),
            "inner_backtests": [asdict(run) for run in self.inner_backtests],
            "unavailable_match_ids": self.unavailable_match_ids,
            "transforms": [model.to_dict() for model in self.transforms],
            "outer_base_model_ids": self.outer_base_model_ids,
        }


@dataclass(frozen=True, slots=True)
class NestedResearchResult:
    runs: tuple[ResearchResult, ...]
    comparison: ModelComparison
    audits: tuple[NestedFoldAudit, ...]


def run_nested_research(
    *,
    observations: list[MatchObservation],
    catalogs: CanonicalCatalogs,
    split_policy: ExpandingWindowPolicy | RollingWindowPolicy,
    model_specs: tuple[ModelTrainingSpec, ...],
    nested_policy: NestedHoldoutPolicy,
    calibration_bins: int = 10,
    code_revision: str | None = None,
) -> NestedResearchResult:
    if len({spec.spec_id for spec in model_specs}) != len(model_specs):
        raise ValueError("Nested base model spec IDs must be unique.")
    if isinstance(split_policy, RollingWindowPolicy):
        if nested_policy.holdout_window >= split_policy.training_window:
            raise ValueError("Inner holdout must be shorter than the rolling training window.")
    baseline = run_research_comparison(
        observations=observations,
        catalogs=catalogs,
        split_policy=split_policy,
        model_specs=model_specs,
        calibration_bins=calibration_bins,
        code_revision=code_revision,
    )
    if any(run.backtest.skipped_folds for run in baseline.runs):
        raise ValueError("Nested research requires every declared outer fold to be valid.")
    records = baseline.runs[0].normalization.normalized
    available = {
        record.match.match_id: DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(record).eligible_at
        for record in records
        if record.home_score is not None
    }
    specs = tuple(
        ModelTrainingSpec(
            f"temperature_{base.spec_id}",
            ModelFamily.TEMPERATURE_SCALING,
            "nested_holdout_v1",
            parameters=(
                ("base_spec_id", base.spec_id),
                ("holdout_seconds", nested_policy.holdout_window.total_seconds()),
                ("min_examples", nested_policy.min_examples),
            ),
        )
        for base in model_specs
    ) + (
        ModelTrainingSpec(
            "convex_probability_ensemble_v1",
            ModelFamily.CONVEX_ENSEMBLE,
            "nested_holdout_v1",
            parameters=(
                ("holdout_seconds", nested_policy.holdout_window.total_seconds()),
                ("min_examples", nested_policy.min_examples),
                ("regularization", nested_policy.regularization),
            ),
        ),
    )
    collected: list[list[BacktestFoldResult]] = [[] for _ in specs]
    audits = []
    for outer_index, cutoff in enumerate(split_policy.cutoffs):
        inner_cutoff = cutoff - nested_policy.holdout_window
        inner_policy: ExpandingWindowPolicy | RollingWindowPolicy
        if isinstance(split_policy, RollingWindowPolicy):
            inner_policy = RollingWindowPolicy(
                "nested_inner_rolling_v1",
                (inner_cutoff,),
                nested_policy.holdout_window,
                split_policy.training_window - nested_policy.holdout_window,
                split_policy.min_train_examples,
            )
        else:
            inner_policy = ExpandingWindowPolicy(
                "nested_inner_expanding_v1",
                (inner_cutoff,),
                nested_policy.holdout_window,
                split_policy.min_train_examples,
            )
        inner = run_research_comparison(
            observations=observations,
            catalogs=catalogs,
            split_policy=inner_policy,
            model_specs=model_specs,
            calibration_bins=calibration_bins,
            code_revision=code_revision,
        )
        inner_folds = [run.backtest.folds[0] for run in inner.runs]
        outer_folds = [run.backtest.folds[outer_index] for run in baseline.runs]
        members = tuple(
            BaseFitEvidence(
                spec.spec_id,
                fold.model_id,
                inner_cutoff,
                fold.train_match_ids,
            )
            for spec, fold in zip(model_specs, inner_folds, strict=True)
        )
        by_member = [{p.match_id: p for p in fold.predictions} for fold in inner_folds]
        examples = tuple(
            HeldOutPrediction(
                p.match_id,
                datetime.fromisoformat(p.prediction_time_iso),
                available[p.match_id],
                p.actual,
                tuple(member[p.match_id].probabilities for member in by_member),
            )
            for p in inner_folds[0].predictions
            if available[p.match_id] <= cutoff
        )
        # Every label used anywhere inside a fold must belong to its outer training population.
        allowed = set(outer_folds[0].train_match_ids)
        used = {e.match_id for e in examples}.union(*(set(m.train_match_ids) for m in members))
        if not used <= allowed:
            raise ValueError("Inner fitting sample escapes the outer training population.")
        transforms = tuple(
            fit_probability_transform(
                method="temperature",
                members=(member,),
                examples=tuple(replace(e, probabilities=(e.probabilities[i],)) for e in examples),
                fitted_at=cutoff,
                min_examples=nested_policy.min_examples,
            )
            for i, member in enumerate(members)
        ) + (
            fit_probability_transform(
                method="convex_ensemble",
                members=members,
                examples=examples,
                fitted_at=cutoff,
                min_examples=nested_policy.min_examples,
                regularization=nested_policy.regularization,
            ),
        )
        outer_predictions = [{p.match_id: p for p in fold.predictions} for fold in outer_folds]
        for i, transform in enumerate(transforms):
            source_indices = (i,) if i < len(model_specs) else tuple(range(len(model_specs)))
            binding = {
                "transform_id": transform.model_id,
                "base_model_ids": [outer_folds[j].model_id for j in source_indices],
            }
            model_id = content_id("nested_model_", binding)
            predictions = tuple(
                replace(
                    p,
                    model_id=model_id,
                    probabilities=transform.predict(
                        {
                            model_specs[j].spec_id: outer_predictions[j][p.match_id].probabilities
                            for j in source_indices
                        },
                        prediction_time=datetime.fromisoformat(p.prediction_time_iso),
                    ),
                )
                for p in outer_folds[0].predictions
            )
            collected[i].append(
                replace(
                    outer_folds[0],
                    model_id=model_id,
                    training_run_id="training_run_" + model_id,
                    train_dataset_id=content_id(
                        "nested_dataset_",
                        {
                            "base_datasets": [
                                outer_folds[j].train_dataset_id for j in source_indices
                            ],
                            "transform_id": transform.model_id,
                        },
                    ),
                    evaluation_dataset_id=content_id(
                        "nested_evaluation_",
                        {
                            "base_predictions": [
                                [asdict(p) for p in outer_folds[j].predictions]
                                for j in source_indices
                            ],
                        },
                    ),
                    predictions=predictions,
                    metrics=evaluate_predictions(
                        [ScoredPrediction(p.actual, p.probabilities) for p in predictions]
                    ),
                )
            )
        audits.append(
            NestedFoldAudit(
                cutoff,
                inner_cutoff,
                tuple(run.backtest for run in inner.runs),
                tuple(
                    p.match_id for p in inner_folds[0].predictions if available[p.match_id] > cutoff
                ),
                transforms,
                tuple(
                    (s.spec_id, f.model_id) for s, f in zip(model_specs, outer_folds, strict=True)
                ),
            )
        )
    derived = []
    for spec, folds in zip(specs, collected, strict=True):
        reference = baseline.runs[0]
        scored = [
            ScoredPrediction(p.actual, p.probabilities) for fold in folds for p in fold.predictions
        ]
        run = replace(
            reference.backtest,
            backtest_run_id=_backtest_run_id(
                split_policy_id=split_policy.policy_id,
                model_spec=spec,
                feature_set_id="base_probabilities_v1",
                cutoff_policy_id=reference.backtest.cutoff_policy_id,
                result_eligibility_policy_id=reference.backtest.result_eligibility_policy_id,
                imputation_policy_id="no_missing_base_probabilities_v1",
                folds=tuple(folds),
            ),
            model_spec_id=spec.spec_id,
            model_family=spec.family.value,
            model_version=spec.model_version,
            model_random_seed=spec.random_seed,
            model_parameters=spec.parameters,
            feature_set_id="base_probabilities_v1",
            imputation_policy_id="no_missing_base_probabilities_v1",
            folds=tuple(folds),
            aggregate_metrics=evaluate_predictions(scored),
        )
        diagnostics = build_calibration_report(scored, n_bins=calibration_bins)
        derived.append(
            ResearchResult(
                reference.normalization,
                run,
                diagnostics,
                build_experiment_manifest(
                    run, calibration=diagnostics, code_revision=code_revision
                ),
            )
        )
    runs = (*baseline.runs, *derived)
    return NestedResearchResult(
        runs,
        compare_backtests(
            [run.backtest for run in runs],
            reference_backtest_run_id=runs[0].backtest.backtest_run_id,
        ),
        tuple(audits),
    )
