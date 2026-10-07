from __future__ import annotations

from dataclasses import dataclass

from football_analytics.data import (
    BatchNormalizationReport,
    CanonicalCatalogs,
    MatchObservation,
    normalize_match_batch,
)
from football_analytics.evaluation import (
    CalibrationReport,
    ExpandingWindowPolicy,
    RollingWindowPolicy,
    ScoredPrediction,
    TemporalBacktestResult,
    build_calibration_report,
    build_expanding_window_folds,
    build_rolling_window_folds,
    run_temporal_backtest,
)
from football_analytics.experiments import ExperimentManifest, build_experiment_manifest
from football_analytics.features import (
    ConstantImputationRule,
    ImputationPolicy,
    RollingFormFeatureProvider,
    build_historical_feature_dataset,
)
from football_analytics.models import ModelTrainingSpec, train_sklearn_model


class ResearchInputError(ValueError):
    """Fail closed while retaining the row-level ingestion audit for callers."""

    def __init__(self, report: BatchNormalizationReport) -> None:
        self.report = report
        super().__init__(
            f"Research requires a fully resolved in-scope batch: "
            f"{len(report.excluded)} excluded, {len(report.quarantined)} quarantined."
        )


@dataclass(frozen=True, slots=True)
class ResearchResult:
    normalization: BatchNormalizationReport
    backtest: TemporalBacktestResult
    calibration: CalibrationReport
    manifest: ExperimentManifest


def run_form_research(
    *,
    observations: list[MatchObservation],
    catalogs: CanonicalCatalogs,
    split_policy: ExpandingWindowPolicy | RollingWindowPolicy,
    model_spec: ModelTrainingSpec,
    calibration_bins: int = 10,
    code_revision: str | None = None,
) -> ResearchResult:
    """Run the declared rolling-form baseline without transport or storage coupling.

    Unresolved or excluded rows stop the run; silently selecting the remaining
    sample would change the research population. Zero imputation is a declared
    baseline convention, accompanied by missingness indicators, not an estimate
    of an unknown team's strength. Scaling is fitted within each training fold.
    """
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive.")
    if any(
        later < earlier + split_policy.evaluation_window
        for earlier, later in zip(split_policy.cutoffs, split_policy.cutoffs[1:], strict=False)
    ):
        raise ValueError("Research evaluation windows must not overlap.")
    normalized = normalize_match_batch(
        observations,
        team_resolver=catalogs.team_resolver(),
        competition_resolver=catalogs.competition_resolver(),
    )
    if normalized.excluded or normalized.quarantined:
        raise ResearchInputError(normalized)
    provider = RollingFormFeatureProvider(normalized.normalized)
    dataset = build_historical_feature_dataset(normalized.normalized, providers=[provider])
    imputation = ImputationPolicy(
        policy_id="rolling_form_zero_with_status_v1",
        rules=tuple(
            ConstantImputationRule(
                feature_name=definition.name,
                value=0.0,
                method="declared_zero_with_missingness_indicator",
            )
            for definition in provider.definitions()
        ),
        include_status_indicators=True,
    )
    if isinstance(split_policy, RollingWindowPolicy):
        folds = build_rolling_window_folds(dataset, split_policy)
    else:
        folds = build_expanding_window_folds(dataset, split_policy)
    backtest = run_temporal_backtest(
        fold_report=folds,
        imputation_policy=imputation,
        model_spec=model_spec,
        trainer=train_sklearn_model,
    )
    calibration = build_calibration_report(
        [
            ScoredPrediction(prediction.actual, prediction.probabilities)
            for fold in backtest.folds
            for prediction in fold.predictions
        ],
        n_bins=calibration_bins,
    )
    manifest = build_experiment_manifest(
        backtest, calibration=calibration, code_revision=code_revision
    )
    return ResearchResult(normalized, backtest, calibration, manifest)
