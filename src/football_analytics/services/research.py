from __future__ import annotations

from dataclasses import dataclass

from football_analytics.data import (
    BatchNormalizationReport,
    CanonicalCatalogs,
    MatchObservation,
    normalize_match_batch,
)
from football_analytics.data.confederations import MembershipHistory
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
from football_analytics.evaluation.comparison import ModelComparison, compare_backtests
from football_analytics.evaluation.score_backtest import BacktestScoreForecast, run_poisson_backtest
from football_analytics.experiments import ExperimentManifest, build_experiment_manifest
from football_analytics.features import (
    ConstantImputationRule,
    ImputationPolicy,
    RollingFormFeatureProvider,
    build_historical_feature_dataset,
)
from football_analytics.features.base import FeatureProvider
from football_analytics.models import ModelFamily, ModelTrainingSpec, train_sklearn_model
from football_analytics.models.competition_frequency import (
    CompetitionIdentityProvider,
    train_competition_frequency,
)
from football_analytics.models.confederation_frequency import (
    ConfederationPairProvider,
    train_confederation_frequency,
)
from football_analytics.models.frequency import train_class_frequency
from football_analytics.models.poisson import PoissonModel


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
    score_forecasts: tuple[BacktestScoreForecast, ...] = ()
    score_models: tuple[PoissonModel, ...] = ()


def run_research(
    *,
    observations: list[MatchObservation],
    catalogs: CanonicalCatalogs,
    split_policy: ExpandingWindowPolicy | RollingWindowPolicy,
    model_spec: ModelTrainingSpec,
    calibration_bins: int = 10,
    code_revision: str | None = None,
    feature_groups: tuple[str, ...] | None = None,
    feature_context: dict | None = None,
    membership_history: MembershipHistory | None = None,
) -> ResearchResult:
    """Run a declared model family without transport or storage coupling.

    Unresolved or excluded rows stop the run; silently selecting the remaining
    sample would change the research population. Zero imputation is a declared
    baseline convention, accompanied by missingness indicators, not an estimate
    of an unknown team's strength. Classifier scaling is fitted within each training fold.
    Poisson consumes canonical goals and team context, not the form matrix.
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
    from football_analytics.features.composition import build_providers

    providers: list[FeatureProvider] = (
        [provider]
        if feature_groups is None
        else build_providers(
            normalized.normalized, catalogs, groups=feature_groups, context=feature_context
        )
    )
    if model_spec.family is ModelFamily.COMPETITION_FREQUENCY:
        if feature_groups is not None or feature_context:
            raise ValueError("Competition baseline uses only canonical competition identity.")
        providers = [CompetitionIdentityProvider(catalogs.competitions)]
    if model_spec.family is ModelFamily.CONFEDERATION_FREQUENCY:
        if membership_history is None:
            raise ValueError("Confederation baseline requires explicit membership history.")
        if feature_groups is not None or feature_context:
            raise ValueError("Confederation baseline uses only published membership pairs.")
        team_ids = {team.team_id for team in catalogs.teams}
        if any(r.team_id not in team_ids for r in membership_history.releases):
            raise ValueError("Membership history contains unknown canonical teams.")
        providers = [ConfederationPairProvider(membership_history)]
    dataset = build_historical_feature_dataset(
        normalized.normalized,
        providers=[] if model_spec.family is ModelFamily.POISSON else providers,
    )
    imputation = ImputationPolicy(
        policy_id="rolling_form_zero_with_status_v1",
        rules=tuple(
            ConstantImputationRule(
                feature_name=definition.name,
                value=0.0,
                method="declared_zero_with_missingness_indicator",
            )
            for selected_provider in providers
            for definition in selected_provider.definitions()
        ),
        include_status_indicators=True,
    )
    if isinstance(split_policy, RollingWindowPolicy):
        folds = build_rolling_window_folds(dataset, split_policy)
    else:
        folds = build_expanding_window_folds(dataset, split_policy)
    score_forecasts: tuple[BacktestScoreForecast, ...] = ()
    score_models: tuple[PoissonModel, ...] = ()
    if model_spec.family is ModelFamily.POISSON:
        score_result = run_poisson_backtest(
            fold_report=folds,
            records=normalized.normalized,
            model_spec=model_spec,
        )
        backtest = score_result.backtest
        score_forecasts = score_result.score_forecasts
        score_models = score_result.models
    else:
        backtest = run_temporal_backtest(
            fold_report=folds,
            imputation_policy=imputation,
            model_spec=model_spec,
            trainer=(
                train_confederation_frequency
                if model_spec.family is ModelFamily.CONFEDERATION_FREQUENCY
                else train_competition_frequency
                if model_spec.family is ModelFamily.COMPETITION_FREQUENCY
                else train_class_frequency
                if model_spec.family is ModelFamily.CLASS_FREQUENCY
                else train_sklearn_model
            ),
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
    return ResearchResult(
        normalized, backtest, calibration, manifest, score_forecasts, score_models
    )


# Compatibility alias for clients of the first V2 workflow.
run_form_research = run_research


@dataclass(frozen=True, slots=True)
class ResearchComparisonResult:
    runs: tuple[ResearchResult, ...]
    comparison: ModelComparison


def run_research_comparison(
    *,
    observations: list[MatchObservation],
    catalogs: CanonicalCatalogs,
    split_policy: ExpandingWindowPolicy | RollingWindowPolicy,
    model_specs: tuple[ModelTrainingSpec, ...],
    calibration_bins: int = 10,
    code_revision: str | None = None,
    feature_groups: tuple[str, ...] | None = None,
    feature_context: dict | None = None,
    membership_history: MembershipHistory | None = None,
) -> ResearchComparisonResult:
    """Evaluate declared models on the same source and split; first model is reference."""
    if len(model_specs) < 2:
        raise ValueError("Declare at least two models for comparison.")
    runs = tuple(
        run_research(
            observations=observations,
            catalogs=catalogs,
            split_policy=split_policy,
            model_spec=spec,
            calibration_bins=calibration_bins,
            code_revision=code_revision,
            feature_groups=feature_groups,
            feature_context=feature_context,
            membership_history=membership_history,
        )
        for spec in model_specs
    )
    comparison = compare_backtests(
        [run.backtest for run in runs],
        reference_backtest_run_id=runs[0].backtest.backtest_run_id,
    )
    return ResearchComparisonResult(runs, comparison)
