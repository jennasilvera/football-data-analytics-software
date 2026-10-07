"""Predeclared leave-one-family-out ablations with paired temporal samples."""

from __future__ import annotations

from dataclasses import dataclass

from football_analytics.evaluation.comparison import ModelComparison, compare_backtests
from football_analytics.models import ModelFamily
from football_analytics.services.research import ResearchResult, run_research


@dataclass(frozen=True)
class AblationResult:
    runs: tuple[ResearchResult, ...]
    removed_groups: tuple[str, ...]
    comparison: ModelComparison


def run_feature_ablation(*, feature_groups: tuple[str, ...], **research_args) -> AblationResult:
    if len(feature_groups) < 2 or len(set(feature_groups)) != len(feature_groups):
        raise ValueError("Ablation requires at least two unique declared feature groups.")
    if research_args["model_spec"].family not in (
        ModelFamily.LOGISTIC_REGRESSION,
        ModelFamily.HIST_GRADIENT_BOOSTING,
    ):
        raise ValueError("Feature ablation requires a feature-dependent classifier.")
    groups = (
        feature_groups,
        *(tuple(g for g in feature_groups if g != removed) for removed in feature_groups),
    )
    runs = tuple(run_research(feature_groups=selected, **research_args) for selected in groups)
    return AblationResult(
        runs,
        ("none", *feature_groups),
        compare_backtests(
            [r.backtest for r in runs], reference_backtest_run_id=runs[0].backtest.backtest_run_id
        ),
    )
