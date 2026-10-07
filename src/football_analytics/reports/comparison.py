from __future__ import annotations

from football_analytics.evaluation.comparison import ModelComparison


def render_model_comparison(comparison: ModelComparison) -> str:
    lines = [
        "# Football model comparison",
        "",
        f"Paired out-of-sample forecasts: **{comparison.paired_prediction_count}**.",
        "",
        f"Reference run: `{comparison.reference_backtest_run_id}`.",
        "",
        "Target: regulation time (including stoppage time); excludes extra time and shootouts.",
        "All models use the same training and evaluation matches and prediction cutoffs.",
        "Lower log loss, Brier score, and ranked probability score (RPS) are better.",
        "Negative deltas indicate a lower score than the reference.",
        "",
        "| Model | Accuracy | Log loss | Brier | RPS | Δ log loss | Δ Brier | Δ RPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.rows:
        metrics = row.metrics
        lines.append(
            f"| {row.model_spec_id or row.model_family} | "
            f"{metrics.accuracy:.4f} | {metrics.log_loss:.4f} | "
            f"{metrics.multiclass_brier_score:.4f} | {metrics.ranked_probability_score:.4f} | "
            f"{row.log_loss_delta_vs_reference:+.4f} | {row.brier_delta_vs_reference:+.4f} | "
            f"{row.rps_delta_vs_reference:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "These are descriptive paired scores, not a significance test or a promotion decision.",
            "Demo data tests software behavior; it does not establish forecasting skill.",
            "Model selection needs nested temporal validation and an untouched final holdout.",
            "The Poisson model assumes independent goals and conditions probabilities on a finite",
            "score grid. Its omitted tail probability is reported for every match.",
            "Calibration diagnostics describe outer predictions. With nested postprocessing,",
            "temperatures and ensemble weights are fitted only on earlier held-out results.",
            "",
            "## Run identities",
            "",
        ]
    )
    lines.extend(f"- {row.model_family}: `{row.backtest_run_id}`" for row in comparison.rows)
    return "\n".join(lines) + "\n"
