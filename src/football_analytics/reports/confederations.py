"""Human-readable coverage and proper scores for historical membership slices."""


def render_confederation_diagnostics(reports: list[dict]) -> str:
    lines = [
        "# Historical confederation diagnostics",
        "",
        "Membership is resolved from releases published by each prediction cutoff.",
        "Unknown memberships remain in the sample. Home, away and ordered-pair slices",
        "each partition the same evaluation population; do not pool across dimensions.",
        "",
    ]
    for report in reports:
        coverage = report["coverage"]
        lines.extend(
            [
                f"## {report['report_id']}",
                "",
                f"Both memberships known: {coverage['both_known']} / "
                f"{coverage['matches']} matches.",
                f"Minimum descriptive slice size: {report['min_sample']}.",
                "",
                "| Dimension | Membership | Matches | Log loss | Brier | RPS | Small sample |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in report["slices"]:
            metrics = row["metrics"]
            lines.append(
                f"| {row['dimension']} | {row['key']} | {metrics['n_predictions']} | "
                f"{metrics['log_loss']:.4f} | {metrics['multiclass_brier_score']:.4f} | "
                f"{metrics['ranked_probability_score']:.4f} | "
                f"{'yes' if row['below_minimum'] else 'no'} |"
            )
        lines.extend(["", f"Membership dataset: `{report['membership_dataset_id']}`.", ""])
    lines.append(
        "Descriptive evaluation only; membership publication claims require source evidence."
    )
    return "\n".join(lines) + "\n"
