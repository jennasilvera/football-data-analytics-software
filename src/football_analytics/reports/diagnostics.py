"""Human-readable descriptive slices and match forecast errors."""

from collections.abc import Sequence

from football_analytics.evaluation.diagnostics import EvaluationDiagnostics


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_evaluation_diagnostics(reports: Sequence[EvaluationDiagnostics]) -> str:
    if not reports:
        raise ValueError("At least one diagnostics report is required.")
    lines = [
        "# Held-out forecast diagnostics",
        "",
        "Regulation-time home/draw/away outcomes. Lower probability losses are better.",
        "Slices are descriptive, not evidence of significance or a model-promotion decision.",
        "Team slices overlap: each match appears for both participants. Do not pool those counts.",
        "Team accuracy describes forecasts of their matches, not that team's win rate.",
        "The minimum-sample flag is a display warning, not a statistical adequacy test.",
        "",
    ]
    for report in reports:
        lines.extend(
            [
                f"## {_cell(report.model_spec_id)}",
                "",
                f"Evaluated matches: {report.metrics.n_predictions}. "
                f"Minimum display sample: {report.min_sample}.",
                "",
                "| Dimension | Group | N | H/D/A | Accuracy | Log loss | Brier | RPS | Sample |",
                "|---|---|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in report.slices:
            m = row.metrics
            flag = "Small sample" if row.below_minimum else "Descriptive"
            lines.append(
                f"| {row.dimension} | {_cell(row.key)} | {m.n_predictions} | "
                f"{row.home_wins}/{row.draws}/{row.away_wins} | {m.accuracy:.3f} | "
                f"{m.log_loss:.4f} | {m.multiclass_brier_score:.4f} | "
                f"{m.ranked_probability_score:.4f} | {flag} |"
            )
        lines.extend(
            [
                "",
                "### Largest match losses",
                "",
                "Up to ten matches, ordered by log loss. Full match diagnostics are in the JSON.",
                "A large loss means the realized outcome received low probability; it does not",
                "by itself identify the cause of the error or prove the model was unreasonable.",
                "",
                "| Match ID | Date | Home | Away | Score | Actual | Forecast | "
                "P(actual) | Log loss |",
                "|---|---|---|---|---|---|---|---:|---:|",
            ]
        )
        for match in sorted(report.matches, key=lambda item: (-item.log_loss, item.match_id))[:10]:
            lines.append(
                f"| {_cell(match.match_id)} | {match.match_date} | {_cell(match.home_team_id)} | "
                f"{_cell(match.away_team_id)} | {match.home_score}–{match.away_score} | "
                f"{match.actual.value} | {match.predicted.value} | "
                f"{match.actual_probability:.4f} | "
                f"{match.log_loss:.4f} |"
            )
        lines.extend(["", f"Diagnostic identity: `{report.diagnostic_id}`.", ""])
    return "\n".join(lines)
