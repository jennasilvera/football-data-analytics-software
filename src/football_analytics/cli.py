"""Thin adapter for the first executable native V2 research workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from datetime import date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from football_analytics.data import load_canonical_catalogs, save_normalization_artifacts
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.data.contracts import ensure_utc
from football_analytics.domain import Match
from football_analytics.domain.scores import REGULATION_TARGET_POLICY_ID, ScoreBasis
from football_analytics.evaluation import ExpandingWindowPolicy, RollingWindowPolicy
from football_analytics.evaluation.diagnostics import build_evaluation_diagnostics
from football_analytics.experiments import JsonExperimentRegistry
from football_analytics.models import hist_gradient_boosting_spec, logistic_regression_spec
from football_analytics.models.artifacts import (
    load_poisson_model,
    save_poisson_model,
    save_probability_transform,
)
from football_analytics.models.frequency import class_frequency_spec
from football_analytics.models.poisson import poisson_spec
from football_analytics.reports.comparison import render_model_comparison
from football_analytics.reports.diagnostics import render_evaluation_diagnostics
from football_analytics.services.nested_research import NestedHoldoutPolicy, run_nested_research
from football_analytics.services.research import (
    ResearchInputError,
    run_research,
    run_research_comparison,
)
from football_analytics.storage.files import publish_immutable


def _timestamp(value: str) -> datetime:
    try:
        return ensure_utc(datetime.fromisoformat(value), "timestamp")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _json_default(value: object) -> str | float:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    raise TypeError(f"Unsupported research artifact value: {type(value).__name__}")


def _save_report(payload: dict[str, object], root: Path) -> Path:
    """Publish complete, content-addressed bytes without replacing an existing run."""
    data = (
        json.dumps(payload, sort_keys=True, indent=2, default=_json_default, allow_nan=False) + "\n"
    ).encode()
    path = root / f"research_{hashlib.sha256(data).hexdigest()}.json"
    return publish_immutable(path, data)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="football-analytics")
    commands = parser.add_subparsers(dest="command", required=True)
    research = commands.add_parser(
        "research", help="Run governed temporal research or compare supported models"
    )
    research.add_argument("--results", type=Path, required=True)
    research.add_argument("--teams", type=Path, required=True)
    research.add_argument("--competitions", type=Path, required=True)
    research.add_argument("--source-id", required=True)
    research.add_argument("--ingested-at", type=_timestamp, required=True)
    research.add_argument(
        "--assert-senior-mens-a",
        action="store_true",
        required=True,
        help="Assert that the input source contains official senior men's A matches",
    )
    research.add_argument(
        "--score-basis", choices=[basis.value for basis in ScoreBasis],
        default=ScoreBasis.UNKNOWN.value,
        help="Assert source score basis when no per-row score_basis column exists",
    )
    research.add_argument("--cutoff", type=_timestamp, action="append", required=True)
    research.add_argument("--evaluation-days", type=int, required=True)
    research.add_argument("--training-days", type=int, help="Use a bounded rolling window")
    research.add_argument("--min-train", type=int, default=30)
    research.add_argument(
        "--model",
        choices=["logistic", "hist-gradient-boosting", "class-frequency", "poisson", "all"],
        default="logistic",
    )
    research.add_argument("--seed", type=int, default=42)
    research.add_argument("--calibration-bins", type=int, default=10)
    research.add_argument("--postprocess-days", type=int,
                          help="Fit nested probability transforms; requires --model all")
    research.add_argument("--min-postprocess", type=int, default=30)
    research.add_argument("--min-diagnostic-sample", type=int, default=30)
    research.add_argument("--code-revision")
    research.add_argument("--legal-use-notes")
    research.add_argument("--output", type=Path, default=Path("outputs/v2-research"))
    predict = commands.add_parser(
        "predict-poisson", help="Forecast from a validated JSON model artifact"
    )
    predict.add_argument("--model-artifact", type=Path, required=True)
    predict.add_argument("--match-id", required=True)
    predict.add_argument("--home-id", required=True)
    predict.add_argument("--away-id", required=True)
    predict.add_argument("--competition-id", required=True)
    predict.add_argument("--match-date", type=date.fromisoformat, required=True)
    predict.add_argument("--kickoff", type=_timestamp)
    predict.add_argument("--prediction-time", type=_timestamp, required=True)
    predict.add_argument("--venue", choices=["neutral", "home"], required=True)
    predict.add_argument("--output", type=Path, default=Path("outputs/v2-forecasts"))
    args = parser.parse_args(argv)
    if args.command == "predict-poisson":
        try:
            model = load_poisson_model(args.model_artifact)
            match = Match(
                args.match_id,
                args.match_date,
                args.home_id,
                args.away_id,
                args.competition_id,
                args.venue == "neutral",
                kickoff_at=args.kickoff,
            )
            if match.kickoff_at is not None and match.kickoff_at.date() != match.match_date:
                raise ValueError("match-date must equal the UTC kickoff date.")
            forecast = model.predict_match(match, prediction_time=args.prediction_time)
            path = _save_report(
                {
                    "schema_version": 2,
                    "target_policy_id": REGULATION_TARGET_POLICY_ID,
                    "kind": "poisson_forecast",
                    "model_id": model.model_id,
                    "training_dataset_id": model.training_dataset_id,
                    "training_cutoff": model.training_cutoff,
                    "match": asdict(match),
                    "prediction_time": args.prediction_time,
                    "forecast": asdict(forecast),
                },
                args.output,
            )
        except (ValueError, TypeError, OSError, KeyError) as exc:
            parser.exit(2, f"Forecast failed: {exc}\n")
        print(json.dumps({"forecast_report": str(path), "model_id": model.model_id}))
        return
    try:
        # Hash and parse the same bytes, so lineage cannot describe another read.
        import io

        raw = args.results.read_bytes()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        catalogs = load_canonical_catalogs(
            teams_path=args.teams, competitions_path=args.competitions
        )
        observations = build_legacy_mens_results_observations(
            pd.read_csv(io.BytesIO(raw)),
            ingested_at=args.ingested_at,
            source_id=args.source_id,
            source_version=source_sha256,
            legal_use_notes=args.legal_use_notes,
            score_basis=ScoreBasis(args.score_basis),
        )
        policy: ExpandingWindowPolicy | RollingWindowPolicy
        if args.training_days is None:
            policy = ExpandingWindowPolicy(
                "cli_expanding_v1",
                tuple(args.cutoff),
                timedelta(days=args.evaluation_days),
                args.min_train,
            )
        else:
            policy = RollingWindowPolicy(
                "cli_rolling_v1",
                tuple(args.cutoff),
                timedelta(days=args.evaluation_days),
                timedelta(days=args.training_days),
                args.min_train,
            )
        specs = {
            "class-frequency": class_frequency_spec(),
            "logistic": logistic_regression_spec(random_seed=args.seed),
            "hist-gradient-boosting": hist_gradient_boosting_spec(random_seed=args.seed),
            "poisson": poisson_spec(),
        }
        comparison = None
        nested = None
        if args.postprocess_days is not None and args.model != "all":
            raise ValueError("--postprocess-days requires --model all.")
        if args.model == "all":
            if args.postprocess_days is not None:
                nested = run_nested_research(
                    observations=observations, catalogs=catalogs, split_policy=policy,
                    model_specs=tuple(specs.values()), calibration_bins=args.calibration_bins,
                    code_revision=args.code_revision, nested_policy=NestedHoldoutPolicy(
                        timedelta(days=args.postprocess_days), args.min_postprocess,
                    ),
                )
            compared = nested if nested is not None else run_research_comparison(
                observations=observations,
                catalogs=catalogs,
                split_policy=policy,
                model_specs=tuple(specs.values()),
                calibration_bins=args.calibration_bins,
                code_revision=args.code_revision,
            )
            runs = compared.runs
            comparison = compared.comparison
            spec = specs["class-frequency"]
        else:
            spec = specs[args.model]
            runs = (
                run_research(
                    observations=observations,
                    catalogs=catalogs,
                    split_policy=policy,
                    model_spec=spec,
                    calibration_bins=args.calibration_bins,
                    code_revision=args.code_revision,
                ),
            )
        result = runs[0]
        diagnostics = tuple(build_evaluation_diagnostics(
            run.backtest, run.normalization.normalized, min_sample=args.min_diagnostic_sample
        ) for run in runs)
        payload: dict[str, object] = {
            "schema_version": 5,
            "target_policy_id": REGULATION_TARGET_POLICY_ID,
            "source_sha256": source_sha256,
            "source_id": args.source_id,
            "ingested_at": args.ingested_at,
            "scope_assertion": "official_senior_mens_a",
            "source_score_basis_assertion": args.score_basis,
            "legal_use_notes": args.legal_use_notes,
            "catalogs": asdict(catalogs),
            "split_policy": asdict(policy),
            "model_spec": asdict(spec),
            "normalized_count": len(result.normalization.normalized),
            "backtest": asdict(result.backtest),
            "calibration": asdict(result.calibration),
            "manifest": result.manifest.to_dict(),
            "runtime": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")},
            },
        }
        payload["runs"] = [
            {
                "backtest": asdict(run.backtest),
                "calibration": asdict(run.calibration),
                "manifest": run.manifest.to_dict(),
                "score_forecasts": [asdict(forecast) for forecast in run.score_forecasts],
                "score_models": [model.to_dict() for model in run.score_models],
            }
            for run in runs
        ]
        payload["diagnostics"] = [asdict(report) for report in diagnostics]
        payload["nested_holdout"] = (
            {"holdout_days": args.postprocess_days, "min_examples": args.min_postprocess,
             "folds": [audit.to_dict() for audit in nested.audits]}
            if nested is not None else None
        )
        payload["comparison"] = asdict(comparison) if comparison is not None else None
        report_path = _save_report(payload, args.output)
        diagnostics_path = publish_immutable(
            report_path.with_suffix(".diagnostics.md"),
            render_evaluation_diagnostics(diagnostics).encode(),
        )
        model_paths = [
            str(save_poisson_model(model, args.output / "models"))
            for run in runs
            for model in run.score_models
        ]
        if nested is not None:
            model_paths.extend(
                str(save_probability_transform(model, args.output / "models"))
                for audit in nested.audits for model in audit.transforms
            )
        for run in runs[1:]:
            JsonExperimentRegistry(args.output / "experiments").put(run.manifest)
        comparison_path = None
        if comparison is not None:
            comparison_path = publish_immutable(
                report_path.with_suffix(".md"),
                render_model_comparison(comparison).encode(),
            )
        manifest_path = JsonExperimentRegistry(args.output / "experiments").put(result.manifest)
    except ResearchInputError as exc:
        paths = save_normalization_artifacts(exc.report, args.output / "rejected-input")
        parser.exit(2, f"{exc} Audit: {paths['quarantined'].parent}\n")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Research failed: {exc}\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "manifest": str(manifest_path),
                "metrics": asdict(result.backtest.aggregate_metrics),
                "model_artifacts": model_paths,
                "diagnostics_report": str(diagnostics_path),
                "comparison_report": str(comparison_path) if comparison_path else None,
            },
            indent=2,
        )
    )
