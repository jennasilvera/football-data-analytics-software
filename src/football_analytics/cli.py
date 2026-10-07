"""Thin adapter for the first executable native V2 research workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from football_analytics.data import load_canonical_catalogs, save_normalization_artifacts
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.data.contracts import ensure_utc
from football_analytics.evaluation import ExpandingWindowPolicy, RollingWindowPolicy
from football_analytics.experiments import JsonExperimentRegistry
from football_analytics.models import hist_gradient_boosting_spec, logistic_regression_spec
from football_analytics.services.research import ResearchInputError, run_form_research


def _timestamp(value: str) -> datetime:
    try:
        return ensure_utc(datetime.fromisoformat(value), "timestamp")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _json_default(value: object) -> str | float:
    if isinstance(value, datetime):
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
    root.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Research artifact content conflict: {path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="football-analytics")
    commands = parser.add_subparsers(dest="command", required=True)
    research = commands.add_parser("research", help="Run a governed rolling-form baseline")
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
    research.add_argument("--cutoff", type=_timestamp, action="append", required=True)
    research.add_argument("--evaluation-days", type=int, required=True)
    research.add_argument("--training-days", type=int, help="Use a bounded rolling window")
    research.add_argument("--min-train", type=int, default=30)
    research.add_argument(
        "--model", choices=["logistic", "hist-gradient-boosting"], default="logistic"
    )
    research.add_argument("--seed", type=int, default=42)
    research.add_argument("--calibration-bins", type=int, default=10)
    research.add_argument("--code-revision")
    research.add_argument("--legal-use-notes")
    research.add_argument("--output", type=Path, default=Path("outputs/v2-research"))
    args = parser.parse_args(argv)
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
        spec = (
            logistic_regression_spec(random_seed=args.seed)
            if args.model == "logistic"
            else hist_gradient_boosting_spec(random_seed=args.seed)
        )
        result = run_form_research(
            observations=observations,
            catalogs=catalogs,
            split_policy=policy,
            model_spec=spec,
            calibration_bins=args.calibration_bins,
            code_revision=args.code_revision,
        )
        payload: dict[str, object] = {
            "schema_version": 1,
            "source_sha256": source_sha256,
            "source_id": args.source_id,
            "ingested_at": args.ingested_at,
            "scope_assertion": "official_senior_mens_a",
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
        report_path = _save_report(payload, args.output)
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
            },
            indent=2,
        )
    )
