"""Validate linked research and bind human approval to an exact evidence revision.

Internal consistency checks do not establish source truth or predictive skill.
"""

from __future__ import annotations

from datetime import datetime

from football_analytics.data.contracts import ensure_utc
from football_analytics.data.serialization import json_safe
from football_analytics.domain import MatchOutcome
from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.domain.scores import REGULATION_TARGET_POLICY_ID
from football_analytics.evaluation.calibration import build_calibration_report
from football_analytics.evaluation.metrics import ScoredPrediction, evaluate_predictions
from football_analytics.experiments import ExperimentManifest
from football_analytics.services.forecasting import validate_bundle
from football_analytics.storage.repository import Repository


def validate_research(report: dict, bundle: dict) -> None:
    """Require reproducible metrics, manifest links and serving-family coverage."""
    validate_bundle(bundle)
    try:
        _validate_research(report, bundle)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed research evidence.") from exc


def _validate_research(report: dict, bundle: dict) -> None:
    if report.get("schema_version") != 5 or report.get("target_policy_id") != (
        REGULATION_TARGET_POLICY_ID
    ):
        raise ValueError("Unsupported research schema or target policy.")
    digest = report.get("source_sha256")
    if not digest or digest not in {r["metadata"]["source_version"] for r in bundle["records"]}:
        raise ValueError("Research source digest does not match the model's training source.")
    if report["catalogs"] != bundle["catalogs"]:
        raise ValueError("Research catalogs do not match the model catalogs.")
    cutoff = ensure_utc(datetime.fromisoformat(bundle["training_cutoff"]))
    context = {
        key: [row for row in rows if row["metadata"].get("available_at")
              and ensure_utc(datetime.fromisoformat(row["metadata"]["available_at"])) <= cutoff]
        for key, rows in (report.get("feature_context") or {}).items()
    }
    if context != bundle["feature_context"]:
        raise ValueError("Research context does not match the frozen model inputs.")
    if json_safe(report.get("feature_groups")) != json_safe(bundle["feature_groups"]):
        raise ValueError("Research feature groups do not match the model.")
    if not report.get("runs"):
        raise ValueError("Research report must contain evaluated runs.")
    manifests = []
    for run in report["runs"]:
        manifest = ExperimentManifest.from_dict(run["manifest"])
        backtest = run["backtest"]
        for field in (
            "backtest_run_id", "target_policy_id", "model_spec_id", "model_family",
            "model_version", "model_random_seed", "model_parameters", "split_policy_id",
            "feature_set_id", "cutoff_policy_id", "result_eligibility_policy_id",
            "imputation_policy_id", "aggregate_metrics",
        ):
            if backtest[field] != manifest.to_dict()[field]:
                raise ValueError(f"Research manifest/backtest mismatch: {field}.")
        all_scored = []
        seen = set()
        folds = backtest["folds"]
        for singular, plural in (
            ("fold_id", "fold_ids"), ("training_run_id", "training_run_ids"),
            ("model_id", "model_ids"), ("train_dataset_id", "train_dataset_ids"),
            ("evaluation_dataset_id", "evaluation_dataset_ids"),
        ):
            if tuple(f[singular] for f in folds) != getattr(manifest, plural):
                raise ValueError("Research fold lineage does not match the manifest.")
        for fold in folds:
            cutoff = ensure_utc(datetime.fromisoformat(fold["cutoff_iso"]))
            train_ids = fold["train_match_ids"]
            if len(set(train_ids)) != len(train_ids) or len(train_ids) != fold["train_count"]:
                raise ValueError("Research training sample count or identities are inconsistent.")
            scored = []
            for prediction in fold["predictions"]:
                match_id = prediction["match_id"]
                if not match_id or match_id in seen or match_id in train_ids:
                    raise ValueError("Research evaluation repeats a match or overlaps training.")
                seen.add(match_id)
                timestamp = ensure_utc(datetime.fromisoformat(prediction["prediction_time_iso"]))
                if timestamp < cutoff or any(
                    prediction[field] != fold[field] for field in ("fold_id", "model_id")
                ):
                    raise ValueError("Research prediction time or fold lineage is inconsistent.")
                scored.append(ScoredPrediction(
                    MatchOutcome(prediction["actual"]),
                    OutcomeProbabilities(**prediction["probabilities"]),
                ))
            if len(scored) != fold["test_count"] or json_safe(evaluate_predictions(scored)) != (
                fold["metrics"]
            ):
                raise ValueError("Research fold metrics do not reproduce from predictions.")
            all_scored.extend(scored)
        if json_safe(evaluate_predictions(all_scored)) != backtest["aggregate_metrics"]:
            raise ValueError("Research aggregate metrics do not reproduce from predictions.")
        if len(all_scored) != manifest.prediction_count:
            raise ValueError("Research prediction count does not match the manifest.")
        bins = manifest.calibration_n_bins
        if type(bins) is not int or not 1 <= bins <= 1000:
            raise ValueError("Research requires calibration diagnostics with 1..1000 bins.")
        calibration = build_calibration_report(all_scored, n_bins=bins)
        if json_safe(calibration) != run["calibration"] or (
            calibration.report_id != manifest.calibration_report_id
            or calibration.macro_expected_calibration_error
            != manifest.macro_expected_calibration_error
        ):
            raise ValueError("Research calibration does not reproduce from predictions.")
        manifests.append(manifest)
    for spec_id, classifier in bundle["classifiers"].items():
        if not any(
            m.model_spec_id == spec_id
            and m.feature_set_id == classifier["feature_set_id"]
            and m.imputation_policy_id == classifier["imputation_policy_id"]
            for m in manifests
        ):
            raise ValueError(f"Research is missing serving model/feature coverage: {spec_id}.")
    if not any(m.model_spec_id == "independent_poisson_v1" for m in manifests):
        raise ValueError("Research is missing serving Poisson coverage.")
    for transform in bundle["transforms"]:
        if transform["method"] == "temperature":
            member = transform["members"][0]["spec_id"]
            spec_id, family = "temperature_" + member, "temperature_scaling"
        else:
            member = next(iter(bundle["classifiers"]))
            spec_id, family = "convex_probability_ensemble_v1", "convex_ensemble"
        if not any(
            m.model_spec_id == spec_id and m.model_family == family
            and m.feature_set_id == "base_probabilities_v1"
            and m.imputation_policy_id == "no_missing_base_probabilities_v1"
            and (family != "temperature_scaling"
                 or dict(m.model_parameters).get("base_spec_id") == member)
            for m in manifests
        ):
            raise ValueError(f"Research is missing calibrated/ensemble coverage: {spec_id}.")


def require_approved_model(repo: Repository, model_id: str, *, as_of: datetime) -> dict:
    """Return the approved release payload for read-only callers."""
    release: dict = require_approved_model_record(repo, model_id, as_of=as_of)["payload"]
    return release


def require_approved_model_record(repo: Repository, model_id: str, *, as_of: datetime) -> dict:
    """Return the exact validated approval record for transactional publication guards."""
    try:
        approval = repo.get("model_release", model_id, as_of=as_of)
        release: dict = approval["payload"]
        metrics = repo.get("model_metrics", model_id, as_of=as_of)
        model = repo.get("model", model_id, as_of=as_of)
    except KeyError as exc:
        raise ValueError("Model requires reviewed research before serving.") from exc
    if release.get("status") != "approved" or (
        release.get("metrics_record_id") != metrics["record_id"]
        or release.get("model_record_id") != model["record_id"]
    ):
        raise ValueError(
            "Model approval is absent, revoked or superseded; review current evidence."
        )
    return approval
