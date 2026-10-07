"""Versioned snapshot training and unified inference using safe JSON artifacts."""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from football_analytics.data import CanonicalCatalogs, MatchObservation, normalize_match_batch
from football_analytics.data.contracts import ensure_utc
from football_analytics.data.serialization import catalogs_from_dict, json_safe, record_from_dict
from football_analytics.domain import Match, MatchStatus
from football_analytics.domain.scores import REGULATION_TARGET_POLICY_ID
from football_analytics.evaluation import ExpandingWindowPolicy
from football_analytics.features import (
    ConstantImputationRule,
    ImputationPolicy,
    PredictionContext,
    RollingFormFeatureProvider,
    build_feature_vector,
    build_historical_feature_dataset,
    materialize_feature_vector,
)
from football_analytics.features.composition import build_providers
from football_analytics.features.history import DEFAULT_RESULT_ELIGIBILITY_POLICY
from football_analytics.models import (
    build_model_dataset,
    hist_gradient_boosting_spec,
    logistic_regression_spec,
    train_sklearn_model,
)
from football_analytics.models.frequency import class_frequency_spec, train_class_frequency
from football_analytics.models.poisson import PoissonModel, fit_poisson, poisson_spec
from football_analytics.models.portable import PortableClassifier, export_classifier
from football_analytics.models.postprocessing import (
    BaseFitEvidence,
    HeldOutPrediction,
    ProbabilityTransform,
    content_id,
    fit_probability_transform,
)
from football_analytics.services.research import ResearchInputError, run_research_comparison


def form_policy(provider: RollingFormFeatureProvider) -> ImputationPolicy:
    return ImputationPolicy(
        "rolling_form_zero_with_status_v1",
        tuple(
            ConstantImputationRule(
                definition.name,
                0,
                "declared_zero_with_missingness_indicator",
            )
            for definition in provider.definitions()
        ),
        include_status_indicators=True,
    )


def train_forecast_bundle(
    observations: list[MatchObservation],
    catalogs: CanonicalCatalogs,
    *,
    cutoff: datetime,
    holdout_days: int = 30,
    min_calibration: int = 30,
    calibrate: bool = True,
    feature_groups: tuple[str, ...] | None = None,
    feature_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cutoff = ensure_utc(cutoff)
    normalized = normalize_match_batch(
        observations,
        team_resolver=catalogs.team_resolver(),
        competition_resolver=catalogs.competition_resolver(),
    )
    if normalized.excluded or normalized.quarantined:
        raise ResearchInputError(normalized)
    records = tuple(
        record
        for record in normalized.normalized
        if record.match.status is MatchStatus.COMPLETED
        and DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(record).eligible_at <= cutoff
    )
    if not records:
        raise ValueError("No completed available training records.")
    provider = RollingFormFeatureProvider(records)
    feature_context = {
        k: [
            r
            for r in rows
            if r["metadata"].get("available_at")
            and datetime.fromisoformat(r["metadata"]["available_at"]) <= cutoff
        ]
        for k, rows in (feature_context or {}).items()
    }
    providers = (
        [provider]
        if feature_groups is None
        else build_providers(records, catalogs, groups=feature_groups, context=feature_context)
    )
    policy = ImputationPolicy(
        "rolling_form_zero_with_status_v1",
        tuple(
            ConstantImputationRule(d.name, 0, "declared_zero_with_missingness_indicator")
            for p in providers
            for d in p.definitions()
        ),
        include_status_indicators=True,
    )
    historical = build_historical_feature_dataset(records, providers=providers)
    data = build_model_dataset(historical, imputation_policy=policy)
    specs = (
        class_frequency_spec(),
        logistic_regression_spec(),
        hist_gradient_boosting_spec(),
        poisson_spec(),
    )
    classifiers = {}
    for spec in specs[:3]:
        fit = train_class_frequency if spec == specs[0] else train_sklearn_model
        model = fit(data, spec).model
        portable = export_classifier(model)
        # Export gate: never persist a conversion that changes fitted-row predictions.
        for example in data.examples:
            expected = model.predict_row(example.row).as_tuple()
            actual = portable.predict_row(example.row).as_tuple()
            if any(abs(a - b) > 1e-10 for a, b in zip(expected, actual, strict=True)):
                raise ValueError("Portable classifier export failed parity verification.")
        classifiers[spec.spec_id] = portable.payload
    transforms = []
    if calibrate:
        if type(holdout_days) is not int or holdout_days < 1:
            raise ValueError("Calibration holdout days must be positive.")
        inner_cutoff = cutoff - timedelta(days=holdout_days)
        inner = run_research_comparison(
            observations=observations,
            catalogs=catalogs,
            split_policy=ExpandingWindowPolicy(
                "deployment_inner_v1", (inner_cutoff,), timedelta(days=holdout_days), 3
            ),
            model_specs=specs,
            feature_groups=feature_groups,
            feature_context=feature_context,
        )
        folds = [run.backtest.folds[0] for run in inner.runs]
        available = {
            r.match.match_id: DEFAULT_RESULT_ELIGIBILITY_POLICY.eligibility_for(r).eligible_at
            for r in records
        }
        maps = [{p.match_id: p for p in f.predictions} for f in folds]
        members = tuple(
            BaseFitEvidence(s.spec_id, f.model_id, inner_cutoff, f.train_match_ids)
            for s, f in zip(specs, folds, strict=True)
        )
        examples = tuple(
            HeldOutPrediction(
                p.match_id,
                datetime.fromisoformat(p.prediction_time_iso),
                available[p.match_id],
                p.actual,
                tuple(mapping[p.match_id].probabilities for mapping in maps),
            )
            for p in folds[0].predictions
            if p.match_id in available
        )
        from dataclasses import replace

        for i, member in enumerate(members):
            transforms.append(
                fit_probability_transform(
                    method="temperature",
                    members=(member,),
                    examples=tuple(
                        replace(e, probabilities=(e.probabilities[i],)) for e in examples
                    ),
                    fitted_at=cutoff,
                    min_examples=min_calibration,
                ).to_dict()
            )
        transforms.append(
            fit_probability_transform(
                method="convex_ensemble",
                members=members,
                examples=examples,
                fitted_at=cutoff,
                min_examples=min_calibration,
            ).to_dict()
        )
    body = {
        "schema_version": 1,
        "target_policy_id": REGULATION_TARGET_POLICY_ID,
        "training_cutoff": cutoff.isoformat(),
        "training_dataset_id": data.dataset_id,
        "records": json_safe(records),
        "catalogs": json_safe(catalogs),
        "classifiers": classifiers,
        "poisson": fit_poisson(records, training_cutoff=cutoff).to_dict(),
        "transforms": transforms,
        "feature_groups": feature_groups,
        "feature_context": feature_context,
        "history_policy": "frozen_snapshot_research_availability_v1",
        "source_ingested_at": max(r.metadata.ingested_at for r in records).isoformat(),
        "availability_warning": (
            "Historical publication/revision history must be independently verified"
        ),
    }
    return {"bundle_id": content_id("forecast_bundle_", body), **body}


def validate_bundle(bundle: dict[str, Any]) -> None:
    if (
        bundle.get("schema_version") != 1
        or bundle.get("target_policy_id") != REGULATION_TARGET_POLICY_ID
    ):
        raise ValueError("Unsupported forecast bundle schema or target.")
    if bundle.get("bundle_id") != content_id(
        "forecast_bundle_", {k: v for k, v in bundle.items() if k != "bundle_id"}
    ):
        raise ValueError("Forecast bundle content identity mismatch.")


def forecast_match(
    bundle: dict[str, Any], match: Match, *, prediction_time: datetime, method: str = "ensemble"
) -> dict[str, Any]:
    validate_bundle(bundle)
    context = PredictionContext(match, prediction_time)
    if match.status is not MatchStatus.SCHEDULED:
        raise ValueError("Operational forecasts require scheduled fixtures.")
    cutoff = datetime.fromisoformat(bundle["training_cutoff"])
    if context.prediction_time < cutoff:
        raise ValueError("Prediction time precedes bundle training cutoff.")
    if match.kickoff_at is not None and match.kickoff_at.date() != match.match_date:
        raise ValueError("Match date must match UTC kickoff date.")
    catalogs = bundle["catalogs"]
    if match.competition_id not in {c["competition_id"] for c in catalogs["competitions"]}:
        raise ValueError("Unknown canonical competition.")
    if not {match.home_team_id, match.away_team_id} <= {t["team_id"] for t in catalogs["teams"]}:
        raise ValueError("Unknown canonical teams.")
    records = tuple(record_from_dict(r) for r in bundle["records"])
    provider = RollingFormFeatureProvider(records)
    providers = (
        [provider]
        if bundle.get("feature_groups") is None
        else build_providers(
            records,
            catalogs_from_dict(catalogs),
            groups=tuple(bundle["feature_groups"]),
            context=bundle.get("feature_context"),
        )
    )
    vector = build_feature_vector(context, providers)
    policy = ImputationPolicy(
        "rolling_form_zero_with_status_v1",
        tuple(
            ConstantImputationRule(d.name, 0, "declared_zero_with_missingness_indicator")
            for p in providers
            for d in p.definitions()
        ),
        include_status_indicators=True,
    )
    row = materialize_feature_vector(vector, policy=policy)
    probabilities = {
        key: PortableClassifier(p).predict_row(row) for key, p in bundle["classifiers"].items()
    }
    score = PoissonModel.from_dict(bundle["poisson"]).predict_match(
        match, prediction_time=prediction_time
    )
    probabilities["independent_poisson_v1"] = score.probabilities
    if method in probabilities:
        result = probabilities[method]
    else:
        selected = next(
            (
                p
                for p in bundle["transforms"]
                if (method == "ensemble" and p["method"] == "convex_ensemble")
                or (
                    p["method"] == "temperature"
                    and method == "calibrated_" + p["members"][0]["spec_id"]
                )
            ),
            None,
        )
        if selected is None:
            raise ValueError("Requested forecast method is unavailable in this bundle.")
        transform = ProbabilityTransform.from_dict(selected)
        result = transform.predict(
            {m.spec_id: probabilities[m.spec_id] for m in transform.members},
            prediction_time=prediction_time,
        )
    missing = [x.feature_name for x in row.applied_imputations]
    drivers = []
    logistic = next((p for p in bundle["classifiers"].values() if p["kind"] == "logistic"), None)
    if logistic:
        state = logistic["state"]
        c = logistic["classes"].index(result.predicted_outcome.value)
        drivers = sorted(
            [
                {
                    "feature": name,
                    "logistic_logit_contribution": (value - state["mean"][i])
                    / state["scale"][i]
                    * state["coef"][c][i],
                }
                for i, (name, value) in enumerate(row.columns)
            ],
            key=lambda x: abs(float(x["logistic_logit_contribution"])),
            reverse=True,
        )[:8]
    body = {
        "schema_version": 1,
        "bundle_id": bundle["bundle_id"],
        "method": method,
        "match": json_safe(match),
        "prediction_time": context.prediction_time.isoformat(),
        "target_policy_id": REGULATION_TARGET_POLICY_ID,
        "probabilities": asdict(result),
        "base_probabilities": {k: asdict(v) for k, v in probabilities.items()},
        "poisson_score_forecast": asdict(score),
        "score_forecast_note": (
            "Uncalibrated Poisson goals; selected 1X2 transform does not recalibrate scores"
        ),
        "entropy_nats": -sum(p * math.log(p) for p in result.as_tuple() if p > 0),
        "uncertainty_interval": None,
        "uncertainty_note": (
            "Probability interval coverage is not validated; base "
            "disagreement is not a confidence interval"
        ),
        "confidence_category": "unvalidated",
        "missing_features": missing,
        "feature_vector": json_safe(vector),
        "explanatory_drivers": drivers,
        "explanation_note": (
            "Contributions to logistic member logits, not causal effects or ensemble attribution"
        ),
        "data_timestamp": cutoff.isoformat(),
        "history_age_days": (context.prediction_time - cutoff).total_seconds() / 86400,
        "market_disagreement": None,
    }
    return {"forecast_id": content_id("forecast_", body), **body}
