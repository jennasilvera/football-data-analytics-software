"""Non-executable JSON inference state for native numeric classifiers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.domain.scores import REGULATION_TARGET_POLICY_ID
from football_analytics.features.materialization import MaterializedFeatureRow
from football_analytics.models.base import ProbabilisticModel
from football_analytics.models.competition_frequency import (
    GROUP_COLUMN,
    CompetitionFrequencyModel,
    competition_code,
)
from football_analytics.models.confederation_frequency import (
    PAIR_COLUMN,
    ConfederationFrequencyModel,
    pair_code,
)
from football_analytics.models.frequency import ClassFrequencyModel
from football_analytics.models.postprocessing import content_id
from football_analytics.models.sklearn_models import SklearnOutcomeModel


@dataclass(frozen=True)
class PortableClassifier:
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        p = self.payload
        if p.get("schema_version") != 1 or p.get("target_policy_id") != REGULATION_TARGET_POLICY_ID:
            raise ValueError("Unsupported portable classifier schema/target.")
        body = {k: v for k, v in p.items() if k != "artifact_id"}
        if p.get("artifact_id") != content_id("classifier_", body):
            raise ValueError("Portable classifier identity mismatch.")
        if p["kind"] not in (
            "frequency",
            "competition_frequency",
            "confederation_frequency",
            "logistic",
            "hist_gradient_boosting",
        ):
            raise ValueError("Unsupported portable classifier kind.")
        if not p["columns"] or len(set(p["columns"])) != len(p["columns"]):
            raise ValueError("Classifier feature names must be unique.")
        if set(p["classes"]) != {"home_win", "draw", "away_win"} or len(p["classes"]) != 3:
            raise ValueError("Classifier requires three canonical classes.")
        # Validate the complete numeric state before any traversal.
        if p["kind"] in ("competition_frequency", "confederation_frequency"):
            column = PAIR_COLUMN if p["kind"] == "confederation_frequency" else GROUP_COLUMN
            if column not in p["columns"]:
                raise ValueError("Competition artifact lacks canonical grouping column.")
            OutcomeProbabilities(**p["state"]["global_probabilities"])
            codes = []
            for code, probabilities in p["state"]["groups"]:
                if type(code) is not int or code < 0 or code in codes:
                    raise ValueError("Invalid or duplicate competition artifact code.")
                if p["kind"] == "confederation_frequency" and not 1 <= code <= 36:
                    raise ValueError("Invalid confederation artifact code.")
                codes.append(code)
                OutcomeProbabilities(**probabilities)
        elif p["kind"] == "logistic":
            state = p["state"]
            n = len(p["columns"])
            for key, shape in (
                ("mean", (n,)),
                ("scale", (n,)),
                ("coef", (3, n)),
                ("intercept", (3,)),
            ):
                values = np.asarray(state[key], dtype=float)
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError("Invalid logistic inference dimensions/values.")
            if any(x <= 0 for x in state["scale"]):
                raise ValueError("Logistic feature scales must be positive.")
        elif p["kind"] == "hist_gradient_boosting":
            state = p["state"]
            if len(state["baseline"]) != 3 or not all(map(math.isfinite, state["baseline"])):
                raise ValueError("Invalid boosting baseline.")
            for stage in state["trees"]:
                if len(stage) != 3:
                    raise ValueError("Boosting stage requires three trees.")
                for tree in stage:
                    for i, node in enumerate(tree):
                        if len(node) != 6 or not math.isfinite(node[0]):
                            raise ValueError("Invalid tree node.")
                        if not node[1]:
                            if (
                                type(node[2]) is not int
                                or not 0 <= node[2] < len(p["columns"])
                                or not math.isfinite(node[3])
                                or any(
                                    type(j) is not int or not i < j < len(tree) for j in node[4:]
                                )
                            ):
                                raise ValueError("Invalid tree split or cyclic tree.")
                    if not tree:
                        raise ValueError("Empty tree.")
        else:
            OutcomeProbabilities(**p["state"])

    @property
    def model_id(self) -> str:
        return str(self.payload["model_id"])

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.payload["columns"])

    @property
    def feature_set_id(self) -> str:
        return str(self.payload["feature_set_id"])

    @property
    def imputation_policy_id(self) -> str:
        return str(self.payload["imputation_policy_id"])

    def predict_row(self, row: MaterializedFeatureRow) -> OutcomeProbabilities:
        if (
            row.feature_set_id != self.feature_set_id
            or row.imputation_policy_id != self.imputation_policy_id
            or tuple(k for k, _ in row.columns) != self.feature_names
        ):
            raise ValueError("Portable classifier feature contract mismatch.")
        x = np.array([v for _, v in row.columns])
        if not np.isfinite(x).all():
            raise ValueError("Portable inference requires finite materialized features.")
        p, state = self.payload, self.payload["state"]
        if p["kind"] == "frequency":
            return OutcomeProbabilities(**state)
        if p["kind"] in ("competition_frequency", "confederation_frequency"):
            probabilities = dict(state["groups"]).get(
                pair_code(row) if p["kind"] == "confederation_frequency" else competition_code(row),
                state["global_probabilities"],
            )
            return OutcomeProbabilities(**probabilities)
        if p["kind"] == "logistic":
            logits = np.asarray(state["coef"]) @ ((x - state["mean"]) / state["scale"])
            logits += state["intercept"]
        else:
            logits = np.array(state["baseline"], dtype=float)
            for stage in state["trees"]:
                for c, tree in enumerate(stage):
                    node = tree[0]
                    while not node[1]:
                        node = tree[node[4] if x[node[2]] <= node[3] else node[5]]
                    logits[c] += node[0]
        exp = np.exp(logits - logits.max())
        mapping = dict(zip(p["classes"], exp / exp.sum(), strict=True))
        return OutcomeProbabilities(*(float(mapping[k]) for k in ("home_win", "draw", "away_win")))


def export_classifier(model: ProbabilisticModel) -> PortableClassifier:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "target_policy_id": REGULATION_TARGET_POLICY_ID,
        "model_id": model.model_id,
        "columns": list(model.feature_names),
        "feature_set_id": model.feature_set_id,
        "imputation_policy_id": model.imputation_policy_id,
        "export_sklearn_version": version("scikit-learn"),
    }
    if isinstance(model, ClassFrequencyModel):
        from dataclasses import asdict

        payload.update(
            kind="frequency",
            state=asdict(model.probabilities),
            classes=["home_win", "draw", "away_win"],
        )
    elif isinstance(model, CompetitionFrequencyModel):
        from dataclasses import asdict

        payload.update(
            kind="confederation_frequency"
            if isinstance(model, ConfederationFrequencyModel)
            else "competition_frequency",
            classes=["home_win", "draw", "away_win"],
            state={
                "global_probabilities": asdict(model.global_probabilities),
                "groups": [[code, asdict(p)] for code, p in model.groups],
            },
        )
    elif isinstance(model, SklearnOutcomeModel):
        estimator = model.estimator
        if isinstance(estimator, Pipeline):
            scaler = estimator.named_steps["scaler"]
            classifier = estimator.named_steps["classifier"]
            payload.update(
                kind="logistic",
                classes=list(classifier.classes_),
                state={
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "coef": classifier.coef_.tolist(),
                    "intercept": classifier.intercept_.tolist(),
                },
            )
        else:
            trees = []
            for stage in estimator._predictors:
                row = []
                for predictor in stage:
                    if any(n["is_categorical"] for n in predictor.nodes):
                        raise ValueError("Categorical boosting export is unsupported.")
                    row.append(
                        [
                            [
                                float(n["value"]),
                                bool(n["is_leaf"]),
                                int(n["feature_idx"]),
                                float(n["num_threshold"]),
                                int(n["left"]),
                                int(n["right"]),
                            ]
                            for n in predictor.nodes
                        ]
                    )
                trees.append(row)
            payload.update(
                kind="hist_gradient_boosting",
                classes=list(estimator.classes_),
                state={"baseline": estimator._baseline_prediction[0].tolist(), "trees": trees},
            )
    else:
        raise ValueError("Unsupported classifier export type.")
    payload["artifact_id"] = content_id("classifier_", payload)
    return PortableClassifier(payload)
