from __future__ import annotations

from datetime import UTC, datetime

import pytest

from football_analytics.features import (
    ConstantImputationRule,
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    FeatureVector,
    ImputationPolicy,
    MissingFeaturePolicyError,
    feature_set_id_for_definitions,
    materialize_feature_vector,
)

OBSERVED = FeatureDefinition(
    name="example.observed",
    version="v1",
    group="example",
    description="Observed example feature.",
)
MISSING = FeatureDefinition(
    name="example.missing",
    version="v1",
    group="example",
    description="Missing example feature.",
)
IMPUTED = FeatureDefinition(
    name="example.imputed",
    version="v1",
    group="example",
    description="Already-imputed example feature.",
)


def _vector() -> FeatureVector:
    prediction_time = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    definitions = [OBSERVED, MISSING, IMPUTED]

    return FeatureVector(
        match_id="match-1",
        prediction_time=prediction_time,
        feature_set_id=feature_set_id_for_definitions(definitions),
        values=(
            FeatureValue(
                definition=OBSERVED,
                status=FeatureStatus.OBSERVED,
                as_of=prediction_time,
                value=2.5,
                lineage=FeatureLineage(source_record_ids=("observed-1",)),
            ),
            FeatureValue(
                definition=MISSING,
                status=FeatureStatus.MISSING,
                as_of=prediction_time,
                missing_reason=FeatureMissingReason.NO_HISTORY,
            ),
            FeatureValue(
                definition=IMPUTED,
                status=FeatureStatus.IMPUTED,
                as_of=prediction_time,
                value=1500.0,
                missing_reason=FeatureMissingReason.NO_HISTORY,
                imputation_method="legacy_default_rating",
            ),
        ),
    )


def test_materialization_requires_explicit_rule_for_missing_feature() -> None:
    with pytest.raises(MissingFeaturePolicyError, match="no explicit imputation rule"):
        materialize_feature_vector(
            _vector(),
            policy=ImputationPolicy(
                policy_id="no-rules",
                rules=(),
            ),
        )


def test_materialization_adds_numeric_value_and_status_indicators() -> None:
    row = materialize_feature_vector(
        _vector(),
        policy=ImputationPolicy(
            policy_id="research-v1",
            rules=(
                ConstantImputationRule(
                    feature_name="example.missing",
                    value=0.0,
                    method="training_median_v1",
                ),
            ),
        ),
    )
    values = row.as_dict()

    assert values["example.observed"] == pytest.approx(2.5)
    assert values["example.observed__is_missing"] == pytest.approx(0.0)
    assert values["example.observed__is_imputed"] == pytest.approx(0.0)

    assert values["example.missing"] == pytest.approx(0.0)
    assert values["example.missing__is_missing"] == pytest.approx(1.0)
    assert values["example.missing__is_imputed"] == pytest.approx(1.0)

    assert values["example.imputed"] == pytest.approx(1500.0)
    assert values["example.imputed__is_missing"] == pytest.approx(0.0)
    assert values["example.imputed__is_imputed"] == pytest.approx(1.0)

    assert row.applied_imputations[0].method == "training_median_v1"


def test_feature_vector_rejects_mismatched_feature_set_id() -> None:
    prediction_time = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="feature_set_id does not match"):
        FeatureVector(
            match_id="match-1",
            prediction_time=prediction_time,
            feature_set_id="features_incorrect",
            values=(
                FeatureValue(
                    definition=OBSERVED,
                    status=FeatureStatus.OBSERVED,
                    as_of=prediction_time,
                    value=1.0,
                ),
            ),
        )
