from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from football_analytics.domain import MatchOutcome
from football_analytics.features import (
    FeatureDefinition,
    FeatureStatus,
    FeatureValue,
    FeatureVector,
    HistoricalFeatureDataset,
    HistoricalFeatureExample,
    ImputationPolicy,
    feature_set_id_for_definitions,
)
from football_analytics.models import (
    ModelInputMismatchError,
    build_model_dataset,
    hist_gradient_boosting_spec,
    logistic_regression_spec,
    train_sklearn_model,
    training_metadata_for,
)

DEFINITIONS = (
    FeatureDefinition(
        name="strength.diff",
        version="v1",
        group="strength",
        description="Synthetic strength difference for model-contract tests.",
    ),
    FeatureDefinition(
        name="form.diff",
        version="v1",
        group="form",
        description="Synthetic form difference for model-contract tests.",
    ),
)
FEATURE_SET_ID = feature_set_id_for_definitions(DEFINITIONS)
IMPUTATION_POLICY = ImputationPolicy(
    policy_id="no_missing_v1",
    rules=(),
    include_status_indicators=False,
)


def _example(
    index: int,
    *,
    target: MatchOutcome,
    strength: float,
    form: float,
) -> HistoricalFeatureExample:
    prediction_time = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index)
    match_id = f"match-{index:02d}"
    vector = FeatureVector(
        match_id=match_id,
        prediction_time=prediction_time,
        feature_set_id=FEATURE_SET_ID,
        values=(
            FeatureValue(
                definition=DEFINITIONS[0],
                status=FeatureStatus.OBSERVED,
                as_of=prediction_time,
                value=strength,
            ),
            FeatureValue(
                definition=DEFINITIONS[1],
                status=FeatureStatus.OBSERVED,
                as_of=prediction_time,
                value=form,
            ),
        ),
    )
    return HistoricalFeatureExample(
        match_id=match_id,
        source_match_id=f"source-{match_id}",
        prediction_time=prediction_time,
        target=target,
        vector=vector,
    )


def _examples() -> tuple[HistoricalFeatureExample, ...]:
    return (
        _example(0, target=MatchOutcome.HOME_WIN, strength=1.8, form=1.2),
        _example(1, target=MatchOutcome.HOME_WIN, strength=1.3, form=0.9),
        _example(2, target=MatchOutcome.DRAW, strength=0.1, form=0.0),
        _example(3, target=MatchOutcome.DRAW, strength=-0.1, form=0.1),
        _example(4, target=MatchOutcome.AWAY_WIN, strength=-1.4, form=-0.8),
        _example(5, target=MatchOutcome.AWAY_WIN, strength=-1.9, form=-1.1),
        _example(6, target=MatchOutcome.HOME_WIN, strength=1.1, form=0.7),
        _example(7, target=MatchOutcome.DRAW, strength=0.0, form=-0.1),
        _example(8, target=MatchOutcome.AWAY_WIN, strength=-1.0, form=-0.6),
    )


def _historical_dataset(
    examples: tuple[HistoricalFeatureExample, ...] | None = None,
) -> HistoricalFeatureDataset:
    return HistoricalFeatureDataset(
        feature_set_id=FEATURE_SET_ID,
        cutoff_policy_id="test_cutoff_v1",
        examples=examples or _examples(),
    )


def test_model_dataset_identity_is_order_independent() -> None:
    examples = _examples()
    forward = build_model_dataset(
        _historical_dataset(examples),
        imputation_policy=IMPUTATION_POLICY,
    )
    reverse = build_model_dataset(
        _historical_dataset(tuple(reversed(examples))),
        imputation_policy=IMPUTATION_POLICY,
    )

    assert forward.dataset_id == reverse.dataset_id
    assert forward.column_names == ("form.diff", "strength.diff")
    assert [example.match_id for example in forward.examples] == [
        f"match-{index:02d}" for index in range(9)
    ]


def test_model_dataset_identity_changes_with_feature_values() -> None:
    baseline = build_model_dataset(
        _historical_dataset(),
        imputation_policy=IMPUTATION_POLICY,
    )
    changed_examples = list(_examples())
    changed_examples[-1] = _example(
        8,
        target=MatchOutcome.AWAY_WIN,
        strength=-0.2,
        form=-0.6,
    )
    changed = build_model_dataset(
        _historical_dataset(tuple(changed_examples)),
        imputation_policy=IMPUTATION_POLICY,
    )

    assert baseline.dataset_id != changed.dataset_id


def test_training_metadata_is_deterministic_for_spec_and_dataset() -> None:
    dataset = build_model_dataset(
        _historical_dataset(),
        imputation_policy=IMPUTATION_POLICY,
    )
    spec = logistic_regression_spec(c=2.0)

    first = training_metadata_for(dataset, spec)
    second = training_metadata_for(dataset, spec)

    assert first == second
    assert first.dataset_id == dataset.dataset_id
    assert first.feature_names == dataset.column_names


@pytest.mark.parametrize(
    "spec",
    [
        logistic_regression_spec(max_iter=500),
        hist_gradient_boosting_spec(max_iter=20, max_leaf_nodes=5),
    ],
)
def test_native_v2_models_produce_valid_three_way_probabilities(spec) -> None:
    dataset = build_model_dataset(
        _historical_dataset(),
        imputation_policy=IMPUTATION_POLICY,
    )

    result = train_sklearn_model(dataset, spec)
    probabilities = result.model.predict_row(dataset.examples[0].row)

    assert result.model.model_id == result.metadata.model_id
    assert probabilities.home_win + probabilities.draw + probabilities.away_win == pytest.approx(
        1.0
    )
    assert all(0.0 <= value <= 1.0 for value in probabilities.as_tuple())


def test_model_rejects_inference_policy_mismatch() -> None:
    dataset = build_model_dataset(
        _historical_dataset(),
        imputation_policy=IMPUTATION_POLICY,
    )
    result = train_sklearn_model(dataset, logistic_regression_spec())
    incompatible = replace(
        dataset.examples[0].row,
        imputation_policy_id="different_policy",
    )

    with pytest.raises(ModelInputMismatchError, match="Imputation-policy"):
        result.model.predict_row(incompatible)


def test_training_requires_all_three_outcome_classes() -> None:
    two_class_examples = tuple(
        example
        for example in _examples()
        if example.target is not MatchOutcome.DRAW
    )
    dataset = build_model_dataset(
        _historical_dataset(two_class_examples),
        imputation_policy=IMPUTATION_POLICY,
    )

    with pytest.raises(ValueError, match="missing canonical outcome classes"):
        train_sklearn_model(dataset, logistic_regression_spec())
