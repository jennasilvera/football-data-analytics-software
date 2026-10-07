from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.cli import main
from football_analytics.data import load_canonical_catalogs
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.domain import MatchOutcome
from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.evaluation import ExpandingWindowPolicy, RollingWindowPolicy
from football_analytics.models.artifacts import (
    load_probability_transform,
    save_probability_transform,
)
from football_analytics.models.frequency import class_frequency_spec
from football_analytics.models.poisson import poisson_spec
from football_analytics.models.postprocessing import (
    BaseFitEvidence,
    HeldOutPrediction,
    ProbabilityTransform,
    fit_probability_transform,
)
from football_analytics.services.nested_research import NestedHoldoutPolicy, run_nested_research

CUTOFF = datetime(2020, 3, 1, tzinfo=UTC)
SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"


@pytest.fixture
def sample():
    members = (BaseFitEvidence("one", "model_one", CUTOFF - timedelta(days=30), ("train",)),)
    examples = tuple(
        HeldOutPrediction(
            f"held_{i}",
            CUTOFF - timedelta(days=20 - i),
            CUTOFF - timedelta(days=19 - i),
            outcome,
            (OutcomeProbabilities(0.98, 0.01, 0.01),),
        )
        for i, outcome in enumerate(tuple(MatchOutcome) * 3)
    )
    return members, examples


def fit(sample, **kwargs):
    members, examples = sample
    return fit_probability_transform(
        method="temperature",
        members=members,
        examples=examples,
        fitted_at=CUTOFF,
        min_examples=3,
        **kwargs,
    )


def test_temperature_softens_overconfidence_and_is_replayable(sample, tmp_path):
    model = fit(sample)
    assert model.temperature > 1
    prediction = model.predict(
        {"one": OutcomeProbabilities(0.98, 0.01, 0.01)}, prediction_time=CUTOFF
    )
    assert prediction.home_win < 0.98
    assert sum(prediction.as_tuple()) == pytest.approx(1)
    path = save_probability_transform(model, tmp_path)
    assert load_probability_transform(path) == model
    assert save_probability_transform(model, tmp_path) == path
    assert ProbabilityTransform.from_dict(model.to_dict()) == model
    for changed in [{"temperature": 1.0}, {"target_policy_id": "unknown"}, {"schema_version": 2}]:
        with pytest.raises(ValueError):
            ProbabilityTransform.from_dict({**model.to_dict(), **changed})
    with pytest.raises(ValueError, match="cutoff"):
        model.predict(
            {"one": OutcomeProbabilities(0.3, 0.3, 0.4)},
            prediction_time=CUTOFF - timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match="member specs"):
        model.predict({"other": OutcomeProbabilities(0.3, 0.3, 0.4)}, prediction_time=CUTOFF)


@pytest.mark.parametrize(
    "failure",
    [
        "in_sample",
        "late_label",
        "late_base",
        "duplicate",
        "missing_class",
        "insufficient",
        "wrong_width",
    ],
)
def test_invalid_fitting_evidence_fails_closed(sample, failure):
    members, examples = sample
    if failure == "in_sample":
        members = (replace(members[0], train_match_ids=(examples[0].match_id,)),)
    elif failure == "late_label":
        examples = (
            replace(examples[0], target_available_at=CUTOFF + timedelta(days=1)),
            *examples[1:],
        )
    elif failure == "late_base":
        members = (replace(members[0], fitted_at=CUTOFF),)
    elif failure == "duplicate":
        examples = (*examples, examples[0])
    elif failure == "missing_class":
        examples = tuple(replace(e, actual=MatchOutcome.HOME_WIN) for e in examples)
    elif failure == "insufficient":
        examples = examples[:2]
    else:
        examples = (
            replace(examples[0], probabilities=examples[0].probabilities * 2),
            *examples[1:],
        )
    with pytest.raises(ValueError):
        fit((members, examples))


def test_convex_ensemble_favors_better_member_and_handles_zero_probabilities(sample):
    members, examples = sample
    members = (*members, replace(members[0], spec_id="two", model_id="model_two"))
    examples = tuple(
        replace(
            e,
            probabilities=(
                OutcomeProbabilities(
                    *(1.0 if outcome is e.actual else 0.0 for outcome in MatchOutcome)
                ),
                OutcomeProbabilities(1 / 3, 1 / 3, 1 / 3),
            ),
        )
        for e in examples
    )
    model = fit_probability_transform(
        method="convex_ensemble",
        members=members,
        examples=examples,
        fitted_at=CUTOFF,
        min_examples=3,
    )
    assert model.weights[0] > 0.99
    result = model.predict(
        {"one": OutcomeProbabilities(1, 0, 0), "two": OutcomeProbabilities(1 / 3, 1 / 3, 1 / 3)},
        prediction_time=CUTOFF,
    )
    assert result.home_win > 0.99
    assert sum(model.weights) == pytest.approx(1)
    assert ProbabilityTransform.from_dict(model.to_dict()) == model
    assert (
        fit_probability_transform(
            method="convex_ensemble",
            members=members,
            examples=examples,
            fitted_at=CUTOFF,
            min_examples=3,
        )
        == model
    )


@pytest.fixture
def inputs():
    return dict(
        observations=build_legacy_mens_results_observations(
            pd.read_csv(SAMPLE / "results.csv"), ingested_at=datetime(2020, 6, 1, tzinfo=UTC)
        ),
        catalogs=load_canonical_catalogs(
            teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
        ),
        split_policy=ExpandingWindowPolicy("outer", (CUTOFF,), timedelta(days=30), 8),
        model_specs=(class_frequency_spec(), poisson_spec()),
        nested_policy=NestedHoldoutPolicy(timedelta(days=21), 5),
    )


def test_outer_labels_cannot_change_fitted_transforms_or_first_forecast(inputs):
    original = run_nested_research(**inputs)
    changed = [
        replace(o, home_score=8, away_score=0) if o.match_date >= CUTOFF.date() else o
        for o in inputs["observations"]
    ]
    modified = run_nested_research(**{**inputs, "observations": changed})
    assert original.audits[0].transforms == modified.audits[0].transforms
    assert len(original.runs) == 5
    assert len(original.comparison.rows) == 5
    for before, after in zip(original.runs, modified.runs, strict=True):
        assert before.backtest.folds[0].model_id == after.backtest.folds[0].model_id
        assert (
            before.backtest.folds[0].predictions[0].probabilities
            == after.backtest.folds[0].predictions[0].probabilities
        )
    for model in original.audits[0].transforms:
        assert all(e.target_available_at <= CUTOFF for e in model.examples)
        assert all(
            e.match_id not in m.train_match_ids for e in model.examples for m in model.members
        )


def test_late_inner_result_is_audited_and_never_fits_transform(inputs):
    before = run_nested_research(**inputs).audits[0]
    observations = inputs["observations"]
    index = max(i for i, o in enumerate(observations) if o.match_date < CUTOFF.date())
    observations[index] = replace(
        observations[index],
        metadata=replace(observations[index].metadata, available_at=CUTOFF + timedelta(days=3)),
    )
    result = run_nested_research(**inputs)
    audit = result.audits[0]
    assert len(audit.unavailable_match_ids) == len(before.unavailable_match_ids) + 1
    assert set(before.unavailable_match_ids) < set(audit.unavailable_match_ids)
    assert all(
        e.match_id not in audit.unavailable_match_ids
        for model in audit.transforms
        for e in model.examples
    )


def test_rolling_inner_training_stays_within_outer_population(inputs):
    inputs["split_policy"] = RollingWindowPolicy(
        "outer", (CUTOFF,), timedelta(days=30), timedelta(days=50), 8
    )
    result = run_nested_research(**inputs)
    allowed = set(result.runs[0].backtest.folds[0].train_match_ids)
    for model in result.audits[0].transforms:
        assert {e.match_id for e in model.examples} <= allowed
        assert all(set(member.train_match_ids) <= allowed for member in model.members)
    inputs["nested_policy"] = NestedHoldoutPolicy(timedelta(days=50), 5)
    with pytest.raises(ValueError, match="shorter"):
        run_nested_research(**inputs)


def test_insufficient_holdout_stops_instead_of_silently_using_identity(inputs):
    inputs["nested_policy"] = NestedHoldoutPolicy(timedelta(days=21), 30)
    with pytest.raises(ValueError, match="Insufficient held-out"):
        run_nested_research(**inputs)


def test_cli_saves_all_nested_artifacts_and_replays(tmp_path, capsys):
    import json

    args = [
        "research",
        "--results",
        str(SAMPLE / "results.csv"),
        "--teams",
        str(SAMPLE / "teams.json"),
        "--competitions",
        str(SAMPLE / "competitions.json"),
        "--source-id",
        "synthetic",
        "--ingested-at",
        "2020-06-01T00:00:00Z",
        "--assert-senior-mens-a",
        "--cutoff",
        CUTOFF.isoformat(),
        "--evaluation-days",
        "30",
        "--min-train",
        "8",
        "--model",
        "all",
        "--postprocess-days",
        "21",
        "--min-postprocess",
        "5",
        "--output",
        str(tmp_path),
    ]
    main(args)
    first = json.loads(capsys.readouterr().out)
    main(args)
    assert first == json.loads(capsys.readouterr().out)
    report = json.loads(Path(first["report"]).read_text())
    assert len(report["runs"]) == 9
    assert len(report["nested_holdout"]["folds"][0]["transforms"]) == 5
    assert len(first["model_artifacts"]) == 6
    for path in first["model_artifacts"]:
        if "probability_transform_" in path:
            model = load_probability_transform(Path(path))
            assert len(model.examples) >= 5
    assert len(list((tmp_path / "experiments").glob("*.json"))) == 9
