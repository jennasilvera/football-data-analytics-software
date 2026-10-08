from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.data import load_canonical_catalogs
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.domain import Match, MatchOutcome
from football_analytics.evaluation import ExpandingWindowPolicy, RollingWindowPolicy
from football_analytics.features import PredictionContext
from football_analytics.features.history import ResultEligibilityBasis
from football_analytics.features.materialization import MaterializedFeatureRow
from football_analytics.models.competition_frequency import (
    GROUP_COLUMN,
    CompetitionIdentityProvider,
    competition_frequency_spec,
    train_competition_frequency,
)
from football_analytics.models.dataset import ModelDataset, ModelExample
from football_analytics.models.frequency import class_frequency_spec
from football_analytics.models.portable import PortableClassifier, export_classifier
from football_analytics.models.postprocessing import content_id
from football_analytics.services.research import run_research, run_research_comparison

SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"


def row(code, identity="fixture"):
    return MaterializedFeatureRow(
        identity,
        "2020-01-01T00:00:00+00:00",
        "features",
        "policy",
        ((GROUP_COLUMN, float(code)),),
        (),
    )


def dataset():
    # Global counts H=3,D=1,A=2; competition 0 all home wins; 1 has D,A,A.
    examples = tuple(
        ModelExample(
            str(i),
            "2020-01-02T00:00:00+00:00",
            ResultEligibilityBasis.CONSERVATIVE_NEXT_UTC_DAY,
            target,
            row(i // 3, str(i)),
        )
        for i, target in enumerate(
            (MatchOutcome.HOME_WIN,) * 3
            + (MatchOutcome.DRAW, MatchOutcome.AWAY_WIN, MatchOutcome.AWAY_WIN)
        )
    )
    return ModelDataset(
        "sample", "features", "cutoff", "eligibility", "policy", (GROUP_COLUMN,), examples
    )


def test_shrinkage_exact_counts_and_unseen_competition_fallback():
    trained = train_competition_frequency(dataset(), competition_frequency_spec(prior_strength=3))
    global_p = (4 / 9, 2 / 9, 3 / 9)
    assert trained.model.predict_row(row(99)).as_tuple() == pytest.approx(global_p)
    assert trained.model.predict_row(row(0)).as_tuple() == pytest.approx(
        ((3 + 3 * global_p[0]) / 6, global_p[1] / 2, global_p[2] / 2)
    )
    assert trained.model.predict_row(row(1)).as_tuple() == pytest.approx(
        (global_p[0] / 2, (1 + 3 * global_p[1]) / 6, (2 + 3 * global_p[2]) / 6)
    )
    raw = train_competition_frequency(dataset(), competition_frequency_spec(prior_strength=0))
    assert raw.model.predict_row(row(0)).home_win == 1
    assert raw.model.model_id != trained.model.model_id


def test_portable_replay_and_invalid_group_contract():
    import json

    trained = train_competition_frequency(dataset(), competition_frequency_spec())
    exported = export_classifier(trained.model)
    loaded = PortableClassifier(json.loads(json.dumps(exported.payload)))
    for code in (0, 1, 99):
        assert loaded.predict_row(row(code)) == trained.model.predict_row(row(code))
    for bad in (-1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            loaded.predict_row(row(bad))
    with pytest.raises(ValueError):
        loaded.predict_row(replace(row(0), feature_set_id="wrong-catalog"))
    flagged = replace(row(0), columns=((GROUP_COLUMN, 0.0), (GROUP_COLUMN + "__is_imputed", 1.0)))
    from football_analytics.models.competition_frequency import competition_code

    with pytest.raises(ValueError, match="missing or imputed"):
        competition_code(flagged)
    payload = json.loads(json.dumps(exported.payload))
    payload["state"]["groups"].append(payload["state"]["groups"][0])
    payload["artifact_id"] = content_id(
        "classifier_", {k: v for k, v in payload.items() if k != "artifact_id"}
    )
    with pytest.raises(ValueError, match="duplicate"):
        PortableClassifier(payload)


@pytest.mark.parametrize("rolling", [False, True])
def test_temporal_pairing_and_future_labels_cannot_change_fit(rolling):
    catalogs = load_canonical_catalogs(
        teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
    )
    observations = build_legacy_mens_results_observations(
        pd.read_csv(SAMPLE / "results.csv"), ingested_at=datetime(2020, 6, 1, tzinfo=UTC)
    )
    cutoff = datetime(2020, 3, 1, tzinfo=UTC)
    policy = (
        RollingWindowPolicy("window", (cutoff,), timedelta(days=30), timedelta(days=60), 12)
        if rolling
        else ExpandingWindowPolicy("window", (cutoff,), timedelta(days=30), 12)
    )
    kwargs = dict(catalogs=catalogs, observations=observations, split_policy=policy)
    comparison = run_research_comparison(
        model_specs=(class_frequency_spec(), competition_frequency_spec()), **kwargs
    )
    baseline = comparison.runs[1].backtest
    assert comparison.comparison.paired_prediction_count == baseline.prediction_count
    changed = [
        replace(o, home_score=8, away_score=0) if o.match_date >= cutoff.date() else o
        for o in observations
    ]
    replay = run_research(
        **{**kwargs, "observations": changed}, model_spec=competition_frequency_spec()
    ).backtest
    assert replay.folds[0].model_id == baseline.folds[0].model_id
    assert [p.probabilities for p in replay.folds[0].predictions] == [
        p.probabilities for p in baseline.folds[0].predictions
    ]
    with pytest.raises(ValueError, match="only canonical"):
        run_research(**kwargs, model_spec=competition_frequency_spec(), feature_groups=("form",))


def test_catalog_identity_is_order_stable_and_changes_when_mapping_changes():
    catalogs = load_canonical_catalogs(
        teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
    )
    competitions = (
        *catalogs.competitions,
        replace(catalogs.competitions[0], competition_id="another"),
    )
    first = CompetitionIdentityProvider(competitions)
    assert (
        first.definitions()
        == CompetitionIdentityProvider(tuple(reversed(competitions))).definitions()
    )
    assert first.definitions() != CompetitionIdentityProvider(catalogs.competitions).definitions()
    match = Match("test", date(2020, 2, 1), "ARG", "BRA", "unknown", True)
    with pytest.raises(ValueError, match="Unknown"):
        first.compute(PredictionContext(match, datetime(2020, 1, 1, tzinfo=UTC)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"smoothing": 0},
        {"smoothing": float("nan")},
        {"prior_strength": -1},
        {"prior_strength": float("inf")},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        competition_frequency_spec(**kwargs)
