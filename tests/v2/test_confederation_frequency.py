import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.data import load_canonical_catalogs
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.data.confederations import MembershipHistory, load_membership_history
from football_analytics.domain import Match, MatchOutcome
from football_analytics.evaluation import ExpandingWindowPolicy, RollingWindowPolicy
from football_analytics.features import PredictionContext
from football_analytics.features.base import FeatureStatus
from football_analytics.features.history import ResultEligibilityBasis
from football_analytics.features.materialization import MaterializedFeatureRow
from football_analytics.models.confederation_frequency import (
    PAIR_COLUMN,
    PAIRS,
    ConfederationPairProvider,
    confederation_frequency_spec,
    train_confederation_frequency,
)
from football_analytics.models.dataset import ModelDataset, ModelExample
from football_analytics.models.frequency import class_frequency_spec
from football_analytics.models.portable import PortableClassifier, export_classifier
from football_analytics.models.postprocessing import content_id
from football_analytics.services.research import run_research, run_research_comparison

SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"


def row(code, identity="fixture"):
    return MaterializedFeatureRow(
        identity, "2020-01-01T00:00:00Z", "features", "policy", ((PAIR_COLUMN, float(code)),), ()
    )


def dataset():
    outcomes = (
        MatchOutcome.HOME_WIN,
        MatchOutcome.HOME_WIN,
        MatchOutcome.DRAW,
        MatchOutcome.AWAY_WIN,
        MatchOutcome.AWAY_WIN,
    )
    examples = tuple(
        ModelExample(
            str(i),
            "2020-01-02T00:00:00Z",
            ResultEligibilityBasis.CONSERVATIVE_NEXT_UTC_DAY,
            target,
            row((1, 1, 2, 0, 0)[i], str(i)),
        )
        for i, target in enumerate(outcomes)
    )
    return ModelDataset(
        "sample", "features", "cutoff", "eligibility", "policy", (PAIR_COLUMN,), examples
    )


def test_group_counts_shrinkage_unknown_and_unseen_fallback():
    fit = train_confederation_frequency(dataset(), confederation_frequency_spec(prior_strength=2))
    global_p = (3 / 8, 2 / 8, 3 / 8)
    assert fit.model.predict_row(row(0)).as_tuple() == pytest.approx(global_p)
    assert fit.model.predict_row(row(36)).as_tuple() == pytest.approx(global_p)
    assert fit.model.predict_row(row(1)).as_tuple() == pytest.approx(
        ((2 + 2 * global_p[0]) / 4, global_p[1] / 2, global_p[2] / 2)
    )
    assert 0 not in dict(fit.model.groups)
    raw = train_confederation_frequency(dataset(), confederation_frequency_spec(prior_strength=0))
    assert raw.model.predict_row(row(1)).home_win == 1


def test_portable_replay_and_invalid_codes():
    model = train_confederation_frequency(dataset(), confederation_frequency_spec()).model
    payload = json.loads(json.dumps(export_classifier(model).payload))
    loaded = PortableClassifier(payload)
    for code in (0, 1, 2, 36):
        assert loaded.predict_row(row(code)) == model.predict_row(row(code))
    for code in (-1, 37, 1.5, float("nan")):
        with pytest.raises(ValueError):
            loaded.predict_row(row(code))
    with pytest.raises(ValueError):
        loaded.predict_row(replace(row(1), feature_set_id="changed"))
    payload["state"]["groups"][0][0] = 0
    payload["artifact_id"] = content_id(
        "classifier_", {k: v for k, v in payload.items() if k != "artifact_id"}
    )
    with pytest.raises(ValueError, match="confederation artifact"):
        PortableClassifier(payload)


def test_provider_preserves_orientation_unknown_status_and_release_lineage():
    history = load_membership_history(
        SAMPLE / "memberships.json", team_ids={"ARG", "BRA", "FRA", "GER"}
    )
    provider = ConfederationPairProvider(history)
    match = Match("fixture", date(2020, 4, 1), "ARG", "FRA", "friendly", True)
    context = PredictionContext(match, datetime(2020, 3, 1, tzinfo=UTC))
    home = provider.compute(context)[0]
    away = provider.compute(
        replace(context, match=replace(match, home_team_id="FRA", away_team_id="ARG"))
    )[0]
    assert home.value == PAIRS.index(("CONMEBOL", "UEFA")) + 1
    assert away.value == PAIRS.index(("UEFA", "CONMEBOL")) + 1
    assert home.status is FeatureStatus.OBSERVED
    assert len(home.lineage.artifact_ids) == 2
    unknown = ConfederationPairProvider(MembershipHistory(())).compute(context)[0]
    assert unknown.value == 0
    assert unknown.status is FeatureStatus.IMPUTED
    assert unknown.imputation_method == "unknown_membership_global_fallback"


@pytest.mark.parametrize("rolling", [False, True])
def test_future_outcomes_and_releases_do_not_change_fitted_forecasts(rolling):
    catalogs = load_canonical_catalogs(
        teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
    )
    observations = build_legacy_mens_results_observations(
        pd.read_csv(SAMPLE / "results.csv"), ingested_at=datetime(2020, 6, 1, tzinfo=UTC)
    )
    history = load_membership_history(
        SAMPLE / "memberships.json", team_ids={t.team_id for t in catalogs.teams}
    )
    cutoff = datetime(2020, 3, 1, tzinfo=UTC)
    policy = (
        RollingWindowPolicy("window", (cutoff,), timedelta(days=30), timedelta(days=60), 12)
        if rolling
        else ExpandingWindowPolicy("window", (cutoff,), timedelta(days=30), 12)
    )
    kwargs = dict(
        observations=observations,
        catalogs=catalogs,
        split_policy=policy,
        membership_history=history,
    )
    compared = run_research_comparison(
        model_specs=(class_frequency_spec(), confederation_frequency_spec()), **kwargs
    )
    baseline = compared.runs[1].backtest
    assert baseline.prediction_count == compared.runs[0].backtest.prediction_count
    later = replace(
        history.releases[0],
        periods=(),
        metadata=replace(
            history.releases[0].metadata, available_at=datetime(2020, 5, 1, tzinfo=UTC)
        ),
    )
    revised = run_research(
        **{
            **kwargs,
            "observations": [
                replace(o, home_score=7, away_score=0) if o.match_date >= cutoff.date() else o
                for o in observations
            ],
            "membership_history": MembershipHistory((*history.releases, later)),
        },
        model_spec=confederation_frequency_spec(),
    ).backtest
    assert baseline.folds[0].model_id == revised.folds[0].model_id
    assert [p.probabilities for p in baseline.folds[0].predictions] == [
        p.probabilities for p in revised.folds[0].predictions
    ]
    with pytest.raises(ValueError, match="explicit membership"):
        run_research(
            **{**kwargs, "membership_history": None}, model_spec=confederation_frequency_spec()
        )
