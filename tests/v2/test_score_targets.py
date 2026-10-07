from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.cli import main
from football_analytics.data import load_canonical_catalogs, normalize_match_batch
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.data.artifacts import normalization_frames
from football_analytics.domain.scores import REGULATION_TARGET_POLICY_ID, ScoreBasis
from football_analytics.evaluation import ExpandingWindowPolicy
from football_analytics.experiments import ExperimentManifest
from football_analytics.features import build_historical_feature_dataset
from football_analytics.features.history import completed_record_is_before_cutoff
from football_analytics.models.frequency import class_frequency_spec
from football_analytics.models.poisson import PoissonModel, fit_poisson, poisson_spec
from football_analytics.ratings.legacy_elo import rating_input_from_record
from football_analytics.services.research import run_research

SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"
CUTOFF = datetime(2020, 3, 1, tzinfo=UTC)
NON_REGULATION = [basis for basis in ScoreBasis if basis is not ScoreBasis.REGULATION_TIME]


@pytest.fixture
def source():
    frame = pd.read_csv(SAMPLE / "results.csv")
    catalogs = load_canonical_catalogs(
        teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
    )
    observations = build_legacy_mens_results_observations(frame, ingested_at=CUTOFF)
    records = normalize_match_batch(
        observations, team_resolver=catalogs.team_resolver(),
        competition_resolver=catalogs.competition_resolver(),
    ).normalized
    return frame, catalogs, observations, records


def test_missing_basis_stays_unknown_until_explicitly_asserted(source):
    frame = source[0].drop(columns="score_basis")
    unknown = build_legacy_mens_results_observations(frame, ingested_at=CUTOFF)
    assert all(row.score_basis is ScoreBasis.UNKNOWN for row in unknown)
    declared = build_legacy_mens_results_observations(
        frame, ingested_at=CUTOFF, score_basis=ScoreBasis.REGULATION_TIME
    )
    assert all(row.score_basis is ScoreBasis.REGULATION_TIME for row in declared)


@pytest.mark.parametrize("basis", NON_REGULATION)
def test_normalization_retains_but_consumers_reject_other_score_bases(source, basis):
    _, catalogs, observations, _ = source
    observations[0] = replace(observations[0], score_basis=basis)
    normalized = normalize_match_batch(
        observations, team_resolver=catalogs.team_resolver(),
        competition_resolver=catalogs.competition_resolver(),
    )
    assert not normalized.quarantined
    record = next(
        row for row in normalized.normalized
        if row.source_match_id == observations[0].source_match_id
    )
    assert record.score_basis is basis
    frame = normalization_frames(normalized)["normalized"]
    selected = frame[frame.source_match_id == record.source_match_id]
    assert selected.iloc[0]["score_basis"] == basis.value
    consumers = [
        lambda: build_historical_feature_dataset([record], providers=[]),
        lambda: completed_record_is_before_cutoff(record, CUTOFF),
        lambda: rating_input_from_record(record, competition_name="Friendly"),
        lambda: fit_poisson([record], training_cutoff=CUTOFF),
    ]
    for consumer in consumers:
        with pytest.raises(ValueError, match="regulation-time research"):
            consumer()


@pytest.mark.parametrize("basis", NON_REGULATION)
@pytest.mark.parametrize("spec", [class_frequency_spec(), poisson_spec()])
def test_mixed_period_research_fails_even_for_row_outside_evaluation(source, basis, spec):
    _, catalogs, observations, _ = source
    observations[-1] = replace(observations[-1], score_basis=basis)
    with pytest.raises(ValueError, match="regulation-time research"):
        run_research(
            observations=observations, catalogs=catalogs, model_spec=spec,
            split_policy=ExpandingWindowPolicy("test", (CUTOFF,), timedelta(days=30), 12),
        )


@pytest.mark.parametrize("raw", ["after_extra_time", "", None, "unknown"])
def test_assertion_cannot_override_mapped_rows(source, raw):
    frame = source[0].copy()
    frame.loc[0, "score_basis"] = raw
    with pytest.raises(ValueError, match="conflicts"):
        build_legacy_mens_results_observations(
            frame, ingested_at=CUTOFF, score_basis=ScoreBasis.REGULATION_TIME
        )


def test_invalid_period_is_rejected_and_blank_remains_unknown(source):
    frame = source[0].copy()
    frame.loc[0, "score_basis"] = "full_time"
    with pytest.raises(ValueError):
        build_legacy_mens_results_observations(frame, ingested_at=CUTOFF)
    frame.loc[0, "score_basis"] = ""
    rows = build_legacy_mens_results_observations(frame, ingested_at=CUTOFF)
    assert rows[0].score_basis is ScoreBasis.UNKNOWN


def test_poisson_artifact_declares_policy_and_rejects_ambiguous_old_schema(source):
    model = fit_poisson(source[3][:20], training_cutoff=CUTOFF)
    payload = model.to_dict()
    assert payload["target_policy_id"] == REGULATION_TARGET_POLICY_ID
    assert PoissonModel.from_dict(payload) == model
    for altered in [
        {**payload, "target_policy_id": "after_extra_time"},
        {**payload, "schema_version": 1},
        {key: value for key, value in payload.items() if key != "target_policy_id"},
    ]:
        with pytest.raises(ValueError):
            PoissonModel.from_dict(altered)


def test_cli_unknown_source_fails_and_explicit_assertion_succeeds(source, tmp_path, capsys):
    results = tmp_path / "results.csv"
    source[0].drop(columns="score_basis").to_csv(results, index=False)
    args = [
        "research", "--results", str(results), "--teams", str(SAMPLE / "teams.json"),
        "--competitions", str(SAMPLE / "competitions.json"), "--source-id", "synthetic",
        "--ingested-at", CUTOFF.isoformat(), "--assert-senior-mens-a",
        "--cutoff", CUTOFF.isoformat(), "--evaluation-days", "30", "--min-train", "12",
        "--model", "class-frequency", "--output", str(tmp_path / "output"),
    ]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "regulation-time research" in capsys.readouterr().err
    assert not list((tmp_path / "output").glob("research_*.json"))
    main([*args, "--score-basis", "regulation_time"])
    import json
    report = json.loads(Path(json.loads(capsys.readouterr().out)["report"]).read_text())
    assert report["target_policy_id"] == REGULATION_TARGET_POLICY_ID
    assert report["source_score_basis_assertion"] == "regulation_time"
    assert report["manifest"]["target_policy_id"] == REGULATION_TARGET_POLICY_ID
    for altered in [
        {**report["manifest"], "schema_version": 1},
        {**report["manifest"], "target_policy_id": "unknown"},
    ]:
        with pytest.raises(ValueError):
            ExperimentManifest.from_dict(altered)
    assert report["backtest"]["target_policy_id"] == REGULATION_TARGET_POLICY_ID
