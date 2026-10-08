import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from football_analytics.cli import main
from football_analytics.data import load_canonical_catalogs
from football_analytics.data.adapters.legacy import build_legacy_mens_results_observations
from football_analytics.data.confederations import (
    MembershipHistory,
    MembershipPeriod,
    MembershipRelease,
    load_membership_history,
)
from football_analytics.data.contracts import LeakageRisk, SourceMetadata
from football_analytics.data.serialization import json_safe
from football_analytics.domain.teams import Confederation
from football_analytics.evaluation import ExpandingWindowPolicy
from football_analytics.evaluation.confederations import confederation_diagnostics
from football_analytics.evaluation.diagnostics import build_evaluation_diagnostics
from football_analytics.models.frequency import class_frequency_spec
from football_analytics.services.research import run_research

SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"


def release(team="ARG", published=datetime(2019, 1, 1, tzinfo=UTC), periods=None):
    return MembershipRelease(
        team,
        periods
        if periods is not None
        else (MembershipPeriod(Confederation.CONMEBOL, date(2018, 1, 1)),),
        SourceMetadata(
            "synthetic-membership",
            datetime(2021, 1, 1, tzinfo=UTC),
            available_at=published,
            source_record_id=team + published.isoformat(),
            legal_use_notes="Synthetic test fixture, not actual membership history",
        ),
    )


def test_valid_time_and_publication_time_are_independent():
    old = release()
    new = release(
        published=datetime(2020, 1, 15, tzinfo=UTC),
        periods=(
            MembershipPeriod(Confederation.CONMEBOL, date(2018, 1, 1), date(2020, 2, 1)),
            MembershipPeriod(Confederation.AFC, date(2020, 2, 1)),
        ),
    )
    history = MembershipHistory((new, old))
    assert (
        history.resolve("ARG", date(2020, 2, 1), datetime(2020, 1, 14, tzinfo=UTC))[0]
        == Confederation.CONMEBOL
    )
    assert history.resolve("ARG", date(2020, 2, 1), datetime(2020, 1, 15, tzinfo=UTC)) == (
        Confederation.AFC,
        new.release_id,
    )
    assert (
        history.resolve("ARG", date(2020, 1, 31), datetime(2020, 1, 30, tzinfo=UTC))[0]
        == Confederation.CONMEBOL
    )
    assert history.resolve("ARG", date(2017, 1, 1), datetime(2020, 1, 30, tzinfo=UTC)) == (
        None,
        new.release_id,
    )
    assert history.resolve("BRA", date(2020, 1, 1), datetime(2020, 1, 1, tzinfo=UTC)) == (
        None,
        None,
    )
    assert history.dataset_id == MembershipHistory((old, new)).dataset_id
    # A complete replacement can retract all earlier claims, without reviving old history.
    retraction = replace(new, periods=())
    assert MembershipHistory((old, retraction)).resolve(
        "ARG", date(2020, 2, 1), datetime(2020, 1, 20, tzinfo=UTC)
    ) == (None, retraction.release_id)


def test_invalid_and_ambiguous_releases_fail_closed():
    valid = release()
    with pytest.raises(ValueError, match="overlap"):
        replace(
            valid, periods=(*valid.periods, MembershipPeriod(Confederation.AFC, date(2020, 1, 1)))
        )
    with pytest.raises(ValueError, match="duplicate"):
        MembershipHistory((valid, replace(valid, periods=())))
    for metadata in (
        replace(valid.metadata, available_at=None),
        replace(valid.metadata, leakage_risk=LeakageRisk.REVIEW),
        replace(valid.metadata, legal_use_notes=""),
        replace(valid.metadata, source_record_id=None),
    ):
        with pytest.raises(ValueError, match="provenance"):
            replace(valid, metadata=metadata)
    with pytest.raises(ValueError):
        MembershipPeriod(Confederation.AFC, date(2020, 1, 1), date(2020, 1, 1))
    with pytest.raises(ValueError):
        MembershipHistory((valid,)).resolve("ARG", date(2020, 1, 1), datetime(2020, 1, 1))


def test_json_loader_roundtrip_and_unknown_team(tmp_path):
    path = tmp_path / "memberships.json"
    path.write_text(json.dumps({"schema_version": 1, "releases": json_safe((release(),))}))
    assert (
        load_membership_history(path, team_ids={"ARG"}).dataset_id
        == MembershipHistory((release(),)).dataset_id
    )
    with pytest.raises(ValueError, match="unknown canonical"):
        load_membership_history(path, team_ids={"BRA"})


def test_slices_keep_unknown_population_and_future_revisions_out():
    run = run_research(
        observations=build_legacy_mens_results_observations(
            pd.read_csv(SAMPLE / "results.csv"), ingested_at=datetime(2020, 6, 1, tzinfo=UTC)
        ),
        catalogs=load_canonical_catalogs(
            teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
        ),
        split_policy=ExpandingWindowPolicy(
            "test", (datetime(2020, 3, 1, tzinfo=UTC),), timedelta(days=30), 12
        ),
        model_spec=class_frequency_spec(),
    )
    diagnostics = build_evaluation_diagnostics(run.backtest, run.normalization.normalized)
    history = MembershipHistory((release(),))
    report = confederation_diagnostics(diagnostics, history)
    assert report["coverage"]["matches"] == 10
    assert report["coverage"]["both_known"] == 0
    for dimension in ("home_confederation", "away_confederation", "confederation_pair"):
        slices = [r for r in report["slices"] if r["dimension"] == dimension]
        assert sum(r["metrics"]["n_predictions"] for r in slices) == 10
        assert sum(
            r["metrics"]["log_loss"] * r["metrics"]["n_predictions"] for r in slices
        ) / 10 == pytest.approx(diagnostics.metrics.log_loss)
    later = release(published=datetime(2020, 5, 1, tzinfo=UTC), periods=())
    future = confederation_diagnostics(diagnostics, MembershipHistory((release(), later)))
    assert report["assignments"] == future["assignments"]
    assert report["slices"] == future["slices"]
    assert report["report_id"] != future["report_id"]  # Full input provenance remains explicit.


def test_cli_embeds_membership_provenance_and_partitioned_metrics(tmp_path, capsys):
    path = tmp_path / "memberships.json"
    path.write_text(json.dumps({"schema_version": 1, "releases": json_safe((release(),))}))
    main(
        [
            "research",
            "--results",
            str(SAMPLE / "results.csv"),
            "--teams",
            str(SAMPLE / "teams.json"),
            "--competitions",
            str(SAMPLE / "competitions.json"),
            "--source-id",
            "synthetic-v2",
            "--ingested-at",
            "2020-06-01T00:00:00Z",
            "--assert-senior-mens-a",
            "--cutoff",
            "2020-03-01T00:00:00Z",
            "--evaluation-days",
            "30",
            "--min-train",
            "12",
            "--model",
            "class-frequency",
            "--membership-history",
            str(path),
            "--output",
            str(tmp_path / "reports"),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    report = json.loads(Path(output["report"]).read_text())
    assert "Unknown memberships remain" in Path(output["confederation_report"]).read_text()
    membership = report["confederation_diagnostics"][0]
    assert membership["membership_dataset_id"] == MembershipHistory((release(),)).dataset_id
    assert len(membership["assignments"]) == 10
    assert membership["membership_releases"][0]["metadata"]["source"] == "synthetic-membership"
