"""Operator-readable data onboarding reports without ingestion side effects."""

import hashlib
import json
from pathlib import Path

import pytest

from football_analytics.cli import main
from football_analytics.storage.repository import Repository

SAMPLE = Path(__file__).resolve().parents[2] / "data/sample/v2"
HEADER = "date,home_team,away_team,home_score,away_score,tournament,neutral,score_basis\n"
VALID = "2020-01-01,Argentina,Brazil,2,0,International Friendly,True,regulation_time\n"


@pytest.mark.parametrize("case", ["valid", "unknown", "duplicate", "empty", "fixtures"])
def test_validation_reports_and_exit_codes_without_ingestion(tmp_path, capsys, case):
    content = HEADER + VALID
    if case == "unknown":
        content += VALID.replace("Brazil", "Atlantis").replace("2020-01-01", "2020-01-02")
    elif case == "duplicate":
        content += VALID
    elif case == "empty":
        content = HEADER
    elif case == "fixtures":
        content = "date,home_team,away_team,tournament,neutral\n"
        content += "2030-01-01,Argentina,Brazil,International Friendly,True\n"
    source = tmp_path / "data.csv"
    source.write_text(content)
    output = tmp_path / "validation.json"
    database = tmp_path / "db"
    command = [
        "validate-data", "--results", str(source), "--teams", str(SAMPLE / "teams.json"),
        "--competitions", str(SAMPLE / "competitions.json"), "--source-id", "onboarding",
        "--assert-senior-mens-a", "--score-basis", "regulation_time",
        "--legal-use-notes", "Synthetic test", "--database", str(database),
        "--output", str(output),
    ]
    if case == "fixtures":
        command += ["--data-kind", "fixtures"]
    invalid = case in ["unknown", "duplicate", "empty"]
    def run():
        if invalid:
            with pytest.raises(SystemExit) as exc:
                main(command)
            assert exc.value.code == 1
        else:
            main(command)
        return json.loads(capsys.readouterr().out)
    report = run()
    assert report == json.loads(output.read_text())
    assert report["valid"] is not invalid
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["data_kind"] == ("fixtures" if case == "fixtures" else "results")
    assert run() == report  # Same report and immutable output on repeated validation.
    with Repository(database).connection() as db:
        assert db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0
    if case == "empty":
        assert report["batch_errors"] == ["empty_batch"]
    elif invalid:
        assert report["counts"]["quarantined"] == 1
        assert report["issues"][0]["source_match_id"]
        assert report["issues"][0]["reasons"]
        assert report["matches"] == 1
    else:
        assert report["counts"]["total"] == 1
        assert report["issues"] == []
        assert report["normalized_matches"][0]["home_team_id"] == "ARG"

    if case == "valid":
        changed = json.loads((SAMPLE / "teams.json").read_text())
        changed.setdefault("aliases", {})["Argentina A"] = "ARG"
        teams_path = tmp_path / "teams.json"
        teams_path.write_text(json.dumps(changed))
        command[command.index("--teams") + 1] = str(teams_path)
        command[command.index("--output") + 1] = str(tmp_path / "changed-report.json")
        revised = run()
        assert revised["normalized_matches"] == report["normalized_matches"]
        assert revised["source_sha256"] == report["source_sha256"]
        assert revised["catalog_id"] != report["catalog_id"]
        assert revised["report_id"] != report["report_id"]

    if case in ["unknown", "duplicate"]:
        ingest = ["ingest-results", *command[1:]]
        # Keep the existing report immutable; rejection happens before export.
        with pytest.raises(SystemExit) as exc:
            main(ingest)
        assert exc.value.code == 2
        assert "Ingestion refused" in capsys.readouterr().err
        with Repository(database).connection() as db:
            assert db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0


def test_unparseable_source_is_operational_error_without_partial_report(tmp_path, capsys):
    source = tmp_path / "bad.csv"
    source.write_text(HEADER + VALID.replace("2020-01-01", "not-a-date"))
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        main([
            "validate-data", "--results", str(source), "--teams", str(SAMPLE / "teams.json"),
            "--competitions", str(SAMPLE / "competitions.json"), "--source-id", "test",
            "--assert-senior-mens-a", "--legal-use-notes", "Synthetic test",
            "--database", str(tmp_path / "db"), "--output", str(output),
        ])
    assert exc.value.code == 2
    assert not output.exists()
    assert not capsys.readouterr().out


def test_report_includes_out_of_scope_decision_from_canonical_normalization():
    from datetime import UTC, date, datetime

    from football_analytics.data import load_canonical_catalogs, normalize_match_batch
    from football_analytics.data.contracts import SourceMetadata
    from football_analytics.data.observations import MatchObservation
    from football_analytics.data.scope import GenderCategory, TeamLevel
    from football_analytics.domain import MatchStatus
    from football_analytics.services.data_quality import validation_report

    catalogs = load_canonical_catalogs(
        teams_path=SAMPLE / "teams.json", competitions_path=SAMPLE / "competitions.json"
    )
    observation = MatchObservation(
        source_match_id="out-of-scope-1", match_date=date(2020, 1, 1),
        home_team_name="Argentina", away_team_name="Brazil",
        competition_name="International Friendly", neutral=True,
        status=MatchStatus.SCHEDULED, gender=GenderCategory.MEN, team_level=TeamLevel.YOUTH,
        official=True, metadata=SourceMetadata(
            source="test", source_record_id="out-of-scope-1",
            ingested_at=datetime(2020, 1, 1, tzinfo=UTC),
        ),
    )
    batch = normalize_match_batch(
        [observation], team_resolver=catalogs.team_resolver(),
        competition_resolver=catalogs.competition_resolver(),
    )
    report = validation_report(
        batch, source_id="test", source_sha256="digest", catalog_id="catalog", data_kind="fixtures"
    )
    assert not report["valid"]
    assert report["counts"]["excluded"] == 1
    assert report["issues"][0]["decision"] == "excluded"
    assert report["issues"][0]["source_match_id"] == "out-of-scope-1"
    assert report["issues"][0]["reasons"]
