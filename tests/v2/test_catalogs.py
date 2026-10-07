from __future__ import annotations

import json

from football_analytics.data import load_canonical_catalogs
from football_analytics.data.entity_resolution import ResolutionStatus


def test_catalog_loader_builds_validated_resolvers(tmp_path) -> None:
    teams_path = tmp_path / "teams.json"
    competitions_path = tmp_path / "competitions.json"

    teams_path.write_text(
        json.dumps(
            {
                "teams": [
                    {
                        "team_id": "USA",
                        "name": "United States",
                        "fifa_code": "USA",
                        "confederation": "CONCACAF",
                    }
                ],
                "aliases": {"USA": "USA"},
            }
        ),
        encoding="utf-8",
    )
    competitions_path.write_text(
        json.dumps(
            {
                "competitions": [
                    {
                        "competition_id": "friendly",
                        "name": "International Friendly",
                        "kind": "friendly",
                    }
                ],
                "aliases": {"Friendly": "friendly"},
            }
        ),
        encoding="utf-8",
    )

    catalogs = load_canonical_catalogs(
        teams_path=teams_path,
        competitions_path=competitions_path,
    )

    assert catalogs.team_resolver().resolve("USA").status is ResolutionStatus.RESOLVED
    assert (
        catalogs.competition_resolver().resolve("Friendly").status
        is ResolutionStatus.RESOLVED
    )
