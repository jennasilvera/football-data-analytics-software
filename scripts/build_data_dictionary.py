"""Generate the native contract and feature inventory from executable definitions."""

import dataclasses
import importlib
from pathlib import Path

from football_analytics.data import load_canonical_catalogs
from football_analytics.features.composition import build_providers

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "domain.teams",
    "domain.competitions",
    "domain.matches",
    "domain.locations",
    "domain.probabilities",
    "data.contracts",
    "data.observations",
    "data.normalization",
    "features.base",
    "features.materialization",
    "features.fifa_ranking",
    "features.availability",
    "features.competition",
    "features.market",
    "features.travel",
    "models.base",
    "evaluation.metrics",
    "evaluation.diagnostics",
    "ratings.glicko",
)


def main():
    dictionary = ROOT / "docs/data_dictionary.md"
    text = dictionary.read_text().split("## Feature inventory")[0]
    text += "## Feature inventory\n\n"
    catalogs = load_canonical_catalogs(
        teams_path=ROOT / "data/sample/v2/teams.json",
        competitions_path=ROOT / "data/sample/v2/competitions.json",
    )
    providers = build_providers(
        [],
        catalogs,
        groups=(
            "form",
            "strength",
            "schedule",
            "competition",
            "rankings",
            "squad",
            "market",
            "travel",
            "manual",
        ),
    )
    text += "| Feature | Version | Group | Definition |\n|---|---|---|---|\n"
    for provider in providers:
        for field in provider.definitions():
            text += f"| {field.name} | {field.version} | {field.group} | {field.description} |\n"
    text += "\n## Typed record fields\n\n"
    for name in MODULES:
        module = importlib.import_module("football_analytics." + name)
        for class_name, cls in vars(module).items():
            if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
                continue
            if cls.__module__ != module.__name__ or class_name.startswith("_"):
                continue
            text += f"### {name}.{class_name}\n\n"
            text += "| Field | Type | Default |\n|---|---|---|\n"
            for field in dataclasses.fields(cls):
                default = (
                    "required" if field.default is dataclasses.MISSING else repr(field.default)
                )
                annotation = str(field.type).replace("|", "\\|")
                default = default.replace("|", "\\|")
                text += f"| {field.name} | `{annotation}` | `{default}` |\n"
            text += "\n"
    (ROOT / "docs/data_dictionary.md").write_text(text.rstrip() + "\n")


if __name__ == "__main__":
    main()
