from __future__ import annotations

import pytest

from football_analytics.data import DataSourceDefinition, SourceRegistry


def _source(source_id: str) -> DataSourceDefinition:
    return DataSourceDefinition(
        source_id=source_id,
        name="Example Results",
        description="Historical international match results.",
        access_method="manual_csv",
        freshness_expectation="monthly",
        legal_use_notes="Use only under the source's documented license.",
    )


def test_source_registry_registers_and_lists_definitions() -> None:
    registry = SourceRegistry()
    registry.register(_source("results"))

    assert registry.get("results").name == "Example Results"
    assert [source.source_id for source in registry.list()] == ["results"]


def test_source_registry_rejects_duplicate_source_id() -> None:
    registry = SourceRegistry()
    registry.register(_source("results"))

    with pytest.raises(ValueError, match="Duplicate source_id"):
        registry.register(_source("results"))
