from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataSourceDefinition:
    """Governance metadata for an external or manually maintained data source."""

    source_id: str
    name: str
    description: str
    access_method: str
    freshness_expectation: str
    legal_use_notes: str

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "name",
            "description",
            "access_method",
            "freshness_expectation",
            "legal_use_notes",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must not be blank.")
            object.__setattr__(self, field_name, value)


class SourceRegistry:
    """Small in-process registry for governed data-source definitions."""

    def __init__(self) -> None:
        self._sources: dict[str, DataSourceDefinition] = {}

    def register(self, source: DataSourceDefinition) -> None:
        if source.source_id in self._sources:
            raise ValueError(f"Duplicate source_id: {source.source_id}")
        self._sources[source.source_id] = source

    def get(self, source_id: str) -> DataSourceDefinition:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise KeyError(f"Unknown source_id: {source_id}") from exc

    def list(self) -> tuple[DataSourceDefinition, ...]:
        return tuple(
            self._sources[source_id] for source_id in sorted(self._sources)
        )
