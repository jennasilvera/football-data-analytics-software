from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from football_analytics.experiments.manifest import ExperimentManifest

EXPERIMENT_ID_PATTERN = re.compile(r"^experiment_[0-9a-f]{64}$")


class ExperimentRegistryConflictError(RuntimeError):
    """Raised when an existing experiment ID has different stored content."""


class ExperimentRegistry(Protocol):
    """Persistence boundary for immutable experiment manifests."""

    def put(self, manifest: ExperimentManifest) -> Path:
        """Persist a manifest idempotently and return its location."""
        ...

    def get(self, experiment_id: str) -> ExperimentManifest | None:
        """Return a manifest by deterministic ID."""
        ...

    def list_ids(self) -> tuple[str, ...]:
        """Return stored experiment IDs in stable order."""
        ...


@dataclass(slots=True)
class JsonExperimentRegistry:
    """Atomic local JSON implementation of the experiment registry contract."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def put(self, manifest: ExperimentManifest) -> Path:
        destination = self._manifest_path(manifest.experiment_id)
        self.root.mkdir(parents=True, exist_ok=True)

        serialized = json.dumps(
            manifest.to_dict(),
            sort_keys=True,
            indent=2,
        ) + "\n"

        if destination.exists():
            existing = self.get(manifest.experiment_id)
            if existing != manifest:
                raise ExperimentRegistryConflictError(
                    f"Experiment {manifest.experiment_id} already exists with "
                    "different content."
                )
            return destination

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=".experiment-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)

            os.replace(temp_path, destination)
            temp_path = None
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

        return destination

    def get(self, experiment_id: str) -> ExperimentManifest | None:
        path = self._manifest_path(experiment_id)
        if not path.exists():
            return None

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Experiment manifest JSON must contain an object.")

        manifest = ExperimentManifest.from_dict(payload)
        if manifest.experiment_id != experiment_id:
            raise ValueError(
                "Stored experiment manifest ID does not match its registry path."
            )
        return manifest

    def list_ids(self) -> tuple[str, ...]:
        if not self.root.exists():
            return ()

        ids = [
            path.stem
            for path in self.root.glob("experiment_*.json")
            if EXPERIMENT_ID_PATTERN.fullmatch(path.stem)
        ]
        return tuple(sorted(ids))

    def _manifest_path(self, experiment_id: str) -> Path:
        if EXPERIMENT_ID_PATTERN.fullmatch(experiment_id) is None:
            raise ValueError("Invalid experiment_id format.")
        return self.root / f"{experiment_id}.json"
