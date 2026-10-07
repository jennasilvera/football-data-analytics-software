"""Portable Poisson inference state, validated JSON only (no executable pickle)."""

from __future__ import annotations

import json
from pathlib import Path

from football_analytics.models.poisson import PoissonModel
from football_analytics.models.postprocessing import ProbabilityTransform
from football_analytics.storage.files import publish_immutable


def save_poisson_model(model: PoissonModel, root: Path) -> Path:
    content = json.dumps(model.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    return publish_immutable(root / f"{model.model_id}.json", content.encode())


def load_poisson_model(path: Path) -> PoissonModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Poisson model JSON must contain an object.")
    return PoissonModel.from_dict(payload)


def save_probability_transform(model: ProbabilityTransform, root: Path) -> Path:
    content = json.dumps(model.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    return publish_immutable(root / f"{model.model_id}.json", content.encode())


def load_probability_transform(path: Path) -> ProbabilityTransform:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Probability transform JSON must contain an object.")
    return ProbabilityTransform.from_dict(payload)
