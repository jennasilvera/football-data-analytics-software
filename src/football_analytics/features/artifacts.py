from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    FeatureVector,
)


FEATURE_VECTOR_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FeatureVectorArtifact:
    """Persisted, content-addressed feature vector."""

    artifact_id: str
    vector: FeatureVector
    schema_version: int = FEATURE_VECTOR_ARTIFACT_SCHEMA_VERSION


def build_feature_vector_artifact(
    vector: FeatureVector,
) -> FeatureVectorArtifact:
    """Build a deterministic content-addressed wrapper for a feature vector."""

    payload = _vector_payload(vector)
    artifact_id = _artifact_id(
        schema_version=FEATURE_VECTOR_ARTIFACT_SCHEMA_VERSION,
        payload=payload,
    )
    return FeatureVectorArtifact(
        artifact_id=artifact_id,
        vector=vector,
    )


def save_feature_vector_artifact(
    vector: FeatureVector,
    path: str | Path,
) -> FeatureVectorArtifact:
    """Persist a feature vector as deterministic JSON."""

    artifact = build_feature_vector_artifact(vector)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    document = _artifact_document(artifact)
    destination.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact


def load_feature_vector_artifact(
    path: str | Path,
) -> FeatureVectorArtifact:
    """Load and verify a persisted feature-vector artifact."""

    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))

    schema_version = int(document["schema_version"])
    if schema_version != FEATURE_VECTOR_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported feature-vector artifact schema version: "
            f"{schema_version}"
        )

    payload = document["vector"]
    artifact_id = str(document["artifact_id"])
    expected_artifact_id = _artifact_id(
        schema_version=schema_version,
        payload=payload,
    )

    if artifact_id != expected_artifact_id:
        raise ValueError("Feature-vector artifact hash verification failed.")

    vector = _vector_from_payload(payload)

    return FeatureVectorArtifact(
        artifact_id=artifact_id,
        vector=vector,
        schema_version=schema_version,
    )


def _artifact_document(
    artifact: FeatureVectorArtifact,
) -> dict[str, object]:
    return {
        "schema_version": artifact.schema_version,
        "artifact_id": artifact.artifact_id,
        "vector": _vector_payload(artifact.vector),
    }


def _artifact_id(
    *,
    schema_version: int,
    payload: object,
) -> str:
    canonical = json.dumps(
        {
            "schema_version": schema_version,
            "vector": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"feature_vector_{digest}"


def _vector_payload(
    vector: FeatureVector,
) -> dict[str, object]:
    return {
        "match_id": vector.match_id,
        "prediction_time": vector.prediction_time.isoformat(),
        "feature_set_id": vector.feature_set_id,
        "values": [
            {
                "definition": {
                    "name": value.definition.name,
                    "version": value.definition.version,
                    "group": value.definition.group,
                    "description": value.definition.description,
                },
                "status": value.status.value,
                "as_of": value.as_of.isoformat(),
                "value": value.value,
                "missing_reason": (
                    value.missing_reason.value
                    if value.missing_reason is not None
                    else None
                ),
                "imputation_method": value.imputation_method,
                "lineage": {
                    "source_record_ids": list(value.lineage.source_record_ids),
                    "artifact_ids": list(value.lineage.artifact_ids),
                    "model_ids": list(value.lineage.model_ids),
                },
            }
            for value in vector.values
        ],
    }


def _vector_from_payload(
    payload: dict[str, object],
) -> FeatureVector:
    raw_values = payload["values"]
    if not isinstance(raw_values, list):
        raise ValueError("Feature-vector artifact values must be a list.")

    values = tuple(_feature_value_from_payload(item) for item in raw_values)

    return FeatureVector(
        match_id=str(payload["match_id"]),
        prediction_time=datetime.fromisoformat(str(payload["prediction_time"])),
        feature_set_id=str(payload["feature_set_id"]),
        values=values,
    )


def _feature_value_from_payload(
    item: object,
) -> FeatureValue:
    if not isinstance(item, dict):
        raise ValueError("Feature-vector values must be JSON objects.")

    definition_payload = item["definition"]
    lineage_payload = item["lineage"]

    if not isinstance(definition_payload, dict):
        raise ValueError("Feature definition payload must be a JSON object.")
    if not isinstance(lineage_payload, dict):
        raise ValueError("Feature lineage payload must be a JSON object.")

    missing_reason_raw = item.get("missing_reason")

    return FeatureValue(
        definition=FeatureDefinition(
            name=str(definition_payload["name"]),
            version=str(definition_payload["version"]),
            group=str(definition_payload["group"]),
            description=str(definition_payload["description"]),
        ),
        status=FeatureStatus(str(item["status"])),
        as_of=datetime.fromisoformat(str(item["as_of"])),
        value=(
            None
            if item.get("value") is None
            else float(item["value"])
        ),
        missing_reason=(
            None
            if missing_reason_raw is None
            else FeatureMissingReason(str(missing_reason_raw))
        ),
        imputation_method=(
            None
            if item.get("imputation_method") is None
            else str(item["imputation_method"])
        ),
        lineage=FeatureLineage(
            source_record_ids=tuple(
                str(value)
                for value in lineage_payload.get("source_record_ids", [])
            ),
            artifact_ids=tuple(
                str(value)
                for value in lineage_payload.get("artifact_ids", [])
            ),
            model_ids=tuple(
                str(value)
                for value in lineage_payload.get("model_ids", [])
            ),
        ),
    )
