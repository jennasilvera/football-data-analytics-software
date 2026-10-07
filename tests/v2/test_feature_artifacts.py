from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from football_analytics.features import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    FeatureVector,
    build_feature_vector_artifact,
    feature_set_id_for_definitions,
    load_feature_vector_artifact,
    save_feature_vector_artifact,
)


def _vector() -> FeatureVector:
    definition = FeatureDefinition(
        name="elo.home_rating",
        version="legacy_elo_v1",
        group="team_strength",
        description="Home rating at prediction cutoff.",
    )
    prediction_time = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    return FeatureVector(
        match_id="match-1",
        prediction_time=prediction_time,
        feature_set_id=feature_set_id_for_definitions([definition]),
        values=(
            FeatureValue(
                definition=definition,
                status=FeatureStatus.IMPUTED,
                as_of=prediction_time,
                value=1500.0,
                missing_reason=FeatureMissingReason.NO_HISTORY,
                imputation_method="legacy_elo_v1:default_rating=1500",
                lineage=FeatureLineage(model_ids=("legacy_elo_v1",)),
            ),
        ),
    )


def test_feature_vector_artifact_id_is_deterministic() -> None:
    first = build_feature_vector_artifact(_vector())
    second = build_feature_vector_artifact(_vector())

    assert first.artifact_id == second.artifact_id
    assert first.artifact_id.startswith("feature_vector_")


def test_feature_vector_artifact_round_trips_with_lineage(tmp_path) -> None:
    path = tmp_path / "feature-vector.json"
    saved = save_feature_vector_artifact(_vector(), path)
    loaded = load_feature_vector_artifact(path)

    assert loaded.artifact_id == saved.artifact_id
    assert loaded.vector == _vector()
    assert loaded.vector.values[0].lineage.model_ids == ("legacy_elo_v1",)


def test_feature_vector_artifact_detects_tampering(tmp_path) -> None:
    path = tmp_path / "feature-vector.json"
    save_feature_vector_artifact(_vector(), path)

    document = json.loads(path.read_text(encoding="utf-8"))
    document["vector"]["values"][0]["value"] = 1900.0
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="hash verification failed"):
        load_feature_vector_artifact(path)
