"""Historical ordered-confederation-pair baseline with explicit global fallback."""

from __future__ import annotations

import math

from football_analytics.data.confederations import MembershipHistory
from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.domain.teams import Confederation
from football_analytics.features.base import (
    FeatureDefinition,
    FeatureLineage,
    FeatureMissingReason,
    FeatureStatus,
    FeatureValue,
    PredictionContext,
)
from football_analytics.features.materialization import MaterializedFeatureRow
from football_analytics.models.base import (
    ModelFamily,
    ModelTrainingResult,
    ModelTrainingSpec,
    training_metadata_for,
)
from football_analytics.models.competition_frequency import (
    OUTCOMES,
    CompetitionFrequencyModel,
    competition_frequency_spec,
)
from football_analytics.models.dataset import ModelDataset
from football_analytics.models.postprocessing import content_id

PAIR_COLUMN = "baseline.confederation_pair_code"
CONFEDERATIONS = tuple(sorted(c.value for c in Confederation))
PAIRS = tuple((home, away) for home in CONFEDERATIONS for away in CONFEDERATIONS)


class ConfederationPairProvider:
    provider_id = "historical_confederation_pair_v1"

    def __init__(self, history: MembershipHistory):
        self.history = history
        self.definition = FeatureDefinition(
            PAIR_COLUMN,
            content_id("confederation_pairs_", PAIRS),
            "confederation_identity",
            "Ordered home/away historical confederation pair; zero means global fallback",
        )

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return (self.definition,)

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        match = context.match
        home, home_release = self.history.resolve(
            match.home_team_id, match.match_date, context.prediction_time
        )
        away, away_release = self.history.resolve(
            match.away_team_id, match.match_date, context.prediction_time
        )
        known = home is not None and away is not None
        code = PAIRS.index((home.value, away.value)) + 1 if home and away else 0
        return (
            FeatureValue(
                self.definition,
                FeatureStatus.OBSERVED if known else FeatureStatus.IMPUTED,
                context.prediction_time,
                float(code),
                missing_reason=None if known else FeatureMissingReason.UPSTREAM_UNAVAILABLE,
                imputation_method=None if known else "unknown_membership_global_fallback",
                lineage=FeatureLineage(
                    artifact_ids=tuple(
                        sorted(set(r for r in (home_release, away_release) if r is not None))
                    )
                ),
            ),
        )


def confederation_frequency_spec(
    *, smoothing: float = 1.0, prior_strength: float = 10.0
) -> ModelTrainingSpec:
    validated = competition_frequency_spec(smoothing=smoothing, prior_strength=prior_strength)
    return ModelTrainingSpec(
        "confederation_frequency_v1",
        ModelFamily.CONFEDERATION_FREQUENCY,
        "v1",
        parameters=validated.parameters,
    )


def pair_code(row: MaterializedFeatureRow) -> int:
    values = row.as_dict()
    value = values[PAIR_COLUMN]
    if not math.isfinite(value) or not float(value).is_integer() or not 0 <= value <= len(PAIRS):
        raise ValueError("Confederation pair code must be an integer in [0, 36].")
    if value and any(
        values.get(PAIR_COLUMN + suffix, 0) != 0 for suffix in ("__is_missing", "__is_imputed")
    ):
        raise ValueError("A known confederation pair cannot be missing or imputed.")
    return int(value)


class ConfederationFrequencyModel(CompetitionFrequencyModel):
    def predict_row(self, row: MaterializedFeatureRow) -> OutcomeProbabilities:
        if (
            row.feature_set_id != self.feature_set_id
            or row.imputation_policy_id != self.imputation_policy_id
            or tuple(k for k, _ in row.columns) != self.feature_names
        ):
            raise ValueError("Confederation-frequency feature contract mismatch.")
        return dict(self.groups).get(pair_code(row), self.global_probabilities)


def train_confederation_frequency(
    dataset: ModelDataset, spec: ModelTrainingSpec
) -> ModelTrainingResult:
    parameters = spec.parameter_dict()
    if spec.family is not ModelFamily.CONFEDERATION_FREQUENCY or set(parameters) != {
        "smoothing",
        "prior_strength",
    }:
        raise ValueError("Expected a confederation-frequency specification.")
    smoothing, prior = float(parameters["smoothing"]), float(parameters["prior_strength"])
    confederation_frequency_spec(smoothing=smoothing, prior_strength=prior)
    if PAIR_COLUMN not in dataset.column_names:
        raise ValueError("Confederation baseline requires historical membership pair codes.")
    global_p = OutcomeProbabilities(
        *(
            (dataset.targets.count(outcome) + smoothing) / (len(dataset.examples) + 3 * smoothing)
            for outcome in OUTCOMES
        )
    )
    counts: dict[int, list[int]] = {}
    for example in dataset.examples:
        code = pair_code(example.row)
        if code:  # Unknown membership contributes to global counts, never its own fitted group.
            counts.setdefault(code, [0, 0, 0])[OUTCOMES.index(example.target)] += 1
    groups = tuple(
        (
            code,
            OutcomeProbabilities(
                *(
                    (count + prior * probability) / (sum(values) + prior)
                    for count, probability in zip(values, global_p.as_tuple(), strict=True)
                )
            ),
        )
        for code, values in sorted(counts.items())
    )
    metadata = training_metadata_for(dataset, spec)
    return ModelTrainingResult(
        ConfederationFrequencyModel(
            metadata.model_id,
            dataset.column_names,
            dataset.feature_set_id,
            dataset.imputation_policy_id,
            global_p,
            groups,
        ),
        metadata,
    )
