"""Training-only competition frequencies shrunk toward the global training sample."""

from __future__ import annotations

import math
from dataclasses import dataclass

from football_analytics.domain import Competition, MatchOutcome
from football_analytics.domain.probabilities import OutcomeProbabilities
from football_analytics.features.base import (
    FeatureDefinition,
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
from football_analytics.models.dataset import ModelDataset
from football_analytics.models.postprocessing import content_id

GROUP_COLUMN = "baseline.competition_code"
OUTCOMES = (MatchOutcome.HOME_WIN, MatchOutcome.DRAW, MatchOutcome.AWAY_WIN)


class CompetitionIdentityProvider:
    """Catalog identity only; no outcomes, future statistics, or numeric distance meaning."""

    provider_id = "competition_identity_v1"

    def __init__(self, competitions: tuple[Competition, ...]):
        names = sorted(c.competition_id for c in competitions)
        if not names or len(set(names)) != len(names):
            raise ValueError("Competition identities must be nonempty and unique.")
        self.codes = {name: i for i, name in enumerate(names)}
        self.definition = FeatureDefinition(
            GROUP_COLUMN,
            content_id("competition_catalog_", self.codes),
            "competition_identity",
            "Nominal canonical competition code; exact grouping only, not an ordinal feature",
        )

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return (self.definition,)

    def compute(self, context: PredictionContext) -> tuple[FeatureValue, ...]:
        if context.match.competition_id not in self.codes:
            raise ValueError("Unknown canonical competition.")
        return (
            FeatureValue(
                self.definition,
                FeatureStatus.OBSERVED,
                context.prediction_time,
                float(self.codes[context.match.competition_id]),
            ),
        )


def competition_frequency_spec(
    *, smoothing: float = 1.0, prior_strength: float = 10.0
) -> ModelTrainingSpec:
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Global smoothing must be finite and positive.")
    if not math.isfinite(prior_strength) or prior_strength < 0:
        raise ValueError("Competition prior strength must be finite and nonnegative.")
    return ModelTrainingSpec(
        "competition_frequency_v1",
        ModelFamily.COMPETITION_FREQUENCY,
        "v1",
        parameters=(("smoothing", float(smoothing)), ("prior_strength", float(prior_strength))),
    )


def competition_code(row: MaterializedFeatureRow) -> int:
    values = row.as_dict()
    value = values[GROUP_COLUMN]
    if not math.isfinite(value) or value < 0 or not float(value).is_integer():
        raise ValueError("Competition code must be a nonnegative integer.")
    if any(
        values.get(GROUP_COLUMN + suffix, 0) != 0 for suffix in ("__is_missing", "__is_imputed")
    ):
        raise ValueError("Competition identity cannot be missing or imputed.")
    return int(value)


@dataclass(frozen=True, slots=True)
class CompetitionFrequencyModel:
    model_id: str
    feature_names: tuple[str, ...]
    feature_set_id: str
    imputation_policy_id: str
    global_probabilities: OutcomeProbabilities
    groups: tuple[tuple[int, OutcomeProbabilities], ...]

    def predict_row(self, row: MaterializedFeatureRow) -> OutcomeProbabilities:
        if (
            row.feature_set_id != self.feature_set_id
            or row.imputation_policy_id != self.imputation_policy_id
            or tuple(k for k, _ in row.columns) != self.feature_names
        ):
            raise ValueError("Competition-frequency feature contract mismatch.")
        return dict(self.groups).get(competition_code(row), self.global_probabilities)


def train_competition_frequency(
    dataset: ModelDataset, spec: ModelTrainingSpec
) -> ModelTrainingResult:
    parameters = spec.parameter_dict()
    if spec.family is not ModelFamily.COMPETITION_FREQUENCY or set(parameters) != {
        "smoothing",
        "prior_strength",
    }:
        raise ValueError("Expected a competition-frequency specification.")
    smoothing, prior = float(parameters["smoothing"]), float(parameters["prior_strength"])
    competition_frequency_spec(smoothing=smoothing, prior_strength=prior)
    if GROUP_COLUMN not in dataset.column_names:
        raise ValueError("Competition-frequency training requires canonical competition codes.")
    global_p = OutcomeProbabilities(
        *(
            (dataset.targets.count(outcome) + smoothing) / (len(dataset.examples) + 3 * smoothing)
            for outcome in OUTCOMES
        )
    )
    counts: dict[int, list[int]] = {}
    for example in dataset.examples:
        code = competition_code(example.row)
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
        CompetitionFrequencyModel(
            metadata.model_id,
            dataset.column_names,
            dataset.feature_set_id,
            dataset.imputation_policy_id,
            global_p,
            groups,
        ),
        metadata,
    )
