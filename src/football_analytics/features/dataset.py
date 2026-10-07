from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

from football_analytics.data.contracts import ensure_utc
from football_analytics.data.normalization import CanonicalMatchRecord
from football_analytics.domain import Match, MatchOutcome, MatchStatus
from football_analytics.domain.scores import require_regulation_score
from football_analytics.features.base import (
    FeatureProvider,
    FeatureVector,
    PredictionContext,
    build_feature_vector,
)
from football_analytics.features.history import (
    DEFAULT_RESULT_ELIGIBILITY_POLICY,
    CompletedResultEligibilityPolicy,
    ResultEligibilityBasis,
    completed_record_is_before_cutoff,
)


class HistoricalFeatureLeakageError(ValueError):
    """Raised when historical feature lineage crosses a prediction cutoff."""


@dataclass(frozen=True, slots=True)
class PredictionCutoffPolicy:
    """Versioned policy for assigning historical forecast cutoffs."""

    policy_id: str
    exact_kickoff_lead: timedelta = timedelta(0)
    date_only_lead_days: int = 1

    def __post_init__(self) -> None:
        policy_id = self.policy_id.strip()

        if not policy_id:
            raise ValueError("policy_id must not be blank.")
        if self.exact_kickoff_lead < timedelta(0):
            raise ValueError("exact_kickoff_lead cannot be negative.")
        if self.date_only_lead_days < 1:
            raise ValueError("date_only_lead_days must be at least 1.")

        object.__setattr__(self, "policy_id", policy_id)

    def cutoff_for(self, match: Match) -> datetime:
        """Return the point-in-time cutoff for one historical match."""

        if match.kickoff_at is not None:
            return match.kickoff_at - self.exact_kickoff_lead

        cutoff_date = match.match_date - timedelta(days=self.date_only_lead_days)
        return datetime.combine(cutoff_date, time.max, tzinfo=UTC)


DEFAULT_HISTORICAL_CUTOFF_POLICY = PredictionCutoffPolicy(
    policy_id="pre_match_default_v1",
)


@dataclass(frozen=True, slots=True)
class HistoricalFeatureExample:
    """One supervised example with separate feature and target availability times."""

    match_id: str
    source_match_id: str
    prediction_time: datetime
    target_available_at: datetime
    target_availability_basis: ResultEligibilityBasis
    target: MatchOutcome
    vector: FeatureVector

    def __post_init__(self) -> None:
        match_id = self.match_id.strip()
        source_match_id = self.source_match_id.strip()
        prediction_time = ensure_utc(self.prediction_time, "prediction_time")
        target_available_at = ensure_utc(
            self.target_available_at,
            "target_available_at",
        )

        if not match_id:
            raise ValueError("Historical example match_id must not be blank.")
        if not source_match_id:
            raise ValueError("Historical example source_match_id must not be blank.")
        if target_available_at <= prediction_time:
            raise ValueError(
                "Historical target must become available after prediction_time."
            )
        if self.vector.match_id != match_id:
            raise ValueError("Historical example match_id does not match feature vector.")
        if self.vector.prediction_time != prediction_time:
            raise ValueError(
                "Historical example prediction_time does not match feature vector."
            )

        object.__setattr__(self, "match_id", match_id)
        object.__setattr__(self, "source_match_id", source_match_id)
        object.__setattr__(self, "prediction_time", prediction_time)
        object.__setattr__(self, "target_available_at", target_available_at)


@dataclass(frozen=True, slots=True)
class HistoricalFeatureDataset:
    """Immutable historical examples sharing temporal and feature policies."""

    feature_set_id: str
    cutoff_policy_id: str
    result_eligibility_policy_id: str
    examples: tuple[HistoricalFeatureExample, ...]

    def __post_init__(self) -> None:
        feature_set_id = self.feature_set_id.strip()
        cutoff_policy_id = self.cutoff_policy_id.strip()
        result_policy_id = self.result_eligibility_policy_id.strip()

        if not feature_set_id:
            raise ValueError("feature_set_id must not be blank.")
        if not cutoff_policy_id:
            raise ValueError("cutoff_policy_id must not be blank.")
        if not result_policy_id:
            raise ValueError("result_eligibility_policy_id must not be blank.")
        if not self.examples:
            raise ValueError("Historical feature dataset must not be empty.")

        match_ids = [example.match_id for example in self.examples]
        if len(match_ids) != len(set(match_ids)):
            raise ValueError("Historical feature dataset contains duplicate match IDs.")

        for example in self.examples:
            if example.vector.feature_set_id != feature_set_id:
                raise ValueError(
                    "Historical feature dataset contains mixed feature-set IDs."
                )

        object.__setattr__(self, "feature_set_id", feature_set_id)
        object.__setattr__(self, "cutoff_policy_id", cutoff_policy_id)
        object.__setattr__(
            self,
            "result_eligibility_policy_id",
            result_policy_id,
        )


def build_historical_feature_dataset(
    records: Sequence[CanonicalMatchRecord],
    *,
    providers: Sequence[FeatureProvider],
    cutoff_policy: PredictionCutoffPolicy = DEFAULT_HISTORICAL_CUTOFF_POLICY,
    result_eligibility_policy: CompletedResultEligibilityPolicy = (
        DEFAULT_RESULT_ELIGIBILITY_POLICY
    ),
) -> HistoricalFeatureDataset:
    """Build point-in-time examples without exposing targets before eligibility."""

    completed = [
        record
        for record in records
        if record.match.status is MatchStatus.COMPLETED
        and record.home_score is not None
        and record.away_score is not None
    ]

    if not completed:
        raise ValueError("No completed scored matches are available for the dataset.")

    completed.sort(key=_record_sort_key)
    records_by_source_id = {
        record.source_match_id: record
        for record in records
    }
    records_by_match_id = {
        record.match.match_id: record
        for record in records
    }

    examples: list[HistoricalFeatureExample] = []

    for record in completed:
        require_regulation_score(record.score_basis, match_id=record.match.match_id)
        prediction_time = cutoff_policy.cutoff_for(record.match)
        target_eligibility = result_eligibility_policy.eligibility_for(record)
        context = PredictionContext(
            match=record.match,
            prediction_time=prediction_time,
        )
        vector = build_feature_vector(context, providers)

        _assert_lineage_before_cutoff(
            vector=vector,
            target=record,
            records_by_source_id=records_by_source_id,
            records_by_match_id=records_by_match_id,
            result_eligibility_policy=result_eligibility_policy,
        )

        home_score = record.home_score
        away_score = record.away_score
        assert home_score is not None
        assert away_score is not None

        examples.append(
            HistoricalFeatureExample(
                match_id=record.match.match_id,
                source_match_id=record.source_match_id,
                prediction_time=prediction_time,
                target_available_at=target_eligibility.eligible_at,
                target_availability_basis=target_eligibility.basis,
                target=_outcome(home_score, away_score),
                vector=vector,
            )
        )

    feature_set_ids = {
        example.vector.feature_set_id
        for example in examples
    }
    if len(feature_set_ids) != 1:
        raise ValueError(
            "Historical feature providers produced inconsistent feature-set IDs."
        )

    return HistoricalFeatureDataset(
        feature_set_id=next(iter(feature_set_ids)),
        cutoff_policy_id=cutoff_policy.policy_id,
        result_eligibility_policy_id=result_eligibility_policy.policy_id,
        examples=tuple(examples),
    )


def _assert_lineage_before_cutoff(
    *,
    vector: FeatureVector,
    target: CanonicalMatchRecord,
    records_by_source_id: dict[str, CanonicalMatchRecord],
    records_by_match_id: dict[str, CanonicalMatchRecord],
    result_eligibility_policy: CompletedResultEligibilityPolicy,
) -> None:
    cutoff = vector.prediction_time

    for feature in vector.values:
        for source_record_id in feature.lineage.source_record_ids:
            linked = records_by_source_id.get(source_record_id)
            if linked is not None and not completed_record_is_before_cutoff(
                linked,
                cutoff,
                policy=result_eligibility_policy,
            ):
                raise HistoricalFeatureLeakageError(
                    f"Feature {feature.definition.name} references source record "
                    f"{source_record_id!r} that is not eligible before the cutoff for "
                    f"{target.match.match_id!r}."
                )

        for artifact_id in feature.lineage.artifact_ids:
            linked = records_by_match_id.get(artifact_id)
            if linked is not None and not completed_record_is_before_cutoff(
                linked,
                cutoff,
                policy=result_eligibility_policy,
            ):
                raise HistoricalFeatureLeakageError(
                    f"Feature {feature.definition.name} references match artifact "
                    f"{artifact_id!r} that is not eligible before the cutoff for "
                    f"{target.match.match_id!r}."
                )


def _outcome(home_score: int, away_score: int) -> MatchOutcome:
    if home_score > away_score:
        return MatchOutcome.HOME_WIN
    if home_score < away_score:
        return MatchOutcome.AWAY_WIN
    return MatchOutcome.DRAW


def _record_sort_key(record: CanonicalMatchRecord) -> tuple[object, ...]:
    kickoff = record.match.kickoff_at

    return (
        record.match.match_date,
        kickoff is None,
        kickoff.isoformat() if kickoff is not None else "",
        record.match.match_id,
    )
