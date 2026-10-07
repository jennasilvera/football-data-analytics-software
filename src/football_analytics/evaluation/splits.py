from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from football_analytics.data.contracts import ensure_utc
from football_analytics.features.dataset import (
    HistoricalFeatureDataset,
    HistoricalFeatureExample,
)


class TemporalFoldSkipReason(StrEnum):
    """Reason a requested temporal fold could not be evaluated."""

    INSUFFICIENT_TRAINING_EXAMPLES = "insufficient_training_examples"
    EMPTY_EVALUATION_WINDOW = "empty_evaluation_window"


@dataclass(frozen=True, slots=True)
class ExpandingWindowPolicy:
    """Versioned expanding-window split policy."""

    policy_id: str
    cutoffs: tuple[datetime, ...]
    evaluation_window: timedelta
    min_train_examples: int = 1

    def __post_init__(self) -> None:
        _validate_common_policy(
            policy_id=self.policy_id,
            cutoffs=self.cutoffs,
            evaluation_window=self.evaluation_window,
            min_train_examples=self.min_train_examples,
        )
        object.__setattr__(self, "policy_id", self.policy_id.strip())
        object.__setattr__(self, "cutoffs", _normalized_cutoffs(self.cutoffs))


@dataclass(frozen=True, slots=True)
class RollingWindowPolicy:
    """Versioned rolling-window split policy with bounded training history."""

    policy_id: str
    cutoffs: tuple[datetime, ...]
    evaluation_window: timedelta
    training_window: timedelta
    min_train_examples: int = 1

    def __post_init__(self) -> None:
        _validate_common_policy(
            policy_id=self.policy_id,
            cutoffs=self.cutoffs,
            evaluation_window=self.evaluation_window,
            min_train_examples=self.min_train_examples,
        )
        if self.training_window <= timedelta(0):
            raise ValueError("training_window must be positive.")

        object.__setattr__(self, "policy_id", self.policy_id.strip())
        object.__setattr__(self, "cutoffs", _normalized_cutoffs(self.cutoffs))


@dataclass(frozen=True, slots=True)
class TemporalBacktestFold:
    """One immutable chronological train/evaluation fold."""

    fold_id: str
    cutoff: datetime
    evaluation_end: datetime
    training_start: datetime | None
    train: tuple[HistoricalFeatureExample, ...]
    test: tuple[HistoricalFeatureExample, ...]

    def __post_init__(self) -> None:
        cutoff = ensure_utc(self.cutoff, "cutoff")
        evaluation_end = ensure_utc(self.evaluation_end, "evaluation_end")
        training_start = (
            ensure_utc(self.training_start, "training_start")
            if self.training_start is not None
            else None
        )

        if not self.fold_id.strip():
            raise ValueError("fold_id must not be blank.")
        if evaluation_end <= cutoff:
            raise ValueError("evaluation_end must be after cutoff.")
        if training_start is not None and training_start >= cutoff:
            raise ValueError("training_start must be before cutoff.")
        if not self.train or not self.test:
            raise ValueError("Backtest folds require non-empty train and test sets.")

        train_ids = {example.match_id for example in self.train}
        test_ids = {example.match_id for example in self.test}
        overlap = train_ids.intersection(test_ids)
        if overlap:
            raise ValueError(
                f"Backtest fold has train/test match overlap: {sorted(overlap)}"
            )

        if any(example.prediction_time >= cutoff for example in self.train):
            raise ValueError("Training examples must be strictly before cutoff.")
        if training_start is not None and any(
            example.prediction_time < training_start for example in self.train
        ):
            raise ValueError("Training example precedes rolling training_start.")
        if any(
            example.prediction_time < cutoff
            or example.prediction_time >= evaluation_end
            for example in self.test
        ):
            raise ValueError("Test examples must lie inside the evaluation window.")

        object.__setattr__(self, "fold_id", self.fold_id.strip())
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "evaluation_end", evaluation_end)
        object.__setattr__(self, "training_start", training_start)


@dataclass(frozen=True, slots=True)
class SkippedTemporalFold:
    """Audit record for a requested cutoff that could not form a valid fold."""

    fold_id: str
    cutoff: datetime
    reason: TemporalFoldSkipReason
    train_count: int
    test_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff", ensure_utc(self.cutoff, "cutoff"))


@dataclass(frozen=True, slots=True)
class TemporalFoldBuildReport:
    """Valid and skipped temporal folds tied to one historical dataset contract."""

    policy_id: str
    source_feature_set_id: str
    source_cutoff_policy_id: str
    folds: tuple[TemporalBacktestFold, ...]
    skipped: tuple[SkippedTemporalFold, ...]

    def __post_init__(self) -> None:
        policy_id = self.policy_id.strip()
        feature_set_id = self.source_feature_set_id.strip()
        cutoff_policy_id = self.source_cutoff_policy_id.strip()

        if not policy_id:
            raise ValueError("policy_id must not be blank.")
        if not feature_set_id:
            raise ValueError("source_feature_set_id must not be blank.")
        if not cutoff_policy_id:
            raise ValueError("source_cutoff_policy_id must not be blank.")

        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "source_feature_set_id", feature_set_id)
        object.__setattr__(self, "source_cutoff_policy_id", cutoff_policy_id)

    @property
    def requested_fold_count(self) -> int:
        return len(self.folds) + len(self.skipped)


def build_expanding_window_folds(
    dataset: HistoricalFeatureDataset,
    policy: ExpandingWindowPolicy,
) -> TemporalFoldBuildReport:
    """Build expanding-window folds from point-in-time historical examples."""

    return _build_folds(
        dataset=dataset,
        policy_id=policy.policy_id,
        cutoffs=policy.cutoffs,
        evaluation_window=policy.evaluation_window,
        training_window=None,
        min_train_examples=policy.min_train_examples,
    )


def build_rolling_window_folds(
    dataset: HistoricalFeatureDataset,
    policy: RollingWindowPolicy,
) -> TemporalFoldBuildReport:
    """Build bounded rolling-window folds from historical examples."""

    return _build_folds(
        dataset=dataset,
        policy_id=policy.policy_id,
        cutoffs=policy.cutoffs,
        evaluation_window=policy.evaluation_window,
        training_window=policy.training_window,
        min_train_examples=policy.min_train_examples,
    )


def _build_folds(
    *,
    dataset: HistoricalFeatureDataset,
    policy_id: str,
    cutoffs: tuple[datetime, ...],
    evaluation_window: timedelta,
    training_window: timedelta | None,
    min_train_examples: int,
) -> TemporalFoldBuildReport:
    examples = tuple(
        sorted(
            dataset.examples,
            key=lambda example: (
                example.prediction_time,
                example.match_id,
            ),
        )
    )
    folds: list[TemporalBacktestFold] = []
    skipped: list[SkippedTemporalFold] = []

    for cutoff in cutoffs:
        evaluation_end = cutoff + evaluation_window
        training_start = (
            cutoff - training_window
            if training_window is not None
            else None
        )

        train = tuple(
            example
            for example in examples
            if example.prediction_time < cutoff
            and (
                training_start is None
                or example.prediction_time >= training_start
            )
        )
        test = tuple(
            example
            for example in examples
            if cutoff <= example.prediction_time < evaluation_end
        )
        fold_id = f"{policy_id}:{cutoff.isoformat()}"

        if len(train) < min_train_examples:
            skipped.append(
                SkippedTemporalFold(
                    fold_id=fold_id,
                    cutoff=cutoff,
                    reason=TemporalFoldSkipReason.INSUFFICIENT_TRAINING_EXAMPLES,
                    train_count=len(train),
                    test_count=len(test),
                )
            )
            continue

        if not test:
            skipped.append(
                SkippedTemporalFold(
                    fold_id=fold_id,
                    cutoff=cutoff,
                    reason=TemporalFoldSkipReason.EMPTY_EVALUATION_WINDOW,
                    train_count=len(train),
                    test_count=0,
                )
            )
            continue

        folds.append(
            TemporalBacktestFold(
                fold_id=fold_id,
                cutoff=cutoff,
                evaluation_end=evaluation_end,
                training_start=training_start,
                train=train,
                test=test,
            )
        )

    return TemporalFoldBuildReport(
        policy_id=policy_id,
        source_feature_set_id=dataset.feature_set_id,
        source_cutoff_policy_id=dataset.cutoff_policy_id,
        folds=tuple(folds),
        skipped=tuple(skipped),
    )


def _validate_common_policy(
    *,
    policy_id: str,
    cutoffs: tuple[datetime, ...],
    evaluation_window: timedelta,
    min_train_examples: int,
) -> None:
    if not policy_id.strip():
        raise ValueError("policy_id must not be blank.")
    if not cutoffs:
        raise ValueError("At least one cutoff is required.")
    if evaluation_window <= timedelta(0):
        raise ValueError("evaluation_window must be positive.")
    if min_train_examples <= 0:
        raise ValueError("min_train_examples must be positive.")

    normalized = tuple(ensure_utc(cutoff, "cutoff") for cutoff in cutoffs)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Temporal split cutoffs must be unique.")


def _normalized_cutoffs(
    cutoffs: tuple[datetime, ...],
) -> tuple[datetime, ...]:
    return tuple(
        sorted(ensure_utc(cutoff, "cutoff") for cutoff in cutoffs)
    )
