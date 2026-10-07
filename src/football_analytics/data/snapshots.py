from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class DataSnapshot:
    """Deterministic fingerprint of an in-memory tabular source snapshot."""

    sha256: str
    row_count: int
    column_count: int


def dataframe_snapshot(frame: pd.DataFrame) -> DataSnapshot:
    """Hash a dataframe using stable column ordering and current row ordering."""

    ordered_columns = sorted(str(column) for column in frame.columns)
    canonical = frame.loc[:, ordered_columns].to_csv(
        index=False,
        lineterminator="\n",
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return DataSnapshot(
        sha256=digest,
        row_count=len(frame),
        column_count=len(frame.columns),
    )
