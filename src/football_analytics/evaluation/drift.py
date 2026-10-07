"""Descriptive feature drift with reference-only histogram boundaries."""

from collections.abc import Sequence

import numpy as np

from football_analytics.features.materialization import MaterializedFeatureRow


def feature_drift(
    reference: Sequence[MaterializedFeatureRow],
    current: Sequence[MaterializedFeatureRow],
    *,
    bins: int = 10,
) -> dict:
    if not reference or not current or not 2 <= bins <= 100:
        raise ValueError("Drift needs two nonempty samples and 2..100 histogram bins.")
    names = tuple(k for k, _ in reference[0].columns)
    for row in (*reference, *current):
        if (
            tuple(k for k, _ in row.columns) != names
            or row.feature_set_id != reference[0].feature_set_id
            or row.imputation_policy_id != reference[0].imputation_policy_id
        ):
            raise ValueError("Drift samples must share exact feature contracts.")
    left = np.array([[v for _, v in r.columns] for r in reference])
    right = np.array([[v for _, v in r.columns] for r in current])
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Drift inputs must be finite materialized features.")
    rows = []
    for i, name in enumerate(names):
        internal = np.unique(np.quantile(left[:, i], np.linspace(0, 1, bins + 1)[1:-1]))
        edges = np.concatenate(([-np.inf], internal, [np.inf]))
        p = np.histogram(left[:, i], bins=edges)[0] / len(left)
        q = np.histogram(right[:, i], bins=edges)[0] / len(right)
        p, q = np.maximum(p, 1e-6), np.maximum(q, 1e-6)
        p, q = p / p.sum(), q / q.sum()
        rows.append(
            {
                "feature": name,
                "psi": float(np.sum((q - p) * np.log(q / p))),
                "reference_mean": float(left[:, i].mean()),
                "current_mean": float(right[:, i].mean()),
                "reference_min": float(left[:, i].min()),
                "reference_max": float(left[:, i].max()),
                "outside_reference_range": float(
                    np.mean((right[:, i] < left[:, i].min()) | (right[:, i] > left[:, i].max()))
                ),
            }
        )
    return {
        "reference_count": len(left),
        "current_count": len(right),
        "features": rows,
        "method": "reference_quantile_psi_v1",
        "smoothing": 1e-6,
        "note": "Descriptive distribution change, not a calibrated statistical alarm",
    }
