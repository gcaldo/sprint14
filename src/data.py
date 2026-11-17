"""Data loading: sole pandas boundary for Sprint 14."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# A FeatureMatrix is a np.ndarray of shape (n, 4) and dtype float64.
# Columns: 0 gender (0/1), 1 age (integral value as float),
#          2 salary, 3 family_members (integral value as float).
# Insurance benefits are never present.
FeatureMatrix = np.ndarray

# A Target is a np.ndarray of shape (n,) and dtype int.
Target = np.ndarray

# A BinaryTarget is a np.ndarray of shape (n,) and dtype int.
BinaryTarget = np.ndarray

# A ColumnNames is list[str] of length 4.
ColumnNames = List[str]

# A StatsDict is a dict with keys:
#   "age_quantiles", "salary_quantiles", "n", "positive_rate".
StatsDict = Dict[str, object]

COLUMN_NAMES: ColumnNames = ["gender", "age", "salary", "family_members"]
_RENAME = {
    "Gender": "gender",
    "Age": "age",
    "Salary": "salary",
    "Family members": "family_members",
    "Insurance benefits": "insurance_benefits",
}
_QUANTILES = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_arrays(
    path: str,
) -> Tuple[FeatureMatrix, Target, BinaryTarget, ColumnNames, StatsDict]:
    """Load the CSV, enforce age as integer, return pure NumPy arrays + stats.

    No pandas object escapes. Age is integer in the source and stored as
    float64 inside FeatureMatrix.
    """
    # read CSV
    df = pd.read_csv(path)

    # rename / type-check (age → int)
    df = df.rename(columns=_RENAME)
    # tolerate already-snake_case headers
    df = df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))

    if "age" in df.columns and len(df) > 0:
        df["age"] = df["age"].astype(int)

    n = int(len(df))
    if n == 0:
        X = np.empty((0, 4), dtype=np.float64)
        y = np.empty((0,), dtype=int)
        y_bin = np.empty((0,), dtype=int)
        stats: StatsDict = {
            "age_quantiles": np.array([], dtype=np.float64),
            "salary_quantiles": np.array([], dtype=np.float64),
            "n": 0,
            "positive_rate": 0.0,
        }
        return X, y, y_bin, list(COLUMN_NAMES), stats

    # build FeatureMatrix, Target, BinaryTarget, ColumnNames, StatsDict
    X = df[COLUMN_NAMES].to_numpy(dtype=np.float64)
    y = df["insurance_benefits"].to_numpy(dtype=int)
    y_bin = (y > 0).astype(int)

    stats = {
        "age_quantiles": np.quantile(X[:, 1], _QUANTILES),
        "salary_quantiles": np.quantile(X[:, 2], _QUANTILES),
        "n": n,
        "positive_rate": float(y_bin.mean()),
    }
    return X, y, y_bin, list(COLUMN_NAMES), stats
