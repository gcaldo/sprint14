"""Feature-block obfuscation by a random invertible matrix P."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data import FeatureMatrix, Target
from src.models import linreg_ols

SEED = 42


@dataclass(frozen=True)
class ObfuscationResult:
    """Outcome of applying invertible P to the feature block only."""

    P: np.ndarray  # 4×4 invertible
    X_obf: np.ndarray
    X_recovered: np.ndarray
    pred_diff: float
    rmse_diff: float


def _draw_invertible(seed: int = SEED, size: int = 4) -> np.ndarray:
    """Draw InvertibleMatrix P (seed 42) until det != 0."""
    rng = np.random.RandomState(seed)
    while True:
        P = rng.normal(size=(size, size))
        if abs(np.linalg.det(P)) > 1e-8:
            return np.asarray(P, dtype=np.float64)


def obfuscate(X: FeatureMatrix, y: Target) -> ObfuscationResult:
    """Apply random invertible P (seed 42) to the feature block only.

    Proves linear predictions and RMSE are unchanged (up to float noise) and
    demonstrates exact recovery via P^{-1}. The intercept ones column is
    constructed inside linreg_ols and is never multiplied by P.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    # draw InvertibleMatrix P (seed 42)
    P = _draw_invertible(SEED)

    # X_obf = X @ P  (feature block only; no ones column here)
    X_obf = X @ P

    # run linreg_ols on original and obfuscated feature matrices
    ols_orig = linreg_ols(X, y, scale=False)
    ols_obf = linreg_ols(X_obf, y, scale=False)

    # recover via P^{-1}
    X_recovered = X_obf @ np.linalg.inv(P)

    pred_diff = float(np.max(np.abs(ols_orig.predictions - ols_obf.predictions)))
    rmse_diff = float(abs(ols_orig.rmse - ols_obf.rmse))

    return ObfuscationResult(
        P=P,
        X_obf=np.asarray(X_obf, dtype=np.float64),
        X_recovered=np.asarray(X_recovered, dtype=np.float64),
        pred_diff=pred_diff,
        rmse_diff=rmse_diff,
    )
