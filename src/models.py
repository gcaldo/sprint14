"""OLS (normal equation) and dummy probability baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.data import BinaryTarget, FeatureMatrix, Target


@dataclass(frozen=True)
class OLSResult:
    """Result of a normal-equation linear regression fit."""

    weights: np.ndarray  # shape (5,)
    predictions: np.ndarray  # shape (n,), raw reals — never truncated
    rmse: float


def _maxabs_features(X: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(X), axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return X / scale


def _design_matrix(X: np.ndarray) -> np.ndarray:
    """DesignMatrix: leading column of ones + feature columns."""
    n = X.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    return np.hstack([ones, X])


def linreg_ols(X: FeatureMatrix, y: Target, scale: bool = False) -> OLSResult:
    """Fit linear regression by the normal equation.

    Predictions are raw reals and are never truncated. If scale=True, MaxAbs
    is applied to the feature block only; the intercept ones column is built
    after scaling and is never scaled itself.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # if scale: MaxAbs on features only
    feats = _maxabs_features(X) if scale else X

    # build DesignMatrix with leading ones
    A = _design_matrix(feats)

    # Normal equation w = (A.T @ A)^{-1} @ A.T @ y, via lstsq for stability.
    # Equivalent to the normal-equation solution when A has full column rank.
    weights, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    predictions = A @ weights  # unrestricted reals — never truncated
    resid = y - predictions
    rmse = float(np.sqrt(np.mean(resid ** 2))) if len(y) else 0.0

    return OLSResult(
        weights=np.asarray(weights, dtype=np.float64),
        predictions=np.asarray(predictions, dtype=np.float64),
        rmse=rmse,
    )


def _f1_and_cm(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """F1 and normalized (by n) confusion-matrix cell rates."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        return {
            "f1": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "tp": 0.0,
        }

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "f1": float(f1),
        "tn": tn / n,
        "fp": fp / n,
        "fn": fn / n,
        "tp": tp / n,
    }


def dummies(
    y_bin: BinaryTarget, seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """F1 and normalized CM summary for four probability baselines.

    Baselines: constant 0, empirical positive rate, 0.5, and 1. Each draw is
    Bernoulli with the given probability (seed governs the RNG).
    """
    y_bin = np.asarray(y_bin, dtype=int)
    n = len(y_bin)
    rate = float(y_bin.mean()) if n else 0.0

    probs = {
        "p_0": 0.0,
        "p_rate": rate,
        "p_0.5": 0.5,
        "p_1": 1.0,
    }

    rng = np.random.RandomState(seed)
    out: Dict[str, Dict[str, float]] = {}
    for name, p in probs.items():
        # four fixed probabilities → binomial / Bernoulli draws
        if n == 0:
            draws = np.empty((0,), dtype=int)
        elif p <= 0.0:
            draws = np.zeros(n, dtype=int)
        elif p >= 1.0:
            draws = np.ones(n, dtype=int)
        else:
            draws = rng.binomial(1, p, size=n)
        out[name] = _f1_and_cm(y_bin, draws)
    return out


def knn_f1_sweep(
    X_train: FeatureMatrix,
    y_train: BinaryTarget,
    X_test: FeatureMatrix,
    y_test: BinaryTarget,
    ks: range = range(1, 11),
) -> list:
    """Pure-NumPy kNN F1 for each k in ks (Euclidean). Helper for the notebook."""
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    # pairwise squared Euclidean: (n_test, n_train)
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    train_sq = np.sum(X_train ** 2, axis=1)
    test_sq = np.sum(X_test ** 2, axis=1)
    cross = X_test @ X_train.T
    dist2 = test_sq[:, None] + train_sq[None, :] - 2.0 * cross

    order = np.argsort(dist2, axis=1, kind="mergesort")
    f1s = []
    for k in ks:
        nn = order[:, :k]
        votes = y_train[nn]
        # majority; ties → 1 if mean >= 0.5
        pred = (votes.mean(axis=1) >= 0.5).astype(int)
        f1s.append(_f1_and_cm(y_test, pred)["f1"])
    return f1s
