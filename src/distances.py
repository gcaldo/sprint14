"""Nearest-neighbor distances for four locked metric × scaling scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.data import FeatureMatrix


@dataclass(frozen=True)
class NeighborResult:
    """One neighbor of a query under a distance scenario."""

    index: int
    features: np.ndarray  # shape (4,), original row
    distance: float
    shares: np.ndarray  # shape (4,), sum to 1


# A Query is a np.ndarray of shape (4,) and dtype float64.
Query = np.ndarray

_KEYS = (
    "euclidean_raw",
    "euclidean_scaled",
    "manhattan_raw",
    "manhattan_scaled",
)


def _maxabs_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit MaxAbs on rows of X; return scaled copy and per-column scales."""
    scale = np.max(np.abs(X), axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return X / scale, scale


def _contribution_shares(delta: np.ndarray, metric: str) -> np.ndarray:
    """Relative contribution of each feature to the distance.

    Manhattan = |Δᵢ| / Σ|Δⱼ|; Euclidean = (Δᵢ)² / Σ(Δⱼ)².
    """
    if metric == "euclidean":
        pieces = delta ** 2
    else:
        pieces = np.abs(delta)
    total = pieces.sum()
    if total == 0.0:
        return np.zeros_like(pieces, dtype=np.float64)
    return pieces / total


def _k_neighbors(
    X: FeatureMatrix,
    X_space: np.ndarray,
    query_space: np.ndarray,
    k: int,
    metric: str,
) -> List[NeighborResult]:
    """Select k nearest neighbors of one query in the given feature space."""
    if metric == "euclidean":
        dists = np.sqrt(np.sum((X_space - query_space) ** 2, axis=1))
    else:
        dists = np.sum(np.abs(X_space - query_space), axis=1)

    n = X_space.shape[0]
    if n == 0:
        return []
    kk = min(k, n)
    # argpartition then sort the shortlist for stable nearest order
    part = np.argpartition(dists, kk - 1)[:kk]
    order = part[np.argsort(dists[part], kind="mergesort")]

    results: List[NeighborResult] = []
    for idx in order:
        i = int(idx)
        delta = X_space[i] - query_space
        shares = _contribution_shares(delta, metric)
        results.append(
            NeighborResult(
                index=i,
                features=np.asarray(X[i], dtype=np.float64).copy(),
                distance=float(dists[i]),
                shares=np.asarray(shares, dtype=np.float64),
            )
        )
    return results


def distances(
    X: FeatureMatrix,
    queries: List[Query],
    k: int = 5,
) -> Dict[str, List[List[NeighborResult]]]:
    """For each of four scenarios return k nearest neighbors of every query.

    Keys: euclidean_raw, euclidean_scaled, manhattan_raw, manhattan_scaled.
    Queries are caller-supplied; never invented here. Neighbor features are
    original rows; shares sum to 1.
    """
    X = np.asarray(X, dtype=np.float64)
    X_scaled, scale = _maxabs_scale(X)

    out: Dict[str, List[List[NeighborResult]]] = {key: [] for key in _KEYS}

    for q in queries:
        q = np.asarray(q, dtype=np.float64).reshape(4,)
        q_scaled = q / scale

        out["euclidean_raw"].append(
            _k_neighbors(X, X, q, k, "euclidean")
        )
        out["euclidean_scaled"].append(
            _k_neighbors(X, X_scaled, q_scaled, k, "euclidean")
        )
        out["manhattan_raw"].append(
            _k_neighbors(X, X, q, k, "manhattan")
        )
        out["manhattan_scaled"].append(
            _k_neighbors(X, X_scaled, q_scaled, k, "manhattan")
        )

    return out
