"""Smoke checks for the functional examples in design.md."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.data import load_arrays
from src.distances import distances
from src.diagnostics import f1_vs_k
from src.models import dummies, linreg_ols
from src.obfuscation import obfuscate


def _write_csv(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def test_load_arrays_mini_csv_shapes_and_rate():
    # 3-row mini CSV → correct shapes, y_bin, quantiles, positive_rate = 1/3.
    path = _write_csv(
        "Gender,Age,Salary,Family members,Insurance benefits\n"
        "1,41.0,49600.0,1,0\n"
        "0,46.0,38000.0,1,1\n"
        "0,29.0,21000.0,0,0\n"
    )
    try:
        X, y, y_bin, cols, stats = load_arrays(path)
        assert X.shape == (3, 4)
        assert X.dtype == np.float64
        assert y.shape == (3,)
        assert y.dtype == int or y.dtype == np.int64 or y.dtype == np.int32
        assert y_bin.tolist() == [0, 1, 0]
        assert cols == ["gender", "age", "salary", "family_members"]
        assert stats["n"] == 3
        assert abs(stats["positive_rate"] - 1.0 / 3.0) < 1e-12
        assert len(stats["age_quantiles"]) == 5
        assert len(stats["salary_quantiles"]) == 5
        # Age written as float in CSV → stored as integral float.
        assert X[0, 1] == 41.0
        assert float(X[0, 1]).is_integer()
    finally:
        os.remove(path)


def test_load_arrays_header_only():
    # Header-only file → n = 0, positive_rate = 0.0.
    path = _write_csv(
        "Gender,Age,Salary,Family members,Insurance benefits\n"
    )
    try:
        X, y, y_bin, cols, stats = load_arrays(path)
        assert X.shape == (0, 4)
        assert y.shape == (0,)
        assert y_bin.shape == (0,)
        assert stats["n"] == 0
        assert stats["positive_rate"] == 0.0
    finally:
        os.remove(path)


def test_distances_four_keys_and_shares():
    # Tiny 4-row matrix + two queries (one close, one salary-dominance).
    X = np.array(
        [
            [0.0, 30.0, 40000.0, 1.0],
            [1.0, 31.0, 40100.0, 1.0],  # close to query0
            [0.0, 50.0, 80000.0, 3.0],
            [1.0, 25.0, 20000.0, 0.0],
        ],
        dtype=np.float64,
    )
    q_close = np.array([1.0, 31.0, 40100.0, 1.0], dtype=np.float64)
    q_salary = np.array([0.0, 30.0, 90000.0, 1.0], dtype=np.float64)

    out = distances(X, [q_close, q_salary], k=3)
    expected_keys = {
        "euclidean_raw",
        "euclidean_scaled",
        "manhattan_raw",
        "manhattan_scaled",
    }
    assert set(out.keys()) == expected_keys
    for key, per_query in out.items():
        assert len(per_query) == 2
        for neighbors in per_query:
            assert len(neighbors) == 3
            for nb in neighbors:
                assert nb.features.shape == (4,)
                assert nb.shares.shape == (4,)
                assert abs(nb.shares.sum() - 1.0) < 1e-9 or nb.distance == 0.0
                if nb.distance == 0.0:
                    assert np.allclose(nb.shares, 0.0) or abs(nb.shares.sum() - 1.0) < 1e-9


def test_linreg_ols_shapes_and_untruncated():
    # Tiny X, y → weights shape (5,), predictions shape (n,), rmse ≥ 0.
    # n > 5 so the design matrix (ones + 4 features) is overdetermined.
    rng = np.random.RandomState(0)
    X = np.column_stack(
        [
            rng.randint(0, 2, size=12).astype(np.float64),
            rng.randint(20, 60, size=12).astype(np.float64),
            rng.uniform(20000, 70000, size=12),
            rng.randint(0, 5, size=12).astype(np.float64),
        ]
    )
    y = rng.randint(0, 3, size=12)

    raw = linreg_ols(X, y, scale=False)
    assert raw.weights.shape == (5,)
    assert raw.predictions.shape == (12,)
    assert raw.rmse >= 0.0
    # predictions remain untruncated (may leave {0,1,2,...})
    assert raw.predictions.dtype == np.float64

    scaled = linreg_ols(X, y, scale=True)
    assert not np.allclose(raw.weights, scaled.weights)
    assert scaled.predictions.shape == (12,)
    # still untruncated
    assert scaled.predictions.dtype == np.float64


def test_obfuscate_identity_and_recovery():
    rng = np.random.RandomState(1)
    X = np.column_stack(
        [
            rng.randint(0, 2, size=20).astype(np.float64),
            rng.randint(20, 60, size=20).astype(np.float64),
            rng.uniform(20000, 70000, size=20),
            rng.randint(0, 5, size=20).astype(np.float64),
        ]
    )
    y = rng.randint(0, 3, size=20)

    result = obfuscate(X, y)
    assert result.P.shape == (4, 4)
    assert abs(np.linalg.det(result.P)) > 1e-8
    assert np.allclose(result.X_recovered, X, atol=1e-8)
    assert result.pred_diff < 1e-8
    assert result.rmse_diff < 1e-8
    # ones column never multiplied by P: X_obf is features-only @ P
    assert result.X_obf.shape == X.shape
    assert np.allclose(result.X_obf, X @ result.P)


def test_dummies_four_baselines():
    y_bin = np.array([0, 0, 0, 1, 1], dtype=int)
    out = dummies(y_bin, seed=42)
    assert set(out.keys()) == {"p_0", "p_rate", "p_0.5", "p_1"}
    for name, metrics in out.items():
        assert "f1" in metrics
        assert set(metrics.keys()) >= {"f1", "tn", "fp", "fn", "tp"}
        for v in metrics.values():
            assert isinstance(v, float)


def test_f1_vs_k_returns_figure():
    fig = f1_vs_k([0.1, 0.2, 0.3], [0.15, 0.25, 0.35])
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_real_csv_load():
    root = os.path.join(os.path.dirname(__file__), "..", "data", "insurance_us.csv")
    X, y, y_bin, cols, stats = load_arrays(root)
    assert X.shape[1] == 4
    assert stats["n"] == X.shape[0]
    assert 0.0 <= stats["positive_rate"] <= 1.0
