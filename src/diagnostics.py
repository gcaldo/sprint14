"""Figures for Sprint 14: F1-vs-k, residuals, pairplot, correlation heatmap."""

from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import COLUMN_NAMES, FeatureMatrix, Target


def f1_vs_k(
    f1_raw: List[float], f1_scaled: List[float]
) -> plt.Figure:
    """Plot F1 against k = 1…10 for raw and MaxAbs-scaled features."""
    k_raw = list(range(1, len(f1_raw) + 1))
    k_scaled = list(range(1, len(f1_scaled) + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_raw, f1_raw, marker="o", label="raw")
    ax.plot(k_scaled, f1_scaled, marker="s", label="MaxAbs-scaled")
    ax.set_xlabel("k")
    ax.set_ylabel("F1")
    ax.set_title("F1 versus k (kNN classifier)")
    ax.set_xticks(sorted(set(k_raw) | set(k_scaled)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def residuals_vs_fitted(
    y: Target, predictions: np.ndarray
) -> plt.Figure:
    """Residuals versus fitted values (makes zero-mass / misspecification visible)."""
    y = np.asarray(y, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    resid = y - predictions

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(predictions, resid, alpha=0.35, s=12)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Fitted (raw OLS prediction)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs fitted (OLS — no truncation)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def age_binned_residual_means(
    age: np.ndarray,
    residuals: np.ndarray,
    bins: Optional[Sequence[float]] = None,
) -> plt.Figure:
    """Mean residual by age bin — surfaces the age-step structure."""
    age = np.asarray(age, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    if bins is None:
        bins = np.arange(age.min(), age.max() + 5, 5)

    digitize = np.digitize(age, bins)
    centers = []
    means = []
    for b in range(1, len(bins)):
        mask = digitize == b
        if not np.any(mask):
            continue
        centers.append(0.5 * (bins[b - 1] + bins[b]))
        means.append(float(residuals[mask].mean()))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centers, means, width=3.5, align="center")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Age (bin center)")
    ax.set_ylabel("Mean residual")
    ax.set_title("Age-binned residual means")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def pairplot(
    X: FeatureMatrix,
    y: Optional[Target] = None,
    column_names: Optional[Sequence[str]] = None,
) -> sns.axisgrid.PairGrid:
    """Original pairwise plot of features (optionally with the target)."""
    names = list(column_names) if column_names is not None else list(COLUMN_NAMES)
    data = {names[i]: X[:, i] for i in range(X.shape[1])}
    if y is not None:
        data["insurance_benefits"] = np.asarray(y)
    df = pd.DataFrame(data)
    g = sns.pairplot(df, kind="hist")
    g.fig.set_size_inches(12, 12)
    return g


def correlation_heatmap(
    X: FeatureMatrix,
    y: Optional[Target] = None,
    column_names: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """Correlation heatmap of features (optionally including the target)."""
    names = list(column_names) if column_names is not None else list(COLUMN_NAMES)
    data = {names[i]: X[:, i] for i in range(X.shape[1])}
    if y is not None:
        data["insurance_benefits"] = np.asarray(y)
    df = pd.DataFrame(data)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig
