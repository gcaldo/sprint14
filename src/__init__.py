"""Sprint 14 (Sure Tomorrow) pure-NumPy helpers."""

from src.data import load_arrays
from src.diagnostics import (
    age_binned_residual_means,
    correlation_heatmap,
    f1_vs_k,
    pairplot,
    residuals_vs_fitted,
)
from src.distances import NeighborResult, distances
from src.models import OLSResult, dummies, knn_f1_sweep, linreg_ols
from src.obfuscation import ObfuscationResult, obfuscate

__all__ = [
    "load_arrays",
    "distances",
    "NeighborResult",
    "linreg_ols",
    "OLSResult",
    "dummies",
    "knn_f1_sweep",
    "obfuscate",
    "ObfuscationResult",
    "f1_vs_k",
    "residuals_vs_fitted",
    "age_binned_residual_means",
    "pairplot",
    "correlation_heatmap",
]
