import numpy as np
import pandas as pd

import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

import pandas as pd
import numpy as np

def contributions(df: pd.DataFrame, reference) -> pd.DataFrame:
    """
    Relative contribution of each feature to the squared Euclidean distance
    from every row of a DataFrame to a single reference vector.

    Intended for marketers: the output is a DataFrame with the same index
    and the original column names, so it can be read directly.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric columns only. Each row is one observation.
    reference : array-like, pd.Series, or dict
        Must align with the columns of df (same length and order, or
        a Series/dict indexed by the column names).

    Returns
    -------
    pd.DataFrame
        Same index and columns as df.
        Values are fractions (0–1) of the total squared distance
        attributable to that feature. Rows sum to 1.0
        (or 0.0 when the distance is exactly zero).

    Notes
    -----
    Under the L2 norm the squared differences are the additive pieces.
    Reporting the relative shares makes dominance visible at a glance:
    if one column is near 1.0, that single feature is driving the distance.
    A marketer with skin in the game can then decide whether the aggregate
    distance is informative or merely the shadow of one coordinate.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("First argument must be a pandas DataFrame")

    # Align reference to the columns of df
    if isinstance(reference, pd.Series):
        reference = reference.reindex(df.columns)
    elif isinstance(reference, dict):
        reference = pd.Series(reference).reindex(df.columns)
    else:
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 1 or len(reference) != df.shape[1]:
            raise ValueError(
                "reference length must match number of columns in df"
            )
        reference = pd.Series(reference, index=df.columns)

    if reference.isna().any():
        raise ValueError("reference contains NaNs after alignment")

    # Pure numeric work
    X = df.to_numpy(dtype=float)
    ref = reference.to_numpy(dtype=float)

    diffs = X - ref
    squared = diffs ** 2
    sum_sq = squared.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(sum_sq > 0, squared / sum_sq, 0.0)

    return pd.DataFrame(relative, index=df.index, columns=df.columns)

def get_distance(query, reference, measure, feature_order):
    """Get the distance between two vectors.
    query: dict, query vector.
    reference: pd.Series, reference vector.
    measure: str, 'straight line' or 'city block', i.e. euclidean distance or manhattan distance.
    """
    # Convert query to vector.Enforce feature order on the query
    query_vector = np.array([query[col] for col in feature_order], dtype=float)

    diff = reference.to_numpy() - query_vector

    if measure == "straight line":
        return np.sqrt(np.sum(diff ** 2)).round(2)
    elif measure == "city block":
        return np.sum(np.abs(diff)).round(2)
    else:
        raise ValueError(f"Invalid measure: {measure}")

def find_like(df, query, measure, k=5):
    """Find similar customers to a given query.
    df:      pd.DataFrame with features.
    query:   dict, query vector.
    measure: str, 'straight line' or 'city block', i.e. euclidean distance or manhattan distance.
    """
    # Convert query to vector. Enforce the order given by the dataframe columns.
    query_vector = np.array([query[col] for col in df.columns], dtype=float).reshape(1, -1)
    # Convert dataframe to numpy array.
    df_array = df.to_numpy()

    print("Query: ", query)

    if measure == 'straight line':
        d2 = np.sqrt(np.sum((df_array - query_vector) ** 2, axis=1))
        k_nearest_idx = np.argpartition(d2, k)[:k]
        print("Similar customers, unscaled data, straight line distance...")
        print('-'*70)
        return df.iloc[k_nearest_idx]
    elif measure == 'city block':
        d1 = np.sum(np.abs(df_array - query_vector), axis=1)
        k_nearest_idx = np.argpartition(d1, k)[:k]
        print("Similar customers, unscaled data, city block distance...")
        print('-'*70)
        return df.iloc[k_nearest_idx]
    else:
        raise ValueError(f"Invalid measure: {measure}")
