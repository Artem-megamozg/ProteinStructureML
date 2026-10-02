from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def select_numeric_features(
    dataframe: pd.DataFrame,
    excluded_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Select numeric experimental measurements from a dataframe.
    """
    excluded = set(excluded_columns or [])

    numeric_columns = dataframe.select_dtypes(
        include=[np.number]
    ).columns

    numeric_columns = [
        column
        for column in numeric_columns
        if column not in excluded
    ]

    return dataframe[numeric_columns].copy()


def handle_missing_values(
    dataframe: pd.DataFrame,
    strategy: str = "mean",
) -> pd.DataFrame:
    """
    Handle missing values in numeric experimental features.
    """
    result = dataframe.copy()

    if result.empty:
        return result

    if strategy == "mean":
        result = result.fillna(result.mean(numeric_only=True))

    elif strategy == "median":
        result = result.fillna(result.median(numeric_only=True))

    elif strategy == "zero":
        result = result.fillna(0.0)

    elif strategy == "drop":
        result = result.dropna()

    else:
        raise ValueError(
            f"Unsupported missing value strategy: '{strategy}'. "
            "Available: mean, median, zero, drop."
        )

    return result


def scale_features(
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardize experimental features.
    """
    if dataframe.empty:
        return np.empty((len(dataframe), 0), dtype=np.float32), StandardScaler()

    scaler = StandardScaler()

    scaled = scaler.fit_transform(dataframe)

    return scaled.astype(np.float32), scaler


def build_experimental_features(
    dataframe: pd.DataFrame,
    excluded_columns: Iterable[str] | None = None,
    missing_value_strategy: str = "mean",
) -> tuple[np.ndarray, StandardScaler | None, list[str]]:
    """
    Build a normalized numeric representation of experimental data.
    """
    features = select_numeric_features(
        dataframe,
        excluded_columns,
    )

    if features.empty:
        return (
            np.empty((len(dataframe), 0), dtype=np.float32),
            None,
            [],
        )

    features = handle_missing_values(
        features,
        missing_value_strategy,
    )

    scaled_features, scaler = scale_features(features)

    return (
        scaled_features,
        scaler,
        features.columns.tolist(),
    )