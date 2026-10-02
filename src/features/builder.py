from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.experimental_features import (
    build_experimental_features,
)
from src.features.sequence_features import (
    build_sequence_features,
)


@dataclass
class FeatureSet:
    """
    Container for model-ready protein features.
    """

    sequence_features: np.ndarray
    experimental_features: np.ndarray
    sequence_lengths: np.ndarray
    amino_acid_composition: np.ndarray
    experimental_feature_names: list[str]
    experimental_scaler: StandardScaler | None

    @property
    def num_samples(self) -> int:
        return self.sequence_features.shape[0]

    @property
    def num_sequence_features(self) -> int:
        return self.sequence_features.shape[1]

    @property
    def num_experimental_features(self) -> int:
        return self.experimental_features.shape[1]


def build_features(
    dataframe: pd.DataFrame,
    sequence_column: str = "sequence",
    max_sequence_length: int = 1024,
    excluded_experimental_columns: list[str] | None = None,
    missing_value_strategy: str = "mean",
) -> FeatureSet:
    """
    Build the complete feature representation for a protein dataset.
    """
    sequence_features = build_sequence_features(
        dataframe=dataframe,
        sequence_column=sequence_column,
        max_length=max_sequence_length,
    )

    excluded_columns = set(excluded_experimental_columns or [])
    excluded_columns.add(sequence_column)

    experimental_features, scaler, feature_names = (
        build_experimental_features(
            dataframe=dataframe,
            excluded_columns=excluded_columns,
            missing_value_strategy=missing_value_strategy,
        )
    )

    return FeatureSet(
        sequence_features=sequence_features["one_hot"],
        experimental_features=experimental_features,
        sequence_lengths=sequence_features["length"],
        amino_acid_composition=sequence_features["composition"],
        experimental_feature_names=feature_names,
        experimental_scaler=scaler,
    )


def flatten_for_baseline(
    feature_set: FeatureSet,
) -> np.ndarray:
    """
    Flatten sequence features and concatenate them with
    global experimental and sequence-level features.

    This representation will be used by the initial MLP baseline.
    """
    sequence_features = feature_set.sequence_features

    global_features = np.concatenate(
        [
            feature_set.sequence_lengths[:, None],
            feature_set.amino_acid_composition,
            feature_set.experimental_features,
        ],
        axis=1,
    )

    flattened_sequence = sequence_features.reshape(
        sequence_features.shape[0],
        -1,
    )

    return np.concatenate(
        [
            flattened_sequence,
            global_features,
        ],
        axis=1,
    ).astype(np.float32)