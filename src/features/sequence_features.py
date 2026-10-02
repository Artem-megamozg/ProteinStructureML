from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    AMINO_ACID_TO_INDEX,
    one_hot_encode_sequences,
)


AMINO_ACIDS = tuple(AMINO_ACID_TO_INDEX.keys())


def calculate_sequence_lengths(
    dataframe: pd.DataFrame,
    sequence_column: str = "sequence",
) -> np.ndarray:
    """
    Calculate protein sequence lengths.
    """
    if sequence_column not in dataframe.columns:
        raise ValueError(
            f"Column '{sequence_column}' not found in dataframe."
        )

    return (
        dataframe[sequence_column]
        .astype(str)
        .str.strip()
        .str.len()
        .to_numpy(dtype=np.float32)
    )


def calculate_amino_acid_composition(
    sequence: str,
) -> np.ndarray:
    """
    Calculate normalized amino-acid composition.

    Output shape:
        (20,)
    """
    sequence = str(sequence).strip().upper()

    if not sequence:
        return np.zeros(len(AMINO_ACIDS), dtype=np.float32)

    counts = np.array(
        [sequence.count(amino_acid) for amino_acid in AMINO_ACIDS],
        dtype=np.float32,
    )

    return counts / len(sequence)


def calculate_amino_acid_compositions(
    dataframe: pd.DataFrame,
    sequence_column: str = "sequence",
) -> np.ndarray:
    """
    Calculate amino-acid composition for all proteins.
    """
    if sequence_column not in dataframe.columns:
        raise ValueError(
            f"Column '{sequence_column}' not found in dataframe."
        )

    return np.stack(
        [
            calculate_amino_acid_composition(sequence)
            for sequence in dataframe[sequence_column]
        ]
    )


def build_sequence_features(
    dataframe: pd.DataFrame,
    sequence_column: str = "sequence",
    max_length: int = 1024,
) -> dict[str, np.ndarray]:
    """
    Build a collection of sequence-derived features.
    """
    sequences = dataframe[sequence_column].astype(str).tolist()

    return {
        "one_hot": one_hot_encode_sequences(
            sequences,
            max_length,
        ),
        "length": calculate_sequence_lengths(
            dataframe,
            sequence_column,
        ),
        "composition": calculate_amino_acid_compositions(
            dataframe,
            sequence_column,
        ),
    }