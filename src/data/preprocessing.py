from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


AMINO_ACID_TO_INDEX = {
    amino_acid: index
    for index, amino_acid in enumerate("ACDEFGHIKLMNPQRSTVWY", start=1)
}

UNKNOWN_AMINO_ACID_INDEX = 0


def normalize_sequence(sequence: str) -> str:
    """
    Normalize a protein sequence.
    """
    if not isinstance(sequence, str):
        raise TypeError("Protein sequence must be a string.")

    return sequence.strip().upper()


def encode_sequence(
    sequence: str,
    max_length: int,
) -> np.ndarray:
    """
    Encode a protein sequence as integer residue IDs
    and pad/truncate it to a fixed length.
    """
    sequence = normalize_sequence(sequence)

    encoded = np.zeros(max_length, dtype=np.int64)

    for position, amino_acid in enumerate(sequence[:max_length]):
        encoded[position] = AMINO_ACID_TO_INDEX.get(
            amino_acid,
            UNKNOWN_AMINO_ACID_INDEX,
        )

    return encoded


def encode_sequences(
    sequences: Sequence[str],
    max_length: int,
) -> np.ndarray:
    """
    Encode multiple protein sequences.
    """
    encoded_sequences = [
        encode_sequence(sequence, max_length)
        for sequence in sequences
    ]

    return np.stack(encoded_sequences)


def one_hot_encode_sequence(
    sequence: str,
    max_length: int,
) -> np.ndarray:
    """
    One-hot encode a protein sequence.

    Output shape:
        (max_length, 20)
    """
    sequence = normalize_sequence(sequence)

    encoded = np.zeros(
        (max_length, len(AMINO_ACID_TO_INDEX)),
        dtype=np.float32,
    )

    for position, amino_acid in enumerate(sequence[:max_length]):
        index = AMINO_ACID_TO_INDEX.get(amino_acid)

        if index is None:
            continue

        encoded[position, index - 1] = 1.0

    return encoded


def one_hot_encode_sequences(
    sequences: Sequence[str],
    max_length: int,
) -> np.ndarray:
    """
    One-hot encode multiple protein sequences.
    """
    encoded_sequences = [
        one_hot_encode_sequence(sequence, max_length)
        for sequence in sequences
    ]

    return np.stack(encoded_sequences)


def preprocess_sequences(
    dataframe: pd.DataFrame,
    sequence_column: str = "sequence",
    max_length: int = 1024,
    encoding: str = "one_hot",
) -> np.ndarray:
    """
    Convert protein sequences from a dataframe into model-ready arrays.
    """
    if sequence_column not in dataframe.columns:
        raise ValueError(
            f"Column '{sequence_column}' not found in dataframe."
        )

    sequences = dataframe[sequence_column].tolist()

    if encoding == "one_hot":
        return one_hot_encode_sequences(
            sequences,
            max_length,
        )

    if encoding == "integer":
        return encode_sequences(
            sequences,
            max_length,
        )

    raise ValueError(
        f"Unsupported encoding: '{encoding}'. "
        "Available encodings: 'one_hot', 'integer'."
    )