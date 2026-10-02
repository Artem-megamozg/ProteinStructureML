from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


class DatasetValidationError(ValueError):
    """Raised when a protein dataset does not satisfy the required schema."""


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Check that all required columns exist in the dataset.
    """
    required_columns = set(required_columns)
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise DatasetValidationError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def validate_protein_sequences(
    dataframe: pd.DataFrame,
    sequence_column: str = "sequence",
    min_length: int = 20,
    max_length: int = 1024,
) -> None:
    """
    Validate protein amino-acid sequences.
    """
    validate_required_columns(dataframe, [sequence_column])

    invalid_rows: list[int] = []

    for index, sequence in dataframe[sequence_column].items():
        if not isinstance(sequence, str):
            invalid_rows.append(index)
            continue

        sequence = sequence.strip().upper()

        if not min_length <= len(sequence) <= max_length:
            invalid_rows.append(index)
            continue

        if not set(sequence).issubset(STANDARD_AMINO_ACIDS):
            invalid_rows.append(index)

    if invalid_rows:
        preview = invalid_rows[:10]

        raise DatasetValidationError(
            f"Invalid protein sequences found in rows: {preview}. "
            f"Total invalid rows: {len(invalid_rows)}"
        )


def validate_unique_protein_ids(
    dataframe: pd.DataFrame,
    protein_id_column: str = "protein_id",
) -> None:
    """
    Validate protein identifiers.
    """
    validate_required_columns(dataframe, [protein_id_column])

    if dataframe[protein_id_column].isna().any():
        raise DatasetValidationError(
            f"Column '{protein_id_column}' contains missing values."
        )

    duplicated = dataframe[protein_id_column].duplicated()

    if duplicated.any():
        duplicated_ids = (
            dataframe.loc[duplicated, protein_id_column]
            .astype(str)
            .tolist()
        )

        raise DatasetValidationError(
            f"Duplicate protein IDs found: {duplicated_ids[:10]}"
        )


def validate_dataset(
    dataframe: pd.DataFrame,
    sequence_column: str = "sequence",
    protein_id_column: str = "protein_id",
    min_length: int = 20,
    max_length: int = 1024,
) -> None:
    """
    Run all basic dataset validation checks.
    """
    if dataframe.empty:
        raise DatasetValidationError("Dataset is empty.")

    validate_required_columns(
        dataframe,
        [protein_id_column, sequence_column],
    )

    validate_unique_protein_ids(
        dataframe,
        protein_id_column,
    )

    validate_protein_sequences(
        dataframe,
        sequence_column,
        min_length,
        max_length,
    )