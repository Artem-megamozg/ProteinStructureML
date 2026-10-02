from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def check_sample_files(
    metadata: pd.DataFrame,
    processed_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """
    Check that feature and target files exist and can be loaded.
    """
    processed_dir = Path(processed_dir)

    features_dir = processed_dir / "features"
    targets_dir = processed_dir / "targets"

    valid_rows = []

    for _, row in metadata.iterrows():
        sample_id = str(row["sample_id"])

        feature_path = (
            features_dir / f"{sample_id}.npz"
        )

        target_path = (
            targets_dir / f"{sample_id}.npz"
        )

        if not feature_path.exists():
            continue

        if not target_path.exists():
            continue

        try:
            feature_data = np.load(
                feature_path,
                allow_pickle=False,
            )

            target_data = np.load(
                target_path,
                allow_pickle=False,
            )

            sequence = str(
                feature_data["sequence"].item()
            )

            chemical_shifts = feature_data[
                "chemical_shifts"
            ]

            chemical_shift_mask = feature_data[
                "chemical_shift_mask"
            ]

            distance_map = target_data[
                "distance_map"
            ]

            target_mask = target_data[
                "target_mask"
            ]

            if chemical_shifts.ndim != 2:
                continue

            if chemical_shift_mask.shape != chemical_shifts.shape:
                continue

            if distance_map.ndim != 2:
                continue

            if distance_map.shape[0] != distance_map.shape[1]:
                continue

            if target_mask.shape != distance_map.shape:
                continue

            valid_rows.append(row.to_dict())

        except Exception:
            continue

    return pd.DataFrame(valid_rows)


def apply_quality_filters(
    metadata: pd.DataFrame,
    min_sequence_length: int = 30,
    max_sequence_length: int = 512,
    min_chemical_shift_coverage: float = 0.40,
    min_structure_coverage: float = 0.80,
) -> pd.DataFrame:
    """
    Apply quality filters to the processed dataset.
    """
    required_columns = {
        "sample_id",
        "sequence_length",
        "chemical_shift_coverage",
        "structure_coverage",
    }

    missing = required_columns - set(
        metadata.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    mask = (
        (metadata["sequence_length"] >= min_sequence_length)
        & (metadata["sequence_length"] <= max_sequence_length)
        & (
            metadata["chemical_shift_coverage"]
            >= min_chemical_shift_coverage
        )
        & (
            metadata["structure_coverage"]
            >= min_structure_coverage
        )
    )

    return (
        metadata.loc[mask]
        .reset_index(drop=True)
    )


def add_sequence_columns(
    metadata: pd.DataFrame,
    processed_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """
    Add the actual protein sequence to metadata.
    """
    processed_dir = Path(processed_dir)

    features_dir = processed_dir / "features"

    sequences = []

    for sample_id in metadata["sample_id"]:
        path = (
            features_dir
            / f"{sample_id}.npz"
        )

        data = np.load(
            path,
            allow_pickle=False,
        )

        sequences.append(
            str(data["sequence"].item())
        )

    result = metadata.copy()
    result["sequence"] = sequences

    result["sequence_key"] = (
        result["sequence"]
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )

    return result