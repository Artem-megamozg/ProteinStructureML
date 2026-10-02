from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ATOM_NAMES = (
    "H",
    "HA",
    "CA",
    "CB",
    "C",
    "N",
)


def fit_shift_statistics(
    index_path: str | Path,
    processed_dir: str | Path = "data/processed",
    split: str = "train",
) -> dict[str, list[float]]:
    index_path = Path(index_path)
    processed_dir = Path(processed_dir)

    metadata = pd.read_csv(
        index_path
    )

    metadata = metadata[
        metadata["split"] == split
    ].reset_index(drop=True)

    if metadata.empty:
        raise ValueError(
            f"No samples found for split '{split}'."
        )

    sums = np.zeros(
        len(ATOM_NAMES),
        dtype=np.float64,
    )

    squared_sums = np.zeros(
        len(ATOM_NAMES),
        dtype=np.float64,
    )

    counts = np.zeros(
        len(ATOM_NAMES),
        dtype=np.float64,
    )

    features_dir = (
        processed_dir / "features"
    )

    for sample_id in metadata[
        "sample_id"
    ]:
        path = (
            features_dir
            / f"{sample_id}.npz"
        )

        data = np.load(
            path,
            allow_pickle=False,
        )

        values = data[
            "chemical_shifts"
        ].astype(
            np.float64
        )

        mask = data[
            "chemical_shift_mask"
        ].astype(
            bool
        )

        for atom_index in range(
            len(ATOM_NAMES)
        ):
            valid = mask[
                :,
                atom_index,
            ]

            atom_values = values[
                :,
                atom_index,
            ][valid]

            if len(atom_values) == 0:
                continue

            sums[atom_index] += (
                atom_values.sum()
            )

            squared_sums[
                atom_index
            ] += np.square(
                atom_values
            ).sum()

            counts[
                atom_index
            ] += len(atom_values)

    if np.any(counts == 0):
        missing_atoms = [
            ATOM_NAMES[index]
            for index, count in enumerate(
                counts
            )
            if count == 0
        ]

        raise ValueError(
            "No values found for atoms: "
            f"{missing_atoms}"
        )

    means = sums / counts

    variances = (
        squared_sums / counts
        - means ** 2
    )

    variances = np.maximum(
        variances,
        1e-8,
    )

    stds = np.sqrt(
        variances
    )

    return {
        "atoms": list(ATOM_NAMES),
        "mean": means.tolist(),
        "std": stds.tolist(),
        "count": counts.tolist(),
    }


def save_shift_statistics(
    statistics: dict,
    path: str | Path,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            statistics,
            file,
            indent=2,
        )


def load_shift_statistics(
    path: str | Path,
) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Chemical shift statistics "
            f"not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def standardize_chemical_shifts(
    chemical_shifts: np.ndarray,
    chemical_shift_mask: np.ndarray,
    statistics: dict,
) -> np.ndarray:
    values = chemical_shifts.astype(
        np.float32,
        copy=True,
    )

    means = np.asarray(
        statistics["mean"],
        dtype=np.float32,
    )

    stds = np.asarray(
        statistics["std"],
        dtype=np.float32,
    )

    mask = chemical_shift_mask.astype(
        bool
    )

    values = (
        values - means
    ) / stds

    values = np.where(
        mask,
        values,
        0.0,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return values.astype(
        np.float32
    )