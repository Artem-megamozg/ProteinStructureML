from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.shift_scaling import (
    load_shift_statistics,
    standardize_chemical_shifts,
)


class ProteinStructureDataset(Dataset):
    """
    Dataset for protein structure prediction.

    Each sample contains:
    - amino-acid sequence
    - standardized experimental chemical shifts
    - chemical-shift availability mask
    - target C-alpha distance map
    - target validity mask
    """

    def __init__(
        self,
        index_path: str | Path,
        split: str,
        processed_dir: str | Path = "data/processed",
        max_distance: float = 20.0,
        shift_stats_path: str | Path | None = None,
    ) -> None:
        self.index_path = Path(
            index_path
        )

        self.processed_dir = Path(
            processed_dir
        )

        self.max_distance = (
            max_distance
        )

        metadata = pd.read_csv(
            self.index_path
        )

        if "split" not in metadata.columns:
            raise ValueError(
                "Dataset index does not contain "
                "'split'."
            )

        self.metadata = (
            metadata[
                metadata["split"] == split
            ]
            .reset_index(drop=True)
        )

        if self.metadata.empty:
            raise ValueError(
                f"No samples found for "
                f"split '{split}'."
            )

        self.features_dir = (
            self.processed_dir
            / "features"
        )

        self.targets_dir = (
            self.processed_dir
            / "targets"
        )

        if shift_stats_path is None:
            shift_stats_path = (
                self.processed_dir
                / "chemical_shift_stats.json"
            )

        self.shift_statistics = (
            load_shift_statistics(
                shift_stats_path
            )
        )

    def __len__(self) -> int:
        return len(
            self.metadata
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        row = self.metadata.iloc[
            index
        ]

        sample_id = str(
            row["sample_id"]
        )

        feature_path = (
            self.features_dir
            / f"{sample_id}.npz"
        )

        target_path = (
            self.targets_dir
            / f"{sample_id}.npz"
        )

        features = np.load(
            feature_path,
            allow_pickle=False,
        )

        targets = np.load(
            target_path,
            allow_pickle=False,
        )

        sequence = str(
            features[
                "sequence"
            ].item()
        )

        chemical_shifts = (
            features[
                "chemical_shifts"
            ].astype(np.float32)
        )

        chemical_shift_mask = (
            features[
                "chemical_shift_mask"
            ].astype(np.float32)
        )

        chemical_shifts = (
            standardize_chemical_shifts(
                chemical_shifts,
                chemical_shift_mask,
                self.shift_statistics,
            )
        )

        distance_map = (
            targets[
                "distance_map"
            ].astype(np.float32)
        )

        target_mask = (
            targets[
                "target_mask"
            ].astype(np.float32)
        )

        distance_map = np.nan_to_num(
            distance_map,
            nan=self.max_distance,
            posinf=self.max_distance,
            neginf=0.0,
        )

        distance_map = np.clip(
            distance_map,
            0.0,
            self.max_distance,
        )

        sequence_tokens = (
            self._encode_sequence(
                sequence
            )
        )

        return {
            "sample_id": sample_id,
            "sequence": sequence,
            "sequence_tokens": torch.from_numpy(
                sequence_tokens
            ),
            "chemical_shifts": torch.from_numpy(
                chemical_shifts
            ),
            "chemical_shift_mask": torch.from_numpy(
                chemical_shift_mask
            ),
            "distance_map": torch.from_numpy(
                distance_map
            ),
            "target_mask": torch.from_numpy(
                target_mask
            ),
        }

    @staticmethod
    def _encode_sequence(
        sequence: str,
    ) -> np.ndarray:
        amino_acids = (
            "ACDEFGHIKLMNPQRSTVWY"
        )

        mapping = {
            amino_acid: index
            for index, amino_acid in enumerate(
                amino_acids,
                start=1,
            )
        }

        return np.asarray(
            [
                mapping.get(
                    amino_acid,
                    0,
                )
                for amino_acid in sequence
            ],
            dtype=np.int64,
        )