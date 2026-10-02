from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ProteinFeatureDataset(Dataset):
    """
    PyTorch dataset for prepared protein features and targets.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        if len(features) != len(targets):
            raise ValueError(
                "Features and targets must contain the same "
                f"number of samples: {len(features)} != {len(targets)}"
            )

        if len(features) == 0:
            raise ValueError("Dataset cannot be empty.")

        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.features[index],
            self.targets[index],
        )