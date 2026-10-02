from __future__ import annotations

import torch
from torch import nn


class MaskedDistanceLoss(nn.Module):
    """
    Weighted masked loss for pairwise distance prediction.

    Short-range distances receive a larger weight because
    they contain more useful local structural information.
    """

    def __init__(
        self,
        max_distance: float = 20.0,
        diagonal_weight: float = 0.0,
    ) -> None:
        super().__init__()

        self.max_distance = max_distance
        self.diagonal_weight = diagonal_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        prediction = torch.nan_to_num(
            prediction,
            nan=0.0,
            posinf=self.max_distance,
            neginf=0.0,
        )

        target = torch.nan_to_num(
            target,
            nan=0.0,
            posinf=self.max_distance,
            neginf=0.0,
        )

        error = (
            prediction - target
        ) ** 2

        weights = torch.ones_like(
            target
        )

        local_weight = (
            1.0
            + 2.0
            * torch.exp(
                -target / 8.0
            )
        )

        weights = (
            weights
            * local_weight
        )

        if self.diagonal_weight == 0.0:
            batch_size, length, _ = target.shape

            diagonal = torch.eye(
                length,
                device=target.device,
            ).unsqueeze(0)

            diagonal = diagonal.expand(
                batch_size,
                -1,
                -1,
            )

            mask = mask * (
                1.0 - diagonal
            )

        weighted_error = (
            error
            * weights
            * mask
        )

        denominator = (
            weights
            * mask
        ).sum().clamp_min(1.0)

        return (
            weighted_error.sum()
            / denominator
        )