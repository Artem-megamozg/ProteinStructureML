from __future__ import annotations

import torch
from torch import nn


class MaskedMSELoss(nn.Module):
    """
    Mean squared error with optional masking.
    """

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        error = (prediction - target) ** 2

        if mask is None:
            return error.mean()

        mask = mask.to(dtype=error.dtype)

        if mask.shape != error.shape:
            mask = torch.broadcast_to(mask, error.shape)

        valid_error = error * mask
        denominator = mask.sum().clamp_min(1.0)

        return valid_error.sum() / denominator


class MaskedMAELoss(nn.Module):
    """
    Mean absolute error with optional masking.
    """

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        error = torch.abs(prediction - target)

        if mask is None:
            return error.mean()

        mask = mask.to(dtype=error.dtype)

        if mask.shape != error.shape:
            mask = torch.broadcast_to(mask, error.shape)

        valid_error = error * mask
        denominator = mask.sum().clamp_min(1.0)

        return valid_error.sum() / denominator


def get_loss_function(name: str) -> nn.Module:
    """
    Create a loss function by name.
    """
    normalized_name = name.lower().strip()

    if normalized_name == "mse":
        return nn.MSELoss()

    if normalized_name == "mae":
        return nn.L1Loss()

    if normalized_name == "masked_mse":
        return MaskedMSELoss()

    if normalized_name == "masked_mae":
        return MaskedMAELoss()

    raise ValueError(
        f"Unknown loss function: '{name}'. "
        "Available: mse, mae, masked_mse, masked_mae."
    )