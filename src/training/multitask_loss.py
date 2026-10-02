from __future__ import annotations

import torch
from torch import nn


class MultiTaskStructureLoss(nn.Module):
    def __init__(
        self,
        contact_threshold: float = 8.0,
        contact_loss_weight: float = 2.0,
        max_positive_weight: float = 25.0,
    ) -> None:
        super().__init__()

        self.contact_threshold = contact_threshold
        self.contact_loss_weight = (
            contact_loss_weight
        )
        self.max_positive_weight = (
            max_positive_weight
        )

        self.distance_loss = nn.SmoothL1Loss(
            reduction="none",
            beta=1.0,
        )

    def forward(
        self,
        predicted_distance: torch.Tensor,
        predicted_contact_logits: torch.Tensor,
        target_distance: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        target_distance = torch.nan_to_num(
            target_distance,
            nan=20.0,
            posinf=20.0,
            neginf=0.0,
        )

        target_distance = torch.clamp(
            target_distance,
            min=0.0,
            max=20.0,
        )

        distance_error = self.distance_loss(
            predicted_distance,
            target_distance,
        )

        distance_weight = (
            1.0
            + 5.0
            * torch.exp(
                -target_distance / 4.0
            )
        )

        distance_mask = target_mask

        diagonal = torch.eye(
            target_distance.shape[-1],
            device=target_distance.device,
        ).unsqueeze(0)

        distance_mask = (
            distance_mask
            * (1.0 - diagonal)
        )

        weighted_distance_error = (
            distance_error
            * distance_weight
            * distance_mask
        )

        distance_denominator = (
            distance_weight
            * distance_mask
        ).sum().clamp_min(1.0)

        distance_loss = (
            weighted_distance_error.sum()
            / distance_denominator
        )

        contact_target = (
            target_distance
            <= self.contact_threshold
        ).float()

        indices = torch.arange(
            target_distance.shape[-1],
            device=target_distance.device,
        )

        separation = torch.abs(
            indices[:, None]
            - indices[None, :]
        )

        contact_mask = (
            separation >= 3
        ).float()

        contact_mask = (
            contact_mask.unsqueeze(0)
            * target_mask
        )

        contact_count = (
            contact_target
            * contact_mask
        ).sum()

        non_contact_count = (
            (1.0 - contact_target)
            * contact_mask
        ).sum()

        positive_weight = (
            non_contact_count
            / contact_count.clamp_min(1.0)
        )

        positive_weight = torch.clamp(
            positive_weight,
            min=3.0,
            max=self.max_positive_weight,
        )

        contact_error = nn.functional.binary_cross_entropy_with_logits(
            predicted_contact_logits,
            contact_target,
            reduction="none",
            pos_weight=positive_weight,
        )

        contact_denominator = (
            contact_mask.sum()
            .clamp_min(1.0)
        )

        contact_loss = (
            (
                contact_error
                * contact_mask
            ).sum()
            / contact_denominator
        )

        total_loss = (
            distance_loss
            + self.contact_loss_weight
            * contact_loss
        )

        return (
            total_loss,
            distance_loss,
            contact_loss,
        )