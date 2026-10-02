from __future__ import annotations

import torch
from torch import nn


class ProteinMLP(nn.Module):
    """
    Baseline MLP for protein feature prediction.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        hidden_dims = hidden_dims or [256, 512, 256]

        layers: list[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("All hidden dimensions must be positive.")

            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )

            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)