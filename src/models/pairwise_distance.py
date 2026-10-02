from __future__ import annotations

import torch
from torch import nn


class ResidueEncoder(nn.Module):
    """
    Encode each residue using sequence and experimental features.
    """

    def __init__(
        self,
        amino_acid_vocab_size: int = 21,
        chemical_shift_dim: int = 6,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.amino_acid_embedding = nn.Embedding(
            amino_acid_vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        input_dim = (
            embedding_dim
            + chemical_shift_dim * 2
        )

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        chemical_shifts: torch.Tensor,
        chemical_shift_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence_embedding = (
            self.amino_acid_embedding(
                sequence_tokens
            )
        )

        chemical_shifts = torch.nan_to_num(
            chemical_shifts,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        chemical_shifts = torch.clamp(
            chemical_shifts,
            min=-100.0,
            max=100.0,
        )

        chemical_shift_features = torch.cat(
            [
                chemical_shifts,
                chemical_shift_mask,
            ],
            dim=-1,
        )

        features = torch.cat(
            [
                sequence_embedding,
                chemical_shift_features,
            ],
            dim=-1,
        )

        return self.encoder(
            features
        )


class PairwiseDistancePredictor(nn.Module):
    """
    Predict a C-alpha distance for every residue pair.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        pair_hidden_dim: int = 256,
        max_distance: float = 20.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.max_distance = max_distance

        pair_input_dim = (
            hidden_dim * 4
            + 1
        )

        self.pair_network = nn.Sequential(
            nn.Linear(
                pair_input_dim,
                pair_hidden_dim,
            ),
            nn.LayerNorm(pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                pair_hidden_dim,
                pair_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                pair_hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        residue_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, hidden_dim = (
            residue_features.shape
        )

        left = residue_features.unsqueeze(2)
        right = residue_features.unsqueeze(1)

        left = left.expand(
            -1,
            -1,
            length,
            -1,
        )

        right = right.expand(
            -1,
            length,
            -1,
            -1,
        )

        absolute_difference = torch.abs(
            left - right
        )

        product = left * right

        indices = torch.arange(
            length,
            device=residue_features.device,
        )

        sequence_separation = (
            torch.abs(
                indices[:, None]
                - indices[None, :]
            )
            .float()
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
            .unsqueeze(-1)
        )

        sequence_separation = (
            sequence_separation
            / max(length - 1, 1)
        )

        pair_features = torch.cat(
            [
                left,
                right,
                absolute_difference,
                product,
                sequence_separation,
            ],
            dim=-1,
        )

        predictions = self.pair_network(
            pair_features
        ).squeeze(-1)

        predictions = torch.sigmoid(
            predictions
        )

        predictions = (
            predictions
            * self.max_distance
        )

        predictions = (
            predictions
            + predictions.transpose(
                1,
                2,
            )
        ) / 2.0

        diagonal = torch.arange(
            length,
            device=residue_features.device,
        )

        predictions[
            :,
            diagonal,
            diagonal,
        ] = 0.0

        return predictions


class ProteinDistanceModel(nn.Module):
    """
    Full ProteinStructureML v1 distance-prediction model.
    """

    def __init__(
        self,
        chemical_shift_dim: int = 6,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        pair_hidden_dim: int = 256,
        max_distance: float = 20.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.residue_encoder = ResidueEncoder(
            chemical_shift_dim=chemical_shift_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.distance_predictor = (
            PairwiseDistancePredictor(
                hidden_dim=hidden_dim,
                pair_hidden_dim=pair_hidden_dim,
                max_distance=max_distance,
                dropout=dropout,
            )
        )

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        chemical_shifts: torch.Tensor,
        chemical_shift_mask: torch.Tensor,
    ) -> torch.Tensor:
        residue_features = (
            self.residue_encoder(
                sequence_tokens=sequence_tokens,
                chemical_shifts=chemical_shifts,
                chemical_shift_mask=chemical_shift_mask,
            )
        )

        return self.distance_predictor(
            residue_features
        )