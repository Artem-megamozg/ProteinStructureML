from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.collate import (
    protein_collate_fn,
)
from src.data.torch_dataset import (
    ProteinStructureDataset,
)
from src.models.pairwise_distance import (
    ProteinDistanceModel,
)
from src.training.structure_loss import (
    MaskedDistanceLoss,
)


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        requested
    )

    return device


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_training = (
        optimizer is not None
    )

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_weight = 0.0

    for batch in loader:
        sequence_tokens = batch[
            "sequence_tokens"
        ].to(device)

        chemical_shifts = batch[
            "chemical_shifts"
        ].to(device)

        chemical_shift_mask = batch[
            "chemical_shift_mask"
        ].to(device)

        distance_maps = batch[
            "distance_maps"
        ].to(device)

        target_masks = batch[
            "target_masks"
        ].to(device)

        if is_training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            is_training
        ):
            predictions = model(
                sequence_tokens=sequence_tokens,
                chemical_shifts=chemical_shifts,
                chemical_shift_mask=chemical_shift_mask,
            )

            loss = criterion(
                predictions,
                distance_maps,
                target_masks,
            )

            if is_training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        batch_weight = (
            target_masks.sum()
            .item()
        )

        total_loss += (
            loss.item()
            * max(batch_weight, 1.0)
        )

        total_weight += max(
            batch_weight,
            1.0,
        )

    return (
        total_loss
        / max(total_weight, 1.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    device = resolve_device(
        args.device
    )

    print(
        f"Device: {device}"
    )

    train_dataset = ProteinStructureDataset(
        index_path=(
            "data/processed/"
            "dataset_index.csv"
        ),
        split="train",
    )

    validation_dataset = ProteinStructureDataset(
        index_path=(
            "data/processed/"
            "dataset_index.csv"
        ),
        split="validation",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=protein_collate_fn,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=protein_collate_fn,
        num_workers=0,
    )

    model = ProteinDistanceModel(
        chemical_shift_dim=6,
        embedding_dim=32,
        hidden_dim=128,
        pair_hidden_dim=256,
        max_distance=20.0,
        dropout=0.2,
    ).to(device)

    criterion = MaskedDistanceLoss(
        max_distance=20.0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_validation_loss = float(
        "inf"
    )

    best_epoch = -1

    output_dir = Path(
        "models"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / "protein_distance_v1.pt"
    )

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        scheduler.step(
            validation_loss
        )

        learning_rate = optimizer.param_groups[
            0
        ]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.5f} | "
            f"val={validation_loss:.5f} | "
            f"lr={learning_rate:.2e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "validation_loss": (
                        validation_loss
                    ),
                },
                model_path,
            )

    print()
    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.5f}"
    )

    print(
        f"Model saved to: "
        f"{model_path}"
    )


if __name__ == "__main__":
    main()