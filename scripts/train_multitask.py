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
from src.models.pairwise_multitask import (
    ProteinMultiTaskModel,
)
from src.training.multitask_loss import (
    MultiTaskStructureLoss,
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

    return torch.device(
        requested
    )


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_distance_loss = 0.0
    total_contact_loss = 0.0
    total_batches = 0

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

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):
            predicted_distance, predicted_contact = (
                model(
                    sequence_tokens=sequence_tokens,
                    chemical_shifts=chemical_shifts,
                    chemical_shift_mask=chemical_shift_mask,
                )
            )

            loss, distance_loss, contact_loss = (
                criterion(
                    predicted_distance=predicted_distance,
                    predicted_contact_logits=predicted_contact,
                    target_distance=distance_maps,
                    target_mask=target_masks,
                )
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        total_loss += loss.item()
        total_distance_loss += (
            distance_loss.item()
        )
        total_contact_loss += (
            contact_loss.item()
        )
        total_batches += 1

    return (
        total_loss / max(total_batches, 1),
        total_distance_loss / max(total_batches, 1),
        total_contact_loss / max(total_batches, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
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

    model = ProteinMultiTaskModel(
        chemical_shift_dim=6,
        embedding_dim=32,
        hidden_dim=128,
        pair_hidden_dim=256,
        max_distance=20.0,
        dropout=0.2,
    ).to(device)

    criterion = MultiTaskStructureLoss(
        contact_threshold=8.0,
        contact_loss_weight=2.0,
        max_positive_weight=25.0,
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
        patience=6,
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
        / "protein_multitask_v2.pt"
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
        train_loss, train_distance, train_contact = (
            run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        validation_loss, validation_distance, validation_contact = (
            run_epoch(
                model,
                validation_loader,
                criterion,
                None,
                device,
            )
        )

        scheduler.step(
            validation_loss
        )

        learning_rate = optimizer.param_groups[
            0
        ]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.4f} "
            f"(dist={train_distance:.4f}, "
            f"contact={train_contact:.4f}) | "
            f"val={validation_loss:.4f} "
            f"(dist={validation_distance:.4f}, "
            f"contact={validation_contact:.4f}) | "
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
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
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