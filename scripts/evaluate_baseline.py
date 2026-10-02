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
from src.evaluation.structure_metrics import (
    contact_metrics_from_distances,
    masked_regression_metrics,
)
from src.models.pairwise_distance import (
    ProteinDistanceModel,
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=(
            "models/"
            "protein_distance_v1.pt"
        ),
    )

    args = parser.parse_args()

    device = resolve_device(
        args.device
    )

    dataset = ProteinStructureDataset(
        index_path=(
            "data/processed/"
            "dataset_index.csv"
        ),
        split="test",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
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

    checkpoint = torch.load(
        args.model,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    all_metrics = []

    predictions_dir = Path(
        "results/predictions"
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with torch.no_grad():
        for batch in loader:
            sample_id = batch[
                "sample_ids"
            ][0]

            sequence = batch[
                "sequences"
            ][0]

            sequence_tokens = batch[
                "sequence_tokens"
            ].to(device)

            chemical_shifts = batch[
                "chemical_shifts"
            ].to(device)

            chemical_shift_mask = batch[
                "chemical_shift_mask"
            ].to(device)

            target = batch[
                "distance_maps"
            ][0].cpu().numpy()

            target_mask = batch[
                "target_masks"
            ][0].cpu().numpy()

            prediction = model(
                sequence_tokens=sequence_tokens,
                chemical_shifts=chemical_shifts,
                chemical_shift_mask=chemical_shift_mask,
            )[0].cpu().numpy()

            length = len(sequence)

            prediction = prediction[
                :length,
                :length,
            ]

            target = target[
                :length,
                :length,
            ]

            target_mask = target_mask[
                :length,
                :length,
            ]

            regression = (
                masked_regression_metrics(
                    prediction,
                    target,
                    target_mask,
                )
            )

            contacts = (
                contact_metrics_from_distances(
                    prediction,
                    target,
                    target_mask,
                )
            )

            metrics = {
                "sample_id": sample_id,
                **regression,
                **contacts,
            }

            all_metrics.append(
                metrics
            )

            np.savez_compressed(
                predictions_dir
                / f"{sample_id}.npz",
                prediction=prediction,
                target=target,
                target_mask=target_mask,
                sequence=np.array(sequence),
            )

            print(
                f"{sample_id}: "
                f"RMSE={regression['rmse']:.3f} Å | "
                f"MAE={regression['mae']:.3f} Å | "
                f"F1={contacts['f1']:.3f}"
            )

    if not all_metrics:
        raise RuntimeError(
            "Test dataset is empty."
        )

    mean_metrics = {
        "rmse": float(
            np.mean(
                [
                    item["rmse"]
                    for item in all_metrics
                ]
            )
        ),
        "mae": float(
            np.mean(
                [
                    item["mae"]
                    for item in all_metrics
                ]
            )
        ),
        "precision": float(
            np.mean(
                [
                    item["precision"]
                    for item in all_metrics
                ]
            )
        ),
        "recall": float(
            np.mean(
                [
                    item["recall"]
                    for item in all_metrics
                ]
            )
        ),
        "f1": float(
            np.mean(
                [
                    item["f1"]
                    for item in all_metrics
                ]
            )
        ),
    }

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(
        f"RMSE:      {mean_metrics['rmse']:.4f} Å"
    )
    print(
        f"MAE:       {mean_metrics['mae']:.4f} Å"
    )
    print(
        f"Precision: {mean_metrics['precision']:.4f}"
    )
    print(
        f"Recall:    {mean_metrics['recall']:.4f}"
    )
    print(
        f"F1:        {mean_metrics['f1']:.4f}"
    )

    print()
    print(
        f"Predictions saved to: "
        f"{predictions_dir}"
    )


if __name__ == "__main__":
    main()