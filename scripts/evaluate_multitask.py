from __future__ import annotations

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
    contact_metrics_from_probabilities,
    masked_regression_metrics,
)
from src.models.pairwise_multitask import (
    ProteinMultiTaskModel,
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
    device = resolve_device(
        "auto"
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

    model = ProteinMultiTaskModel(
        chemical_shift_dim=6,
        embedding_dim=32,
        hidden_dim=128,
        pair_hidden_dim=256,
        max_distance=20.0,
        dropout=0.2,
    ).to(device)

    checkpoint = torch.load(
        "models/protein_multitask_v2.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    predictions_dir = Path(
        "results/predictions_v2"
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = []

    with torch.no_grad():
        for batch in loader:
            sample_id = batch[
                "sample_ids"
            ][0]

            sequence = batch[
                "sequences"
            ][0]

            predicted_distance, predicted_contact_logits = (
                model(
                    sequence_tokens=batch[
                        "sequence_tokens"
                    ].to(device),
                    chemical_shifts=batch[
                        "chemical_shifts"
                    ].to(device),
                    chemical_shift_mask=batch[
                        "chemical_shift_mask"
                    ].to(device),
                )
            )

            prediction = predicted_distance[
                0
            ].cpu().numpy()

            contact_probability = torch.sigmoid(
                predicted_contact_logits[0]
            ).cpu().numpy()

            target = batch[
                "distance_maps"
            ][0].numpy()

            target_mask = batch[
                "target_masks"
            ][0].numpy()

            length = len(
                sequence
            )

            prediction = prediction[
                :length,
                :length,
            ]

            contact_probability = (
                contact_probability[
                    :length,
                    :length,
                ]
            )

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

            distance_contacts = (
                contact_metrics_from_distances(
                    prediction,
                    target,
                    target_mask,
                )
            )

            probability_contacts = (
                contact_metrics_from_probabilities(
                    probability=contact_probability,
                    target=target,
                    mask=target_mask,
                    threshold=0.5,
                )
            )

            metrics.append(
                {
                    **regression,
                    **probability_contacts,
                }
            )

            np.savez_compressed(
                predictions_dir
                / f"{sample_id}.npz",
                prediction=prediction,
                contact_probability=contact_probability,
                target=target,
                target_mask=target_mask,
                sequence=np.array(sequence),
            )

            print()
            print(
                sample_id
            )

            print(
                f"  Distance MAE: "
                f"{regression['mae']:.3f} Å"
            )

            print(
                f"  Distance RMSE: "
                f"{regression['rmse']:.3f} Å"
            )

            print(
                f"  Distance-derived F1: "
                f"{distance_contacts['f1']:.3f}"
            )

            print(
                f"  Contact F1: "
                f"{probability_contacts['f1']:.3f}"
            )

            print(
                f"  Contact precision: "
                f"{probability_contacts['precision']:.3f}"
            )

            print(
                f"  Contact recall: "
                f"{probability_contacts['recall']:.3f}"
            )

            print(
                f"  Contact AP: "
                f"{probability_contacts['average_precision']:.3f}"
            )

            print(
                f"  Top-L/5 precision: "
                f"{probability_contacts['top_l5_precision']:.3f}"
            )

            print(
                f"  Top-L/10 precision: "
                f"{probability_contacts['top_l10_precision']:.3f}"
            )

    if not metrics:
        raise RuntimeError(
            "Test dataset is empty."
        )

    print()
    print("=" * 60)
    print("MULTI-TASK V2 TEST RESULTS")
    print("=" * 60)

    print(
        f"Distance RMSE: "
        f"{np.mean([m['rmse'] for m in metrics]):.4f} Å"
    )

    print(
        f"Distance MAE:  "
        f"{np.mean([m['mae'] for m in metrics]):.4f} Å"
    )

    print(
        f"Contact Precision: "
        f"{np.mean([m['precision'] for m in metrics]):.4f}"
    )

    print(
        f"Contact Recall:    "
        f"{np.mean([m['recall'] for m in metrics]):.4f}"
    )

    print(
        f"Contact F1:        "
        f"{np.mean([m['f1'] for m in metrics]):.4f}"
    )

    print(
        f"Contact AP:        "
        f"{np.mean([m['average_precision'] for m in metrics]):.4f}"
    )

    print(
        f"Top-L/5 Precision: "
        f"{np.mean([m['top_l5_precision'] for m in metrics]):.4f}"
    )

    print(
        f"Top-L/10 Precision: "
        f"{np.mean([m['top_l10_precision'] for m in metrics]):.4f}"
    )

    print()
    print(
        f"Predictions: "
        f"{predictions_dir}"
    )


if __name__ == "__main__":
    main()