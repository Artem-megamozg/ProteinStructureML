from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.structure.reconstruction import (
    calculate_rmsd,
    classical_mds,
    kabsch_align,
)


def reconstruct_rmsd(
    distance_matrix: np.ndarray,
    target_coordinates: np.ndarray,
    allow_reflection: bool,
) -> float:
    predicted_coordinates = classical_mds(
        distance_matrix
    )

    aligned_coordinates, _ = kabsch_align(
        predicted_coordinates,
        target_coordinates,
        allow_reflection=allow_reflection,
    )

    return calculate_rmsd(
        aligned_coordinates,
        target_coordinates,
    )


def calculate_distance_errors(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    valid = np.isfinite(
        target
    )

    difference = np.abs(
        prediction[valid]
        - target[valid]
    )

    return {
        "mae": float(
            difference.mean()
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    difference ** 2
                )
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    sample = args.sample

    prediction_path = (
        Path("results/predictions_v2")
        / f"{sample}.npz"
    )

    target_path = (
        Path("data/processed/targets")
        / f"{sample}.npz"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction not found: "
            f"{prediction_path}"
        )

    if not target_path.exists():
        raise FileNotFoundError(
            f"Target not found: "
            f"{target_path}"
        )

    prediction_data = np.load(
        prediction_path,
        allow_pickle=False,
    )

    target_data = np.load(
        target_path,
        allow_pickle=False,
    )

    predicted_distance = (
        prediction_data["prediction"]
    )

    true_distance = (
        target_data["distance_map"]
    )

    target_coordinates = (
        target_data["coordinates"]
    )

    valid_mask = (
        target_data["valid_residue_mask"]
    )

    finite_coordinates = np.all(
        np.isfinite(
            target_coordinates
        ),
        axis=1,
    )

    valid = (
        valid_mask
        & finite_coordinates
    )

    valid_indices = np.flatnonzero(
        valid
    )

    predicted_distance = (
        predicted_distance[
            np.ix_(
                valid_indices,
                valid_indices,
            )
        ]
    )

    true_distance = (
        true_distance[
            np.ix_(
                valid_indices,
                valid_indices,
            )
        ]
    )

    coordinates = (
        target_coordinates[
            valid_indices
        ]
    )

    predicted_distance = (
        predicted_distance
        + predicted_distance.T
    ) / 2.0

    true_distance = (
        true_distance
        + true_distance.T
    ) / 2.0

    np.fill_diagonal(
        predicted_distance,
        0.0,
    )

    np.fill_diagonal(
        true_distance,
        0.0,
    )

    clipped_true_distance = np.clip(
        true_distance,
        0.0,
        20.0,
    )

    predicted_metrics = (
        calculate_distance_errors(
            predicted_distance,
            true_distance,
        )
    )

    clipped_metrics = (
        calculate_distance_errors(
            predicted_distance,
            clipped_true_distance,
        )
    )

    true_rotation_rmsd = reconstruct_rmsd(
        true_distance,
        coordinates,
        allow_reflection=False,
    )

    true_reflection_rmsd = reconstruct_rmsd(
        true_distance,
        coordinates,
        allow_reflection=True,
    )

    clipped_rotation_rmsd = reconstruct_rmsd(
        clipped_true_distance,
        coordinates,
        allow_reflection=False,
    )

    clipped_reflection_rmsd = reconstruct_rmsd(
        clipped_true_distance,
        coordinates,
        allow_reflection=True,
    )

    predicted_rotation_rmsd = reconstruct_rmsd(
        predicted_distance,
        coordinates,
        allow_reflection=False,
    )

    predicted_reflection_rmsd = reconstruct_rmsd(
        predicted_distance,
        coordinates,
        allow_reflection=True,
    )

    print()
    print("=" * 70)
    print("RECONSTRUCTION DIAGNOSTIC")
    print("=" * 70)

    print()
    print(
        f"Sample: {sample}"
    )

    print(
        f"Residues: {len(valid_indices)}"
    )

    print()
    print("-" * 70)
    print("Distance prediction")
    print("-" * 70)

    print(
        f"Prediction vs full target MAE: "
        f"{predicted_metrics['mae']:.4f} Å"
    )

    print(
        f"Prediction vs full target RMSE: "
        f"{predicted_metrics['rmse']:.4f} Å"
    )

    print(
        f"Prediction vs clipped target MAE: "
        f"{clipped_metrics['mae']:.4f} Å"
    )

    print(
        f"Prediction range: "
        f"{predicted_distance.min():.3f} - "
        f"{predicted_distance.max():.3f} Å"
    )

    print(
        f"True distance range: "
        f"{true_distance.min():.3f} - "
        f"{true_distance.max():.3f} Å"
    )

    print()
    print("-" * 70)
    print("MDS reconstruction")
    print("-" * 70)

    print(
        f"True full distance → "
        f"rotation-only RMSD: "
        f"{true_rotation_rmsd:.4f} Å"
    )

    print(
        f"True full distance → "
        f"reflection-allowed RMSD: "
        f"{true_reflection_rmsd:.4f} Å"
    )

    print(
        f"True clipped distance → "
        f"rotation-only RMSD: "
        f"{clipped_rotation_rmsd:.4f} Å"
    )

    print(
        f"True clipped distance → "
        f"reflection-allowed RMSD: "
        f"{clipped_reflection_rmsd:.4f} Å"
    )

    print(
        f"Predicted distance → "
        f"rotation-only RMSD: "
        f"{predicted_rotation_rmsd:.4f} Å"
    )

    print(
        f"Predicted distance → "
        f"reflection-allowed RMSD: "
        f"{predicted_reflection_rmsd:.4f} Å"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()