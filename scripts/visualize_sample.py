from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    target_path = Path(
        "data/processed/targets"
    ) / f"{args.sample}.npz"

    feature_path = Path(
        "data/processed/features"
    ) / f"{args.sample}.npz"

    if not target_path.exists():
        raise FileNotFoundError(
            f"Target not found: {target_path}"
        )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}"
        )

    target = np.load(
        target_path,
        allow_pickle=False,
    )

    features = np.load(
        feature_path,
        allow_pickle=False,
    )

    distance_map = target[
        "distance_map"
    ]

    target_mask = target[
        "target_mask"
    ]

    sequence = str(
        features[
            "sequence"
        ].item()
    )

    display_map = distance_map.copy()

    display_map[
        target_mask == 0
    ] = np.nan

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        display_map,
        interpolation="nearest",
    )

    plt.title(
        f"{args.sample} | "
        f"L={len(sequence)}"
    )

    plt.xlabel(
        "Residue index"
    )

    plt.ylabel(
        "Residue index"
    )

    plt.colorbar(
        label="Cα-Cα distance (Å)"
    )

    plt.tight_layout()

    output_dir = Path(
        "results/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{args.sample}_distance_map.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()