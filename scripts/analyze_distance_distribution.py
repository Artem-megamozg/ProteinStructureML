from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    index_path = Path(
        "data/processed/dataset_index.csv"
    )

    targets_dir = Path(
        "data/processed/targets"
    )

    metadata = pd.read_csv(
        index_path
    )

    metadata = metadata[
        metadata["split"] == "train"
    ].reset_index(drop=True)

    if metadata.empty:
        raise RuntimeError(
            "Training dataset is empty."
        )

    all_distances = []

    global_max = 0.0

    print(
        f"Training samples: {len(metadata)}"
    )

    for sample_id in metadata[
        "sample_id"
    ]:
        path = (
            targets_dir
            / f"{sample_id}.npz"
        )

        data = np.load(
            path,
            allow_pickle=False,
        )

        distance_map = data[
            "distance_map"
        ]

        target_mask = data[
            "target_mask"
        ]

        length = distance_map.shape[0]

        indices = np.arange(
            length
        )

        separation = np.abs(
            indices[:, None]
            - indices[None, :]
        )

        valid = (
            (target_mask > 0)
            & (separation >= 3)
        )

        values = distance_map[
            valid
        ]

        values = values[
            np.isfinite(values)
        ]

        values = values[
            values > 0
        ]

        if len(values) == 0:
            continue

        all_distances.append(
            values.astype(
                np.float32
            )
        )

        sample_max = float(
            values.max()
        )

        global_max = max(
            global_max,
            sample_max,
        )

    if not all_distances:
        raise RuntimeError(
            "No valid distance values found."
        )

    distances = np.concatenate(
        all_distances
    )

    print()
    print("=" * 70)
    print("TRAIN DISTANCE DISTRIBUTION")
    print("=" * 70)

    print(
        f"Pairs: {len(distances):,}"
    )

    print(
        f"Minimum: "
        f"{distances.min():.3f} Å"
    )

    print(
        f"Maximum: "
        f"{distances.max():.3f} Å"
    )

    print()

    quantiles = [
        0.50,
        0.75,
        0.90,
        0.95,
        0.97,
        0.99,
        0.995,
        0.999,
    ]

    for quantile in quantiles:
        print(
            f"{quantile * 100:6.2f}%: "
            f"{np.quantile(distances, quantile):.3f} Å"
        )

    print()
    print(
        f"Global maximum: "
        f"{global_max:.3f} Å"
    )

    print()
    print("-" * 70)
    print("Distance ranges")
    print("-" * 70)

    ranges = [
        (0, 4),
        (4, 6),
        (6, 8),
        (8, 10),
        (10, 12),
        (12, 16),
        (16, 20),
        (20, 24),
        (24, 32),
        (32, 48),
        (48, 64),
        (64, 80),
        (80, 100),
        (100, 150),
        (150, float("inf")),
    ]

    total = len(distances)

    for lower, upper in ranges:
        if np.isinf(upper):
            mask = distances >= lower
            name = f">={lower}"
        else:
            mask = (
                (distances >= lower)
                & (distances < upper)
            )
            name = f"{lower}-{upper}"

        count = int(
            mask.sum()
        )

        percentage = (
            count / total * 100
        )

        print(
            f"{name:>10} Å: "
            f"{count:10,} "
            f"({percentage:6.2f}%)"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()