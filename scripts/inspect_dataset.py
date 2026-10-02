from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        type=str,
        default=(
            "data/processed/"
            "metadata.csv"
        ),
    )

    args = parser.parse_args()

    metadata_path = Path(
        args.metadata
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    metadata = pd.read_csv(
        metadata_path
    )

    print()
    print("=" * 60)
    print("ProteinStructureML dataset")
    print("=" * 60)

    print(
        f"Samples: {len(metadata)}"
    )

    if metadata.empty:
        print(
            "Dataset is empty."
        )
        return

    print(
        f"Sequence length: "
        f"{metadata['sequence_length'].min():.0f} - "
        f"{metadata['sequence_length'].max():.0f}"
    )

    print(
        f"Mean sequence length: "
        f"{metadata['sequence_length'].mean():.2f}"
    )

    print(
        f"Mean chemical-shift coverage: "
        f"{metadata['chemical_shift_coverage'].mean():.3f}"
    )

    print(
        f"Mean structure coverage: "
        f"{metadata['structure_coverage'].mean():.3f}"
    )

    print(
        f"Mean sequence identity: "
        f"{metadata['sequence_identity'].mean():.3f}"
    )

    print(
        f"Mean sequence coverage: "
        f"{metadata['sequence_coverage'].mean():.3f}"
    )

    feature_dir = (
        metadata_path.parent
        / "features"
    )

    target_dir = (
        metadata_path.parent
        / "targets"
    )

    valid_samples = 0

    for sample_id in metadata[
        "sample_id"
    ]:
        feature_path = (
            feature_dir
            / f"{sample_id}.npz"
        )

        target_path = (
            target_dir
            / f"{sample_id}.npz"
        )

        if not (
            feature_path.exists()
            and target_path.exists()
        ):
            continue

        feature_data = np.load(
            feature_path
        )

        target_data = np.load(
            target_path
        )

        print()
        print(
            f"Sample: {sample_id}"
        )
        print(
            f"  sequence: "
            f"{feature_data['sequence'].item()[:50]}"
            f"{'...' if len(feature_data['sequence'].item()) > 50 else ''}"
        )
        print(
            f"  chemical shifts: "
            f"{feature_data['chemical_shifts'].shape}"
        )
        print(
            f"  distance map: "
            f"{target_data['distance_map'].shape}"
        )
        print(
            f"  target coverage: "
            f"{target_data['target_mask'].mean():.3f}"
        )

        valid_samples += 1

        if valid_samples >= 3:
            break

    print()
    print(
        f"Verified samples: "
        f"{valid_samples}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()