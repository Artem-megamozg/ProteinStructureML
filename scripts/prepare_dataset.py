from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.quality import (
    add_sequence_columns,
    apply_quality_filters,
    check_sample_files,
)
from src.data.splitting import (
    create_sequence_groups,
    split_by_sequence_group,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-length",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--min-shift-coverage",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--min-structure-coverage",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    metadata_path = Path(
        "data/processed/metadata.csv"
    )

    output_path = Path(
        "data/processed/"
        "dataset_index.csv"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: "
            f"{metadata_path}"
        )

    metadata = pd.read_csv(
        metadata_path
    )

    print(
        f"Initial samples: "
        f"{len(metadata)}"
    )

    metadata = check_sample_files(
        metadata
    )

    print(
        f"Valid files: "
        f"{len(metadata)}"
    )

    metadata = apply_quality_filters(
        metadata,
        min_sequence_length=(
            args.min_length
        ),
        max_sequence_length=(
            args.max_length
        ),
        min_chemical_shift_coverage=(
            args.min_shift_coverage
        ),
        min_structure_coverage=(
            args.min_structure_coverage
        ),
    )

    print(
        f"After quality filtering: "
        f"{len(metadata)}"
    )

    if metadata.empty:
        raise RuntimeError(
            "No samples remain after "
            "quality filtering."
        )

    metadata = add_sequence_columns(
        metadata
    )

    metadata = create_sequence_groups(
        metadata
    )

    metadata = split_by_sequence_group(
        metadata,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_state=args.seed,
    )

    metadata.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Final dataset:")

    print(
        metadata["split"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        f"Unique sequence groups: "
        f"{metadata['sequence_group'].nunique()}"
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()