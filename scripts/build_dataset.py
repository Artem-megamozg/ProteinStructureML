from __future__ import annotations

import argparse

from src.data.dataset_builder import (
    build_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to build.",
    )

    args = parser.parse_args()

    build_dataset(
        mapping_path=(
            "data/raw/bmrb/"
            "bmrb_pdb_mapping.csv"
        ),
        bmrb_dir=(
            "data/raw/bmrb"
        ),
        pdb_dir=(
            "data/raw/pdb"
        ),
        output_dir=(
            "data/processed"
        ),
        max_samples=args.limit,
    )


if __name__ == "__main__":
    main()