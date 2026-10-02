from __future__ import annotations

import itertools

import pandas as pd

from src.data.similarity import (
    calculate_sequence_similarity,
)


def main() -> None:
    metadata = pd.read_csv(
        "data/processed/dataset_index.csv"
    )

    if metadata.empty:
        raise RuntimeError(
            "Dataset index is empty."
        )

    print("=" * 70)
    print("ProteinStructureML split audit")
    print("=" * 70)

    print()
    print(
        metadata["split"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        f"Total samples: {len(metadata)}"
    )

    train = metadata[
        metadata["split"] == "train"
    ].reset_index(drop=True)

    validation = metadata[
        metadata["split"] == "validation"
    ].reset_index(drop=True)

    test = metadata[
        metadata["split"] == "test"
    ].reset_index(drop=True)

    print()
    print("-" * 70)
    print("Cross-split similarity")
    print("-" * 70)

    pairs = [
        (
            "train-test",
            train,
            test,
        ),
        (
            "train-validation",
            train,
            validation,
        ),
        (
            "validation-test",
            validation,
            test,
        ),
    ]

    for pair_name, left, right in pairs:
        max_identity = 0.0
        max_pair = None

        high_similarity_count = 0

        for i, j in itertools.product(
            range(len(left)),
            range(len(right)),
        ):
            result = calculate_sequence_similarity(
                left.loc[i, "sequence"],
                right.loc[j, "sequence"],
            )

            effective_coverage = min(
                result.coverage_a,
                result.coverage_b,
            )

            if (
                result.identity >= 0.70
                and effective_coverage >= 0.80
            ):
                high_similarity_count += 1

            if result.identity > max_identity:
                max_identity = result.identity
                max_pair = (
                    left.loc[i, "sample_id"],
                    right.loc[j, "sample_id"],
                    result.identity,
                    effective_coverage,
                )

        print()
        print(
            f"{pair_name}:"
        )

        print(
            f"  Highest identity: "
            f"{max_identity:.4f}"
        )

        if max_pair is not None:
            print(
                f"  Pair: "
                f"{max_pair[0]} <-> {max_pair[1]}"
            )

            print(
                f"  Coverage: "
                f"{max_pair[3]:.4f}"
            )

        print(
            f"  Similar pairs "
            f"(identity >= 0.70, coverage >= 0.80): "
            f"{high_similarity_count}"
        )

    print()
    print("-" * 70)
    print("Duplicate sequence groups")
    print("-" * 70)

    duplicate_groups = (
        metadata[
            metadata.duplicated(
                "sequence_key",
                keep=False,
            )
        ]
        .sort_values(
            "sequence_key"
        )
    )

    if duplicate_groups.empty:
        print(
            "No exact sequence duplicates."
        )
    else:
        for sequence_key, group in (
            duplicate_groups.groupby(
                "sequence_key"
            )
        ):
            print()
            print(
                f"Sequence length: "
                f"{len(sequence_key)}"
            )

            print(
                group[
                    [
                        "sample_id",
                        "split",
                        "pdb_id",
                    ]
                ].to_string(
                    index=False
                )
            )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()