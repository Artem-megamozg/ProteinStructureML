from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.long_range import (
    long_range_metrics,
)


def main() -> None:
    predictions_dir = Path(
        "results/predictions_v2"
    )

    files = sorted(
        predictions_dir.glob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            "No v2 prediction files found."
        )

    results = []

    print("=" * 70)
    print(
        "ProteinStructureML long-range contact benchmark"
    )
    print("=" * 70)

    for path in files:
        data = np.load(
            path,
            allow_pickle=False,
        )

        probability = data[
            "contact_probability"
        ]

        target_distance = data[
            "target"
        ]

        target_mask = data[
            "target_mask"
        ]

        result = long_range_metrics(
            probability=probability,
            target_distance=target_distance,
            target_mask=target_mask,
            contact_threshold=8.0,
            min_sequence_separation=24,
        )

        if result is None:
            print()
            print(
                f"{path.stem}: "
                "no valid long-range pairs"
            )
            continue

        results.append(
            result
        )

        print()
        print(
            path.stem
        )

        print(
            f"  Long-range pairs: "
            f"{int(result['num_pairs'])}"
        )

        print(
            f"  Real long-range contacts: "
            f"{int(result['num_contacts'])}"
        )

        print(
            f"  Average Precision: "
            f"{result['average_precision']:.4f}"
        )

        print(
            f"  Top-L/5 precision: "
            f"{result['top_l5_precision']:.4f}"
        )

        print(
            f"  Top-L/10 precision: "
            f"{result['top_l10_precision']:.4f}"
        )

        print(
            f"  Top-L/20 precision: "
            f"{result['top_l20_precision']:.4f}"
        )

    if not results:
        raise RuntimeError(
            "No proteins contain enough "
            "long-range pairs."
        )

    print()
    print("=" * 70)
    print("AGGREGATED LONG-RANGE RESULTS")
    print("=" * 70)

    print(
        f"Proteins evaluated: "
        f"{len(results)}"
    )

    print(
        f"Average Precision: "
        f"{np.mean([r['average_precision'] for r in results]):.4f}"
    )

    print(
        f"Top-L/5 precision: "
        f"{np.mean([r['top_l5_precision'] for r in results]):.4f}"
    )

    print(
        f"Top-L/10 precision: "
        f"{np.mean([r['top_l10_precision'] for r in results]):.4f}"
    )

    print(
        f"Top-L/20 precision: "
        f"{np.mean([r['top_l20_precision'] for r in results]):.4f}"
    )


if __name__ == "__main__":
    main()