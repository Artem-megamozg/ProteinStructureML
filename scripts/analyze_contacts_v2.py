from __future__ import annotations

from pathlib import Path

import numpy as np


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

    for path in files:
        data = np.load(
            path,
            allow_pickle=False,
        )

        probability = data[
            "contact_probability"
        ]

        target = data[
            "target"
        ]

        mask = data[
            "target_mask"
        ]

        sequence = str(
            data["sequence"].item()
        )

        length = len(
            sequence
        )

        indices = np.arange(
            length
        )

        separation = np.abs(
            indices[:, None]
            - indices[None, :]
        )

        valid = (
            (mask > 0)
            & (separation >= 3)
        )

        probabilities = probability[
            valid
        ]

        target_contacts = (
            target[valid] <= 8.0
        )

        print()
        print("=" * 60)
        print(path.stem)
        print("=" * 60)

        print(
            f"Probability min: "
            f"{probabilities.min():.4f}"
        )

        print(
            f"Probability max: "
            f"{probabilities.max():.4f}"
        )

        print(
            f"Probability mean: "
            f"{probabilities.mean():.4f}"
        )

        print(
            f"Probability std: "
            f"{probabilities.std():.4f}"
        )

        print(
            f"P(contact) >= 0.5: "
            f"{np.mean(probabilities >= 0.5):.4f}"
        )

        print(
            f"Real contact fraction: "
            f"{target_contacts.mean():.4f}"
        )

        for threshold in (
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ):
            predicted = (
                probabilities >= threshold
            )

            tp = np.logical_and(
                predicted,
                target_contacts,
            ).sum()

            fp = np.logical_and(
                predicted,
                ~target_contacts,
            ).sum()

            fn = np.logical_and(
                ~predicted,
                target_contacts,
            ).sum()

            precision = (
                tp / max(tp + fp, 1)
            )

            recall = (
                tp / max(tp + fn, 1)
            )

            f1 = (
                2
                * precision
                * recall
                / max(
                    precision + recall,
                    1e-12,
                )
            )

            print(
                f"threshold={threshold:.1f} | "
                f"precision={precision:.3f} | "
                f"recall={recall:.3f} | "
                f"F1={f1:.3f}"
            )


if __name__ == "__main__":
    main()