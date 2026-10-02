from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def analyze_sample(
    path: Path,
) -> dict[str, float]:
    data = np.load(
        path,
        allow_pickle=False,
    )

    prediction = data["prediction"]
    target = data["target"]
    mask = data["target_mask"]

    valid = mask > 0

    indices = np.arange(
        prediction.shape[0]
    )

    separation = np.abs(
        indices[:, None]
        - indices[None, :]
    )

    valid = (
        valid
        & (separation >= 3)
    )

    prediction_values = prediction[valid]
    target_values = target[valid]

    prediction_contacts = (
        prediction_values <= 8.0
    )

    target_contacts = (
        target_values <= 8.0
    )

    predicted_contact_count = (
        int(prediction_contacts.sum())
    )

    target_contact_count = (
        int(target_contacts.sum())
    )

    if (
        len(prediction_values) > 1
        and np.std(prediction_values) > 0
        and np.std(target_values) > 0
    ):
        pearson = float(
            pearsonr(
                prediction_values,
                target_values,
            ).statistic
        )

        spearman = float(
            spearmanr(
                prediction_values,
                target_values,
            ).statistic
        )
    else:
        pearson = 0.0
        spearman = 0.0

    contact_target_values = (
        target_values[target_contacts]
    )

    contact_prediction_values = (
        prediction_values[target_contacts]
    )

    if len(contact_target_values) > 0:
        contact_mae = float(
            np.mean(
                np.abs(
                    contact_prediction_values
                    - contact_target_values
                )
            )
        )
    else:
        contact_mae = 0.0

    sorted_indices = np.argsort(
        prediction_values
    )

    top_k = max(
        1,
        len(sorted_indices) // 5,
    )

    top_indices = (
        sorted_indices[:top_k]
    )

    top_k_target_contacts = (
        target_contacts[top_indices]
    )

    top_k_precision = float(
        top_k_target_contacts.mean()
    )

    return {
        "prediction_min": float(
            prediction_values.min()
        ),
        "prediction_max": float(
            prediction_values.max()
        ),
        "prediction_mean": float(
            prediction_values.mean()
        ),
        "prediction_std": float(
            prediction_values.std()
        ),
        "target_min": float(
            target_values.min()
        ),
        "target_max": float(
            target_values.max()
        ),
        "target_mean": float(
            target_values.mean()
        ),
        "target_std": float(
            target_values.std()
        ),
        "predicted_contact_fraction": float(
            prediction_contacts.mean()
        ),
        "target_contact_fraction": float(
            target_contacts.mean()
        ),
        "predicted_contact_count": float(
            predicted_contact_count
        ),
        "target_contact_count": float(
            target_contact_count
        ),
        "contact_mae": contact_mae,
        "pearson": pearson,
        "spearman": spearman,
        "top_l5_precision": top_k_precision,
    }


def main() -> None:
    predictions_dir = Path(
        "results/predictions"
    )

    prediction_files = sorted(
        predictions_dir.glob("*.npz")
    )

    if not prediction_files:
        raise FileNotFoundError(
            "No prediction files found."
        )

    print("=" * 70)
    print("ProteinStructureML prediction diagnostics")
    print("=" * 70)

    for path in prediction_files:
        metrics = analyze_sample(
            path
        )

        print()
        print(path.stem)

        print(
            f"Prediction range: "
            f"{metrics['prediction_min']:.3f} - "
            f"{metrics['prediction_max']:.3f} Å"
        )

        print(
            f"Prediction mean/std: "
            f"{metrics['prediction_mean']:.3f} / "
            f"{metrics['prediction_std']:.3f} Å"
        )

        print(
            f"Target range: "
            f"{metrics['target_min']:.3f} - "
            f"{metrics['target_max']:.3f} Å"
        )

        print(
            f"Target mean/std: "
            f"{metrics['target_mean']:.3f} / "
            f"{metrics['target_std']:.3f} Å"
        )

        print(
            f"Predicted contacts <= 8 Å: "
            f"{metrics['predicted_contact_fraction']:.4f}"
        )

        print(
            f"Real contacts <= 8 Å: "
            f"{metrics['target_contact_fraction']:.4f}"
        )

        print(
            f"Predicted contact count: "
            f"{int(metrics['predicted_contact_count'])}"
        )

        print(
            f"Real contact count: "
            f"{int(metrics['target_contact_count'])}"
        )

        print(
            f"MAE on real contacts: "
            f"{metrics['contact_mae']:.3f} Å"
        )

        print(
            f"Pearson correlation: "
            f"{metrics['pearson']:.4f}"
        )

        print(
            f"Spearman correlation: "
            f"{metrics['spearman']:.4f}"
        )

        print(
            f"Top-L/5 contact precision: "
            f"{metrics['top_l5_precision']:.4f}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()