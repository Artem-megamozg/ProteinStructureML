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

    prediction_path = (
        Path("results/predictions")
        / f"{args.sample}.npz"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction not found: "
            f"{prediction_path}"
        )

    data = np.load(
        prediction_path,
        allow_pickle=False,
    )

    prediction = data[
        "prediction"
    ]

    target = data[
        "target"
    ]

    target_mask = data[
        "target_mask"
    ]

    length = prediction.shape[0]

    prediction_display = prediction.copy()
    target_display = target.copy()

    prediction_display[
        target_mask == 0
    ] = np.nan

    target_display[
        target_mask == 0
    ] = np.nan

    error = np.abs(
        prediction_display
        - target_display
    )

    figure = plt.figure(
        figsize=(16, 5)
    )

    axes = [
        figure.add_subplot(
            1,
            3,
            1,
        ),
        figure.add_subplot(
            1,
            3,
            2,
        ),
        figure.add_subplot(
            1,
            3,
            3,
        ),
    ]

    axes[0].imshow(
        target_display,
        interpolation="nearest",
    )

    axes[0].set_title(
        "Experimental distance map"
    )

    axes[1].imshow(
        prediction_display,
        interpolation="nearest",
    )

    axes[1].set_title(
        "Predicted distance map"
    )

    axes[2].imshow(
        error,
        interpolation="nearest",
    )

    axes[2].set_title(
        "Absolute error"
    )

    for axis in axes:
        axis.set_xlabel(
            "Residue index"
        )
        axis.set_ylabel(
            "Residue index"
        )

    figure.suptitle(
        f"{args.sample} | L={length}"
    )

    figure.tight_layout()

    output_dir = Path(
        "results/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{args.sample}_prediction.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()