from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """
    Calculate standard regression metrics.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: "
            f"{prediction.shape} != {target.shape}"
        )

    prediction_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)

    return {
        "mse": float(
            mean_squared_error(
                target_flat,
                prediction_flat,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    target_flat,
                    prediction_flat,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                target_flat,
                prediction_flat,
            )
        ),
        "r2": float(
            r2_score(
                target_flat,
                prediction_flat,
            )
        ),
    }


def contact_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calculate binary contact prediction metrics.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    prediction_binary = prediction >= threshold
    target_binary = target >= threshold

    true_positive = np.logical_and(
        prediction_binary,
        target_binary,
    ).sum()

    false_positive = np.logical_and(
        prediction_binary,
        ~target_binary,
    ).sum()

    false_negative = np.logical_and(
        ~prediction_binary,
        target_binary,
    ).sum()

    precision = true_positive / max(
        true_positive + false_positive,
        1,
    )

    recall = true_positive / max(
        true_positive + false_negative,
        1,
    )

    f1 = 2 * precision * recall / max(
        precision + recall,
        1e-12,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }