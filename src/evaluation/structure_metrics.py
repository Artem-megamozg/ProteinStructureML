from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def masked_regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """
    Calculate regression metrics only on valid residue pairs.
    """
    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    valid = mask > 0

    if not np.any(valid):
        raise ValueError(
            "No valid entries in target mask."
        )

    prediction_valid = prediction[valid]
    target_valid = target[valid]

    error = (
        prediction_valid
        - target_valid
    )

    mse = float(
        np.mean(error ** 2)
    )

    rmse = float(
        np.sqrt(mse)
    )

    mae = float(
        np.mean(np.abs(error))
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def contact_metrics_from_distances(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    contact_threshold: float = 8.0,
    min_sequence_separation: int = 3,
) -> dict[str, float]:
    """
    Calculate contact metrics from predicted distances.
    """
    prediction = np.asarray(
        prediction
    )

    target = np.asarray(
        target
    )

    mask = np.asarray(
        mask
    )

    length = prediction.shape[0]

    indices = np.arange(
        length
    )

    separation = np.abs(
        indices[:, None]
        - indices[None, :]
    )

    valid_pairs = (
        (mask > 0)
        & (separation >= min_sequence_separation)
    )

    prediction_contacts = (
        prediction <= contact_threshold
    )

    target_contacts = (
        target <= contact_threshold
    )

    return _binary_contact_metrics(
        prediction_contacts=prediction_contacts,
        target_contacts=target_contacts,
        valid_pairs=valid_pairs,
    )


def contact_metrics_from_probabilities(
    probability: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
    contact_distance: float = 8.0,
    min_sequence_separation: int = 3,
) -> dict[str, float]:
    """
    Calculate contact metrics from predicted contact probabilities.
    """
    probability = np.asarray(
        probability,
        dtype=np.float32,
    )

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    length = probability.shape[0]

    indices = np.arange(
        length
    )

    separation = np.abs(
        indices[:, None]
        - indices[None, :]
    )

    valid_pairs = (
        (mask > 0)
        & (separation >= min_sequence_separation)
    )

    prediction_contacts = (
        probability >= threshold
    )

    target_contacts = (
        target <= contact_distance
    )

    return _binary_contact_metrics(
        prediction_contacts=prediction_contacts,
        target_contacts=target_contacts,
        valid_pairs=valid_pairs,
        probabilities=probability,
    )


def _binary_contact_metrics(
    prediction_contacts: np.ndarray,
    target_contacts: np.ndarray,
    valid_pairs: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    prediction_contacts = (
        prediction_contacts
        & valid_pairs
    )

    target_contacts = (
        target_contacts
        & valid_pairs
    )

    true_positive = np.logical_and(
        prediction_contacts,
        target_contacts,
    ).sum()

    false_positive = np.logical_and(
        prediction_contacts,
        ~target_contacts,
        valid_pairs,
    ).sum()

    false_negative = np.logical_and(
        ~prediction_contacts,
        target_contacts,
        valid_pairs,
    ).sum()

    precision = (
        true_positive
        / max(
            true_positive
            + false_positive,
            1,
        )
    )

    recall = (
        true_positive
        / max(
            true_positive
            + false_negative,
            1,
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        / max(
            precision + recall,
            1e-12,
        )
    )

    result = {
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
    }

    if probabilities is not None:
        valid_probabilities = probabilities[
            valid_pairs
        ].reshape(-1)

        valid_targets = target_contacts[
            valid_pairs
        ].astype(np.int32).reshape(-1)

        result["average_precision"] = float(
            average_precision_score(
                valid_targets,
                valid_probabilities,
            )
        )

        result.update(
            _top_l_precision(
                probabilities=probabilities,
                target_contacts=target_contacts,
                valid_pairs=valid_pairs,
            )
        )

    return result


def _top_l_precision(
    probabilities: np.ndarray,
    target_contacts: np.ndarray,
    valid_pairs: np.ndarray,
) -> dict[str, float]:
    length = probabilities.shape[0]

    valid_positions = np.argwhere(
        valid_pairs
    )

    if len(valid_positions) == 0:
        return {
            "top_l5_precision": 0.0,
            "top_l10_precision": 0.0,
            "top_l20_precision": 0.0,
        }

    scores = probabilities[
        valid_pairs
    ]

    targets = target_contacts[
        valid_pairs
    ].astype(bool)

    order = np.argsort(
        scores
    )[::-1]

    ordered_targets = targets[
        order
    ]

    total_pairs_l5 = max(
        length // 5,
        1,
    )

    total_pairs_l10 = max(
        length // 10,
        1,
    )

    total_pairs_l20 = max(
        length // 20,
        1,
    )

    top_l5 = ordered_targets[
        :min(
            total_pairs_l5,
            len(ordered_targets),
        )
    ]

    top_l10 = ordered_targets[
        :min(
            total_pairs_l10,
            len(ordered_targets),
        )
    ]

    top_l20 = ordered_targets[
        :min(
            total_pairs_l20,
            len(ordered_targets),
        )
    ]

    return {
        "top_l5_precision": float(
            top_l5.mean()
            if len(top_l5)
            else 0.0
        ),
        "top_l10_precision": float(
            top_l10.mean()
            if len(top_l10)
            else 0.0
        ),
        "top_l20_precision": float(
            top_l20.mean()
            if len(top_l20)
            else 0.0
        ),
    }