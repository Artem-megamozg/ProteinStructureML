from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def long_range_metrics(
    probability: np.ndarray,
    target_distance: np.ndarray,
    target_mask: np.ndarray,
    contact_threshold: float = 8.0,
    min_sequence_separation: int = 24,
) -> dict[str, float] | None:
    """
    Evaluate long-range residue contact prediction.

    Only unique residue pairs i < j are evaluated.
    Long-range pairs have sequence separation >= 24.
    """

    probability = np.asarray(
        probability,
        dtype=np.float32,
    )

    target_distance = np.asarray(
        target_distance,
        dtype=np.float32,
    )

    target_mask = np.asarray(
        target_mask,
        dtype=np.float32,
    )

    length = probability.shape[0]

    indices_i, indices_j = np.triu_indices(
        length,
        k=1,
    )

    sequence_separation = (
        indices_j - indices_i
    )

    valid = (
        (sequence_separation >= min_sequence_separation)
        & (
            target_mask[
                indices_i,
                indices_j,
            ]
            > 0
        )
    )

    if not np.any(valid):
        return None

    scores = probability[
        indices_i[valid],
        indices_j[valid],
    ]

    targets = (
        target_distance[
            indices_i[valid],
            indices_j[valid],
        ]
        <= contact_threshold
    )

    if not np.any(targets):
        return {
            "num_pairs": float(len(targets)),
            "num_contacts": 0.0,
            "average_precision": 0.0,
            "top_l5_precision": 0.0,
            "top_l10_precision": 0.0,
            "top_l20_precision": 0.0,
        }

    order = np.argsort(
        scores
    )[::-1]

    sorted_targets = targets[
        order
    ]

    ap = float(
        average_precision_score(
            targets.astype(np.int32),
            scores,
        )
    )

    def precision_at_k(
        k: int,
    ) -> float:
        k = min(
            k,
            len(sorted_targets),
        )

        if k <= 0:
            return 0.0

        return float(
            sorted_targets[:k].mean()
        )

    l5_k = max(
        length // 5,
        1,
    )

    l10_k = max(
        length // 10,
        1,
    )

    l20_k = max(
        length // 20,
        1,
    )

    return {
        "num_pairs": float(
            len(targets)
        ),
        "num_contacts": float(
            targets.sum()
        ),
        "average_precision": ap,
        "top_l5_precision": precision_at_k(
            l5_k
        ),
        "top_l10_precision": precision_at_k(
            l10_k
        ),
        "top_l20_precision": precision_at_k(
            l20_k
        ),
    }