from __future__ import annotations

import random

import pandas as pd


def create_sequence_groups(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign the same group to identical protein sequences.
    """
    result = metadata.copy()

    group_mapping: dict[str, int] = {}
    groups = []

    next_group_id = 0

    for sequence in result["sequence_key"]:
        if sequence not in group_mapping:
            group_mapping[sequence] = next_group_id
            next_group_id += 1

        groups.append(
            group_mapping[sequence]
        )

    result["sequence_group"] = groups

    return result


def split_by_sequence_group(
    metadata: pd.DataFrame,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Split samples into train/validation/test without
    putting identical sequences into different splits.
    """
    total_ratio = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            "Split ratios must sum to 1.0."
        )

    if "sequence_group" not in metadata.columns:
        raise ValueError(
            "Column 'sequence_group' is required."
        )

    groups = list(
        metadata["sequence_group"]
        .unique()
    )

    rng = random.Random(
        random_state
    )

    rng.shuffle(groups)

    total_samples = len(metadata)

    train_target = (
        total_samples
        * train_ratio
    )

    validation_target = (
        total_samples
        * validation_ratio
    )

    train_groups = []
    validation_groups = []
    test_groups = []

    train_count = 0
    validation_count = 0

    for group in groups:
        group_size = int(
            (
                metadata["sequence_group"]
                == group
            ).sum()
        )

        if (
            train_count < train_target
        ):
            train_groups.append(group)
            train_count += group_size

        elif (
            validation_count
            < validation_target
        ):
            validation_groups.append(group)
            validation_count += group_size

        else:
            test_groups.append(group)

    def assign_split(group: int) -> str:
        if group in train_groups:
            return "train"

        if group in validation_groups:
            return "validation"

        return "test"

    result = metadata.copy()

    result["split"] = (
        result["sequence_group"]
        .map(assign_split)
    )

    return result