from __future__ import annotations

import torch


def protein_collate_fn(
    batch: list[dict],
) -> dict[str, torch.Tensor | list[str]]:
    """
    Pad variable-length protein samples into a batch.
    """
    batch_size = len(batch)

    lengths = torch.tensor(
        [
            len(sample["sequence"])
            for sample in batch
        ],
        dtype=torch.long,
    )

    max_length = int(
        lengths.max().item()
    )

    chemical_shift_dim = batch[0][
        "chemical_shifts"
    ].shape[1]

    sequence_tokens = torch.zeros(
        (
            batch_size,
            max_length,
        ),
        dtype=torch.long,
    )

    chemical_shifts = torch.zeros(
        (
            batch_size,
            max_length,
            chemical_shift_dim,
        ),
        dtype=torch.float32,
    )

    chemical_shift_mask = torch.zeros(
        (
            batch_size,
            max_length,
            chemical_shift_dim,
        ),
        dtype=torch.float32,
    )

    distance_maps = torch.zeros(
        (
            batch_size,
            max_length,
            max_length,
        ),
        dtype=torch.float32,
    )

    target_masks = torch.zeros(
        (
            batch_size,
            max_length,
            max_length,
        ),
        dtype=torch.float32,
    )

    residue_mask = torch.zeros(
        (
            batch_size,
            max_length,
        ),
        dtype=torch.float32,
    )

    sample_ids = []
    sequences = []

    for index, sample in enumerate(batch):
        length = lengths[index].item()

        sequence_tokens[
            index,
            :length,
        ] = sample["sequence_tokens"]

        chemical_shifts[
            index,
            :length,
        ] = sample["chemical_shifts"]

        chemical_shift_mask[
            index,
            :length,
        ] = sample["chemical_shift_mask"]

        distance_maps[
            index,
            :length,
            :length,
        ] = sample["distance_map"]

        target_masks[
            index,
            :length,
            :length,
        ] = sample["target_mask"]

        residue_mask[
            index,
            :length,
        ] = 1.0

        sample_ids.append(
            sample["sample_id"]
        )

        sequences.append(
            sample["sequence"]
        )

    return {
        "sample_ids": sample_ids,
        "sequences": sequences,
        "sequence_tokens": sequence_tokens,
        "chemical_shifts": chemical_shifts,
        "chemical_shift_mask": chemical_shift_mask,
        "distance_maps": distance_maps,
        "target_masks": target_masks,
        "residue_mask": residue_mask,
        "lengths": lengths,
    }