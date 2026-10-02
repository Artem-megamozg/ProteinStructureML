from __future__ import annotations

from dataclasses import dataclass

from Bio.Align import PairwiseAligner


@dataclass
class SequenceSimilarity:
    identity: float
    coverage_a: float
    coverage_b: float


def calculate_sequence_similarity(
    sequence_a: str,
    sequence_b: str,
) -> SequenceSimilarity:
    sequence_a = str(sequence_a).upper()
    sequence_b = str(sequence_b).upper()

    aligner = PairwiseAligner()

    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(
        sequence_a,
        sequence_b,
    )[0]

    matches = 0
    aligned_a = 0
    aligned_b = 0
    aligned_positions = 0

    for block_a, block_b in zip(
        alignment.aligned[0],
        alignment.aligned[1],
    ):
        start_a, end_a = block_a
        start_b, end_b = block_b

        block_length = min(
            end_a - start_a,
            end_b - start_b,
        )

        for offset in range(block_length):
            if (
                sequence_a[start_a + offset]
                == sequence_b[start_b + offset]
            ):
                matches += 1

            aligned_positions += 1

        aligned_a += end_a - start_a
        aligned_b += end_b - start_b

    if aligned_positions == 0:
        return SequenceSimilarity(
            identity=0.0,
            coverage_a=0.0,
            coverage_b=0.0,
        )

    return SequenceSimilarity(
        identity=(
            matches
            / aligned_positions
        ),
        coverage_a=(
            aligned_a
            / max(len(sequence_a), 1)
        ),
        coverage_b=(
            aligned_b
            / max(len(sequence_b), 1)
        ),
    )