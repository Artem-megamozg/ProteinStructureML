from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import MMCIFParser


@dataclass
class PDBChain:
    chain_id: str
    sequence: str
    coordinates: np.ndarray


@dataclass
class PDBStructure:
    pdb_id: str
    chain_id: str
    sequence: str
    coordinates: np.ndarray
    distance_map: np.ndarray
    valid_residue_mask: np.ndarray
    n_models: int
    identity: float
    coverage: float


def _three_to_one(residue_name: str) -> str:
    residue_name = residue_name.strip().capitalize()
    return protein_letters_3to1.get(residue_name, "X")


def extract_chains(
    structure,
) -> list[PDBChain]:
    chains: list[PDBChain] = []

    model = next(structure.get_models())

    for chain in model.get_chains():
        sequence = []
        coordinates = []

        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue

            if "CA" not in residue:
                continue

            sequence.append(
                _three_to_one(residue.resname)
            )

            coordinates.append(
                residue["CA"].coord.astype(np.float32)
            )

        if not sequence:
            continue

        chains.append(
            PDBChain(
                chain_id=str(chain.id),
                sequence="".join(sequence),
                coordinates=np.asarray(
                    coordinates,
                    dtype=np.float32,
                ),
            )
        )

    return chains


def _align_sequences(
    reference: str,
    candidate: str,
):
    aligner = PairwiseAligner()

    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5

    return aligner.align(
        reference,
        candidate,
    )[0]


def _alignment_statistics(
    alignment,
    reference: str,
    candidate: str,
) -> tuple[float, float]:
    matches = 0
    aligned_positions = 0

    for reference_block, candidate_block in zip(
        alignment.aligned[0],
        alignment.aligned[1],
    ):
        reference_start, reference_end = reference_block
        candidate_start, candidate_end = candidate_block

        block_length = min(
            reference_end - reference_start,
            candidate_end - candidate_start,
        )

        for offset in range(block_length):
            if (
                reference[reference_start + offset]
                == candidate[candidate_start + offset]
            ):
                matches += 1

            aligned_positions += 1

    if not aligned_positions:
        return 0.0, 0.0

    identity = matches / aligned_positions
    coverage = aligned_positions / len(reference)

    return identity, coverage


def _map_coordinates(
    alignment,
    reference_length: int,
    candidate_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mapped_coordinates = np.full(
        (
            reference_length,
            3,
        ),
        np.nan,
        dtype=np.float32,
    )

    valid_mask = np.zeros(
        reference_length,
        dtype=bool,
    )

    for reference_block, candidate_block in zip(
        alignment.aligned[0],
        alignment.aligned[1],
    ):
        reference_start, reference_end = reference_block
        candidate_start, candidate_end = candidate_block

        block_length = min(
            reference_end - reference_start,
            candidate_end - candidate_start,
        )

        for offset in range(block_length):
            reference_index = (
                reference_start + offset
            )

            candidate_index = (
                candidate_start + offset
            )

            if candidate_index >= len(candidate_coordinates):
                continue

            mapped_coordinates[
                reference_index
            ] = candidate_coordinates[
                candidate_index
            ]

            valid_mask[
                reference_index
            ] = True

    return mapped_coordinates, valid_mask


def _build_distance_map(
    coordinates: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    length = len(coordinates)

    distance_map = np.full(
        (
            length,
            length,
        ),
        np.nan,
        dtype=np.float32,
    )

    valid_indices = np.flatnonzero(valid_mask)

    if len(valid_indices) == 0:
        return distance_map

    valid_coordinates = coordinates[
        valid_indices
    ]

    differences = (
        valid_coordinates[:, None, :]
        - valid_coordinates[None, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=-1,
    ).astype(np.float32)

    for i, index_i in enumerate(valid_indices):
        for j, index_j in enumerate(valid_indices):
            distance_map[
                index_i,
                index_j,
            ] = distances[i, j]

    return distance_map


def parse_pdb_structure(
    path: str | Path,
    target_sequence: str,
    pdb_id: str | None = None,
    min_identity: float = 0.70,
    min_coverage: float = 0.70,
) -> PDBStructure:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDB file not found: {path}"
        )

    parser = MMCIFParser(
        QUIET=True,
    )

    structure_id = (
        pdb_id
        if pdb_id is not None
        else path.stem
    )

    structure = parser.get_structure(
        structure_id,
        path,
    )

    chains = extract_chains(structure)

    if not chains:
        raise ValueError(
            f"No protein chains with C-alpha atoms found in {path}."
        )

    best_chain = None
    best_alignment = None
    best_identity = -1.0
    best_coverage = -1.0
    best_score = -np.inf

    for chain in chains:
        alignment = _align_sequences(
            target_sequence,
            chain.sequence,
        )

        identity, coverage = _alignment_statistics(
            alignment,
            target_sequence,
            chain.sequence,
        )

        score = (
            identity * 0.6
            + coverage * 0.4
        )

        if score > best_score:
            best_score = score
            best_chain = chain
            best_alignment = alignment
            best_identity = identity
            best_coverage = coverage

    if (
        best_chain is None
        or best_alignment is None
    ):
        raise ValueError(
            f"Unable to match a PDB chain to target sequence."
        )

    if best_identity < min_identity:
        raise ValueError(
            f"Best chain identity too low: "
            f"{best_identity:.3f} < {min_identity:.3f}"
        )

    if best_coverage < min_coverage:
        raise ValueError(
            f"Best chain coverage too low: "
            f"{best_coverage:.3f} < {min_coverage:.3f}"
        )

    mapped_coordinates, valid_mask = _map_coordinates(
        best_alignment,
        len(target_sequence),
        best_chain.coordinates,
    )

    distance_map = _build_distance_map(
        mapped_coordinates,
        valid_mask,
    )

    n_models = len(
        list(structure.get_models())
    )

    return PDBStructure(
        pdb_id=str(structure_id).upper(),
        chain_id=best_chain.chain_id,
        sequence=target_sequence,
        coordinates=mapped_coordinates,
        distance_map=distance_map,
        valid_residue_mask=valid_mask,
        n_models=n_models,
        identity=best_identity,
        coverage=best_coverage,
    )