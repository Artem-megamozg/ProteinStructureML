from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.structure.reconstruction import (
    reconstruct_and_align,
)


THREE_LETTER = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def write_ca_pdb(
    sequence: str,
    coordinates: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for index, (
            amino_acid,
            coordinate,
        ) in enumerate(
            zip(
                sequence,
                coordinates,
            ),
            start=1,
        ):
            residue_name = (
                THREE_LETTER.get(
                    amino_acid,
                    "UNK",
                )
            )

            x, y, z = coordinate

            line = (
                f"ATOM  "
                f"{index:5d} "
                f" CA "
                f"{residue_name:>3s} "
                f"A"
                f"{index:4d}    "
                f"{x:8.3f}"
                f"{y:8.3f}"
                f"{z:8.3f}"
                f"  1.00"
                f" 20.00"
                f"           C\n"
            )

            file.write(
                line
            )

        file.write(
            "END\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    prediction_path = (
        Path("results/predictions_v2")
        / f"{args.sample}.npz"
    )

    target_path = (
        Path("data/processed/targets")
        / f"{args.sample}.npz"
    )

    feature_path = (
        Path("data/processed/features")
        / f"{args.sample}.npz"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction not found: "
            f"{prediction_path}"
        )

    if not target_path.exists():
        raise FileNotFoundError(
            f"Target not found: "
            f"{target_path}"
        )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Features not found: "
            f"{feature_path}"
        )

    prediction_data = np.load(
        prediction_path,
        allow_pickle=False,
    )

    target_data = np.load(
        target_path,
        allow_pickle=False,
    )

    feature_data = np.load(
        feature_path,
        allow_pickle=False,
    )

    distance_map = (
        prediction_data[
            "prediction"
        ]
    )

    target_coordinates = (
        target_data[
            "coordinates"
        ]
    )

    valid_residue_mask = (
        target_data[
            "valid_residue_mask"
        ]
    )

    sequence = str(
        feature_data[
            "sequence"
        ].item()
    )

    valid = (
        valid_residue_mask
        & np.all(
            np.isfinite(
                target_coordinates
            ),
            axis=1,
        )
    )

    valid_indices = np.flatnonzero(
        valid
    )

    if len(valid_indices) < 10:
        raise RuntimeError(
            "Not enough valid residues "
            "for reconstruction."
        )

    valid_distance_map = (
        distance_map[
            np.ix_(
                valid_indices,
                valid_indices,
            )
        ]
    )

    valid_coordinates = (
        target_coordinates[
            valid_indices
        ]
    )

    valid_sequence = "".join(
        sequence[index]
        for index in valid_indices
    )

    predicted_coordinates, rmsd = (
        reconstruct_and_align(
            distance_matrix=valid_distance_map,
            target_coordinates=valid_coordinates,
        )
    )

    output_dir = Path(
        "results/reconstructed"
    )

    output_path = (
        output_dir
        / f"{args.sample}_reconstructed.pdb"
    )

    write_ca_pdb(
        sequence=valid_sequence,
        coordinates=predicted_coordinates,
        path=output_path,
    )

    print()
    print("=" * 60)
    print("3D RECONSTRUCTION")
    print("=" * 60)

    print(
        f"Sample: {args.sample}"
    )

    print(
        f"Residues used: "
        f"{len(valid_indices)}"
    )

    print(
        f"C-alpha RMSD: "
        f"{rmsd:.4f} Å"
    )

    print(
        f"PDB saved: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()