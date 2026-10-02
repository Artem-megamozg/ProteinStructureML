from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.bmrb_parser import (
    parse_bmrb_entry,
)
from src.data.pdb_parser import (
    parse_pdb_structure,
)


def _clean_pdb_id(value: object) -> str:
    value = str(value).strip().upper()

    if not value:
        raise ValueError("Empty PDB ID.")

    return value


def build_dataset(
    mapping_path: str | Path,
    bmrb_dir: str | Path,
    pdb_dir: str | Path,
    output_dir: str | Path,
    max_samples: int | None = None,
) -> pd.DataFrame:
    mapping_path = Path(mapping_path)
    bmrb_dir = Path(bmrb_dir)
    pdb_dir = Path(pdb_dir)
    output_dir = Path(output_dir)

    features_dir = output_dir / "features"
    targets_dir = output_dir / "targets"

    features_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping = pd.read_csv(
        mapping_path
    )

    required_columns = {
        "bmrb_id",
        "pdb_id",
    }

    missing_columns = (
        required_columns
        - set(mapping.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in mapping: "
            f"{sorted(missing_columns)}"
        )

    mapping = mapping.drop_duplicates(
        subset=[
            "bmrb_id",
            "pdb_id",
        ]
    )

    if max_samples is not None:
        mapping = mapping.head(max_samples)

    metadata = []

    bmrb_cache = {}

    for index, row in mapping.iterrows():
        bmrb_id = str(
            int(row["bmrb_id"])
        )

        pdb_id = _clean_pdb_id(
            row["pdb_id"]
        )

        print(
            f"[{index + 1}/{len(mapping)}] "
            f"BMRB {bmrb_id} + PDB {pdb_id}"
        )

        try:
            if bmrb_id not in bmrb_cache:
                bmrb_path = (
                    bmrb_dir
                    / f"bmr{bmrb_id}_3.str"
                )

                if not bmrb_path.exists():
                    bmrb_path = (
                        bmrb_dir
                        / f"bmr{bmrb_id}_21.str"
                    )

                bmrb_cache[bmrb_id] = (
                    parse_bmrb_entry(
                        bmrb_path
                    )
                )

            bmrb = bmrb_cache[bmrb_id]

            pdb_path = (
                pdb_dir
                / f"{pdb_id}.cif"
            )

            pdb = parse_pdb_structure(
                path=pdb_path,
                target_sequence=bmrb.sequence,
                pdb_id=pdb_id,
            )

            sample_id = (
                f"bmrb{bmrb_id}_pdb{pdb_id}"
            )

            target_mask = np.outer(
                pdb.valid_residue_mask,
                pdb.valid_residue_mask,
            ).astype(np.float32)

            np.fill_diagonal(
                target_mask,
                0.0,
            )

            np.savez_compressed(
                features_dir / f"{sample_id}.npz",
                sequence=np.array(
                    bmrb.sequence
                ),
                chemical_shifts=bmrb.chemical_shifts,
                chemical_shift_mask=(
                    bmrb.chemical_shift_mask
                ),
            )

            np.savez_compressed(
                targets_dir / f"{sample_id}.npz",
                coordinates=pdb.coordinates,
                distance_map=pdb.distance_map,
                target_mask=target_mask,
                valid_residue_mask=(
                    pdb.valid_residue_mask
                ),
            )

            shift_coverage = float(
                bmrb.chemical_shift_mask.mean()
            )

            target_coverage = float(
                pdb.valid_residue_mask.mean()
            )

            metadata.append(
                {
                    "sample_id": sample_id,
                    "bmrb_id": bmrb_id,
                    "pdb_id": pdb_id,
                    "chain_id": pdb.chain_id,
                    "entity_id": bmrb.entity_id,
                    "shift_list_id": bmrb.shift_list_id,
                    "sequence_length": len(
                        bmrb.sequence
                    ),
                    "chemical_shift_coverage": (
                        shift_coverage
                    ),
                    "structure_coverage": (
                        target_coverage
                    ),
                    "sequence_identity": (
                        pdb.identity
                    ),
                    "sequence_coverage": (
                        pdb.coverage
                    ),
                    "n_models": pdb.n_models,
                }
            )

        except Exception as exc:
            print(
                f"  SKIPPED: {exc}"
            )

    metadata_df = pd.DataFrame(
        metadata
    )

    metadata_path = (
        output_dir
        / "metadata.csv"
    )

    metadata_df.to_csv(
        metadata_path,
        index=False,
    )

    print()
    print(
        f"Successfully built "
        f"{len(metadata_df)} samples."
    )
    print(
        f"Metadata: {metadata_path}"
    )

    return metadata_df