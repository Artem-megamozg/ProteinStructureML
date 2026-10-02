from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pynmrstar


CHEMICAL_SHIFT_ATOMS = ("H", "HA", "CA", "CB", "C", "N")


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass
class BMRBRecord:
    bmrb_id: str
    sequence: str
    chemical_shifts: np.ndarray
    chemical_shift_mask: np.ndarray
    entity_id: str
    shift_list_id: str


def _to_float(value: object) -> float | None:
    if value is None:
        return None

    value = str(value).strip()

    if not value or value in {".", "?"}:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _three_to_one(comp_id: object) -> str:
    comp_id = str(comp_id).strip().upper()

    return THREE_TO_ONE.get(
        comp_id,
        "X",
    )


def _clean_sequence(sequence: str) -> str:
    return "".join(
        char
        for char in str(sequence).upper()
        if char.isalpha()
    )


def _extract_sequences(
    entry: pynmrstar.Entry,
) -> dict[str, dict[int, str]]:
    sequences: dict[str, dict[int, str]] = {}

    for loop in entry.get_loops_by_category(
        "Entity_poly_seq"
    ):
        try:
            rows = loop.get_tag(
                [
                    "Entity_ID",
                    "Num",
                    "Mon_ID",
                ]
            )
        except KeyError:
            continue

        for row in rows:
            entity_id = str(row[0]).strip()

            try:
                position = int(row[1])
            except (TypeError, ValueError):
                continue

            amino_acid = _three_to_one(
                row[2]
            )

            sequences.setdefault(
                entity_id,
                {},
            )

            sequences[
                entity_id
            ][position] = amino_acid

    return sequences


def _extract_entity_sequences(
    entry: pynmrstar.Entry,
) -> dict[str, str]:
    sequences: dict[str, str] = {}

    for saveframe in entry.get_saveframes_by_category(
        "entity"
    ):
        try:
            entity_id = str(
                saveframe.get_tag("ID")[0]
            ).strip()
        except (KeyError, IndexError):
            continue

        sequence = None

        possible_tags = (
            "Polymer_seq_one_letter_code_can",
            "Polymer_seq_one_letter_code",
        )

        for tag in possible_tags:
            try:
                values = saveframe.get_tag(tag)

                if values:
                    value = values[0]

                    if value not in {
                        None,
                        ".",
                        "?",
                    }:
                        sequence = _clean_sequence(
                            value
                        )
                        break

            except KeyError:
                continue

        if sequence:
            sequences[
                entity_id
            ] = sequence

    return sequences


def _extract_shift_records(
    entry: pynmrstar.Entry,
) -> dict[
    str,
    dict[
        str,
        dict[
            tuple[int, str],
            list[float],
        ],
    ],
]:
    grouped: dict[
        str,
        dict[
            str,
            dict[
                tuple[int, str],
                list[float],
            ],
        ],
    ] = {}

    for loop in entry.get_loops_by_category(
        "Atom_chem_shift"
    ):
        try:
            rows = loop.get_tag(
                [
                    "Entity_ID",
                    "Comp_index_ID",
                    "Atom_ID",
                    "Val",
                    "Assigned_chem_shift_list_ID",
                ]
            )

        except KeyError:
            try:
                rows = loop.get_tag(
                    [
                        "Entity_ID",
                        "Comp_index_ID",
                        "Atom_ID",
                        "Val",
                    ]
                )

                rows = [
                    list(row) + ["."]
                    for row in rows
                ]

            except KeyError:
                continue

        for row in rows:
            if len(row) < 4:
                continue

            entity_id = str(
                row[0]
            ).strip()

            try:
                residue_index = int(
                    row[1]
                )
            except (TypeError, ValueError):
                continue

            atom_id = str(
                row[2]
            ).strip().upper()

            value = _to_float(
                row[3]
            )

            shift_list_id = (
                str(row[4]).strip()
                if len(row) > 4
                else "1"
            )

            if value is None:
                continue

            if atom_id not in CHEMICAL_SHIFT_ATOMS:
                continue

            grouped.setdefault(
                shift_list_id,
                {},
            ).setdefault(
                entity_id,
                {},
            ).setdefault(
                (
                    residue_index,
                    atom_id,
                ),
                [],
            ).append(value)

    return grouped


def parse_bmrb_entry(
    path: str | Path,
) -> BMRBRecord:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"BMRB file not found: {path}"
        )

    entry = pynmrstar.Entry.from_file(
        str(path),
        convert_data_types=True,
    )

    bmrb_id = str(
        entry.entry_id
    )

    sequences = _extract_sequences(
        entry
    )

    entity_sequences = _extract_entity_sequences(
        entry
    )

    grouped_shifts = _extract_shift_records(
        entry
    )

    if not grouped_shifts:
        raise ValueError(
            f"No usable chemical shifts found "
            f"in BMRB entry {bmrb_id}."
        )

    best_shift_list_id = None
    best_entity_id = None
    best_coverage = -1

    for shift_list_id, entities in grouped_shifts.items():
        for entity_id, records in entities.items():
            residue_count = len(
                {
                    residue
                    for residue, _ in records
                }
            )

            if (
                entity_id in sequences
                and len(sequences[entity_id]) > 0
            ):
                sequence_length = len(
                    sequences[entity_id]
                )
            elif entity_id in entity_sequences:
                sequence_length = len(
                    entity_sequences[entity_id]
                )
            else:
                sequence_length = residue_count

            coverage = (
                residue_count
                / max(sequence_length, 1)
            )

            score = (
                sequence_length * 0.2
                + residue_count * 0.8
                + coverage * 100
            )

            if score > best_coverage:
                best_coverage = score
                best_shift_list_id = shift_list_id
                best_entity_id = entity_id

    if (
        best_shift_list_id is None
        or best_entity_id is None
    ):
        raise ValueError(
            f"Unable to select chemical shift set "
            f"for BMRB {bmrb_id}."
        )

    if best_entity_id in sequences:
        residue_sequence = sequences[
            best_entity_id
        ]

        sequence_length = max(
            residue_sequence
        )

        sequence = "".join(
            residue_sequence.get(
                index,
                "X",
            )
            for index in range(
                1,
                sequence_length + 1,
            )
        )

    elif best_entity_id in entity_sequences:
        sequence = entity_sequences[
            best_entity_id
        ]

        sequence_length = len(
            sequence
        )

    else:
        selected_records = grouped_shifts[
            best_shift_list_id
        ][
            best_entity_id
        ]

        sequence_length = max(
            residue
            for residue, _ in selected_records
        )

        sequence = "X" * sequence_length

    selected_records = grouped_shifts[
        best_shift_list_id
    ][
        best_entity_id
    ]

    chemical_shifts = np.full(
        (
            sequence_length,
            len(CHEMICAL_SHIFT_ATOMS),
        ),
        np.nan,
        dtype=np.float32,
    )

    chemical_shift_mask = np.zeros(
        (
            sequence_length,
            len(CHEMICAL_SHIFT_ATOMS),
        ),
        dtype=np.float32,
    )

    atom_to_index = {
        atom: index
        for index, atom in enumerate(
            CHEMICAL_SHIFT_ATOMS
        )
    }

    for (
        residue_index,
        atom,
    ), values in selected_records.items():

        if not (
            1
            <= residue_index
            <= sequence_length
        ):
            continue

        atom_index = atom_to_index[
            atom
        ]

        value = float(
            np.mean(values)
        )

        chemical_shifts[
            residue_index - 1,
            atom_index,
        ] = value

        chemical_shift_mask[
            residue_index - 1,
            atom_index,
        ] = 1.0

    if (
        not sequence
        or set(sequence) == {"X"}
    ):
        raise ValueError(
            f"Unable to extract a valid "
            f"protein sequence from BMRB {bmrb_id}."
        )

    return BMRBRecord(
        bmrb_id=bmrb_id,
        sequence=sequence,
        chemical_shifts=chemical_shifts,
        chemical_shift_mask=chemical_shift_mask,
        entity_id=best_entity_id,
        shift_list_id=best_shift_list_id,
    )