from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd


MAPPING_PATH = Path(
    "data/raw/bmrb/bmrb_pdb_mapping.csv"
)

PDB_DIR = Path(
    "data/raw/pdb"
)

USER_AGENT = (
    "ProteinStructureML/0.1 "
    "(research dataset downloader)"
)


def download_file(
    url: str,
    target: Path,
) -> bool:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
        },
    )

    try:
        with urlopen(
            request,
            timeout=60,
        ) as response:
            target.write_bytes(
                response.read()
            )

        return True

    except HTTPError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between downloads.",
    )

    args = parser.parse_args()

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Mapping not found: {MAPPING_PATH}. "
            "Run download_bmrb.py first."
        )

    PDB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping = pd.read_csv(
        MAPPING_PATH
    )

    pdb_ids = sorted(
        set(
            mapping["pdb_id"]
            .astype(str)
            .str.upper()
        )
    )

    downloaded = 0

    for index, pdb_id in enumerate(
        pdb_ids,
        start=1,
    ):
        target = (
            PDB_DIR
            / f"{pdb_id}.cif"
        )

        if target.exists():
            print(
                f"[{index}/{len(pdb_ids)}] "
                f"{pdb_id}: already exists"
            )
            downloaded += 1
            continue

        url = (
            "https://files.rcsb.org/"
            f"download/{pdb_id}.cif"
        )

        print(
            f"[{index}/{len(pdb_ids)}] "
            f"Downloading {pdb_id}..."
        )

        if download_file(
            url,
            target,
        ):
            downloaded += 1
            print(
                f"  saved: {target}"
            )
        else:
            print(
                "  failed"
            )

        time.sleep(
            args.sleep
        )

    print()
    print(
        f"Available PDB files: "
        f"{downloaded}/{len(pdb_ids)}"
    )


if __name__ == "__main__":
    main()