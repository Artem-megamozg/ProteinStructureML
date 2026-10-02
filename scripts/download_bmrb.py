from __future__ import annotations

import argparse
import ssl
import time
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd


QUERY_URL = (
    "https://bmrb.io/search/query_grid/"
    "?data_types%5B%5D=pdb_ids"
    "&output=csv"
    "&polymer_join_type=AND"
)

RAW_DIR = Path("data/raw/bmrb")
MAPPING_PATH = RAW_DIR / "bmrb_pdb_mapping.csv"

USER_AGENT = (
    "ProteinStructureML/0.1 "
    "(research dataset downloader)"
)

SSL_CONTEXT = ssl._create_unverified_context()


def download_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
        },
    )

    with urlopen(
        request,
        timeout=60,
        context=SSL_CONTEXT,
    ) as response:
        return response.read()


def normalize_pdb_ids(
    value: object,
) -> list[str]:
    if value is None:
        return []

    value = str(value).strip()

    if not value or value in {
        "nan",
        "None",
        ".",
    }:
        return []

    normalized = (
        value
        .replace(",", " ")
        .replace(";", " ")
    )

    pdb_ids = []

    for token in normalized.split():
        token = token.strip().upper()

        if len(token) != 4:
            continue

        if not token.isalnum():
            continue

        pdb_ids.append(token)

    return pdb_ids


def build_mapping(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe.columns = [
        str(column)
        .strip()
        .strip('"')
        for column in dataframe.columns
    ]

    required_columns = {
        "Entry_ID",
        "pdb_ids",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Unexpected BMRB CSV format. "
            f"Missing columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: "
            f"{list(dataframe.columns)}"
        )

    rows = []

    for _, row in dataframe.iterrows():
        bmrb_value = str(
            row["Entry_ID"]
        ).strip()

        if not bmrb_value:
            continue

        try:
            bmrb_id = str(
                int(float(bmrb_value))
            )
        except ValueError:
            continue

        pdb_ids = normalize_pdb_ids(
            row["pdb_ids"]
        )

        for pdb_id in pdb_ids:
            rows.append(
                {
                    "bmrb_id": bmrb_id,
                    "pdb_id": pdb_id,
                }
            )

    mapping = (
        pd.DataFrame(rows)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if mapping.empty:
        raise RuntimeError(
            "BMRB query returned no valid "
            "BMRB-PDB pairs."
        )

    mapping = (
        mapping
        .drop_duplicates(
            subset=["bmrb_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return mapping


def download_bmrb_entry(
    bmrb_id: str,
    output_dir: Path,
) -> Path | None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    possible_files = [
        output_dir
        / f"bmr{bmrb_id}_3.str",
        output_dir
        / f"bmr{bmrb_id}_21.str",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path

    urls = [
        (
            "https://bmrb.io/ftp/pub/bmrb/"
            f"entry_directories/bmr{bmrb_id}/"
            f"bmr{bmrb_id}_3.str"
        ),
        (
            "https://bmrb.io/ftp/pub/bmrb/"
            f"entry_directories/bmr{bmrb_id}/"
            f"bmr{bmrb_id}_21.str"
        ),
    ]

    for url, target in zip(
        urls,
        possible_files,
    ):
        try:
            content = download_bytes(url)

            target.write_bytes(
                content
            )

            return target

        except HTTPError:
            continue

    return None


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help=(
            "Number of unique BMRB entries "
            "to select."
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between downloads.",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError(
            "--limit must be greater than zero."
        )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading BMRB-PDB mapping..."
    )

    csv_content = download_bytes(
        QUERY_URL
    )

    dataframe = pd.read_csv(
        BytesIO(csv_content)
    )

    print(
        f"Received {len(dataframe)} "
        "BMRB records from BMRB."
    )

    mapping = build_mapping(
        dataframe
    )

    mapping = mapping.head(
        args.limit
    )

    mapping.to_csv(
        MAPPING_PATH,
        index=False,
    )

    print(
        f"Selected {len(mapping)} "
        "unique BMRB entries."
    )

    downloaded = 0

    for index, row in mapping.iterrows():
        bmrb_id = str(
            row["bmrb_id"]
        )

        print(
            f"[{index + 1}/{len(mapping)}] "
            f"Downloading BMRB {bmrb_id}..."
        )

        result = download_bmrb_entry(
            bmrb_id=bmrb_id,
            output_dir=RAW_DIR,
        )

        if result is not None:
            downloaded += 1

            print(
                f"  saved: {result.name}"
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
        f"Downloaded {downloaded}/"
        f"{len(mapping)} BMRB entries."
    )

    print(
        f"Mapping: {MAPPING_PATH}"
    )


if __name__ == "__main__":
    main()