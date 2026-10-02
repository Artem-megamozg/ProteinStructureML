from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load a protein dataset from CSV, Parquet, JSON or JSONL.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    extension = path.suffix.lower()

    if extension == ".csv":
        dataframe = pd.read_csv(path)

    elif extension == ".parquet":
        dataframe = pd.read_parquet(path)

    elif extension == ".json":
        with path.open("r", encoding="utf-8") as file:
            data: Any = json.load(file)

        dataframe = pd.DataFrame(data)

    elif extension == ".jsonl":
        dataframe = pd.read_json(path, lines=True)

    else:
        raise RuntimeError(f"Unable to load dataset: {path}")

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return dataframe


def save_dataset(dataframe: pd.DataFrame, path: str | Path) -> None:
    """
    Save a dataframe to CSV, Parquet, JSON or JSONL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    extension = path.suffix.lower()

    if extension == ".csv":
        dataframe.to_csv(path, index=False)

    elif extension == ".parquet":
        dataframe.to_parquet(path, index=False)

    elif extension == ".json":
        dataframe.to_json(
            path,
            orient="records",
            indent=2,
            force_ascii=False,
        )

    elif extension == ".jsonl":
        dataframe.to_json(
            path,
            orient="records",
            lines=True,
            force_ascii=False,
        )

    else:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )


def load_experimental_data(path: str | Path) -> pd.DataFrame:
    """
    Load experimental molecular measurements.

    This function is intentionally kept separate from the general dataset
    loader because experimental measurements may later require additional
    domain-specific processing.
    """
    dataframe = load_dataset(path)

    return dataframe