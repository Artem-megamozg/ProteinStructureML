from __future__ import annotations

from src.data.shift_scaling import (
    fit_shift_statistics,
    save_shift_statistics,
)


def main() -> None:
    statistics = fit_shift_statistics(
        index_path=(
            "data/processed/"
            "dataset_index.csv"
        ),
        processed_dir=(
            "data/processed"
        ),
        split="train",
    )

    output_path = (
        "data/processed/"
        "chemical_shift_stats.json"
    )

    save_shift_statistics(
        statistics,
        output_path,
    )

    print(
        f"Saved chemical shift statistics "
        f"to: {output_path}"
    )

    for atom, mean, std, count in zip(
        statistics["atoms"],
        statistics["mean"],
        statistics["std"],
        statistics["count"],
    ):
        print(
            f"{atom:>2}: "
            f"mean={mean:8.3f} | "
            f"std={std:8.3f} | "
            f"n={int(count)}"
        )


if __name__ == "__main__":
    main()