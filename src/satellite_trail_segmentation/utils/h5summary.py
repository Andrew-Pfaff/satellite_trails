import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def build_h5_summary(h5_path):
    """
    Builds a tabular summary of source images and patch counts from an HDF5 dataset.

    Reads source-level metadata from the dataset, aggregates patch counts per source image, and returns one summary row per source.

    Args:
        h5_path (str): Path to the HDF5 dataset file.

    Returns:
        list[dict]: A list of summary rows containing source file, split, and patch counts.
    """

    with h5py.File(h5_path, "r") as f:
        source_files = f["source_files"][:]
        source_split = f["source_split"][:]
        patch_source_index = f["source_index"][:]
        patch_has_trail = f["patch_has_trail"][:].astype(bool)

    rows = []
    for source_index in range(len(source_split)):
        patch_mask = patch_source_index == source_index
        num_patches = int(np.sum(patch_mask))
        num_trail_patches = int(np.sum(patch_has_trail[patch_mask]))

        source_file = source_files[source_index]
        if isinstance(source_file, bytes):
            source_file = source_file.decode("utf-8")

        split_name = SPLIT_NAMES.get(int(source_split[source_index]), str(int(source_split[source_index])))

        rows.append(
            {
                "source_index": source_index,
                "source_file": source_file,
                "split": split_name,
                "num_patches": num_patches,
                "num_trail_patches": num_trail_patches,
            }
        )

    return rows


def print_summary(rows):
    """
    Prints a formatted table of summary rows to standard output.

    Args:
        rows (list[dict]): Summary rows produced by `build_h5_summary`.
    """

    headers = ["source_index", "source_file", "split", "num_patches", "num_trail_patches"]
    table = [headers] + [[str(row[h]) for h in headers] for row in rows]
    widths = [max(len(cell) for cell in col) for col in zip(*table)]

    for i, row in enumerate(table):
        line = "  ".join(cell.ljust(width) for cell, width in zip(row, widths))
        print(line)
        if i == 0:
            print("  ".join("-" * width for width in widths))


def write_summary_csv(rows, csv_path):
    """
    Writes summary rows to a CSV file.

    Creates parent directories as needed, then writes the summary table with a fixed column order.

    Args:
        rows (list[dict]): Summary rows produced by `build_h5_summary`.
        csv_path (str): File path for the output CSV.
    """

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_index", "source_file", "split", "num_patches", "num_trail_patches"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize source images and splits in an HDF5 patch dataset")
    
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--no-print", action="store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = build_h5_summary(args.data_path)

    if not args.no_print:
        print_summary(rows)

    if args.csv_path is not None:
        write_summary_csv(rows, args.csv_path)
