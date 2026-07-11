"""Evaluate a small ASTA postprocessing grid on saved validation masks."""

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from satellite_trail_segmentation.ml_utils.metrics import (
    conf_counts_from_arrays,
    init_conf_counts,
    metrics_from_conf_counts,
    update_conf_counts_batch,
)
from satellite_trail_segmentation.postprocess.pipeline import postprocess_segmentation


Image.MAX_IMAGE_PIXELS = 150000000

HOUGH_THRESHOLDS = (25, 50, 75)
MIN_LINE_LENGTHS = (50, 100)
MAX_LINE_GAPS = (125, 250, 375)
CONTOUR_AREA_THRESHOLDS = (1500, 3000)

FIXED_PARAMS = {
    "morph_kernel_size": 3,
    "min_component_size": 500,
    "contour_filter": True,
}


def parameter_grid():
    """Returns all 36 postprocessing parameter combinations."""

    return [
        {
            "hough_threshold": hough_threshold,
            "min_line_length": min_line_length,
            "max_line_gap": max_line_gap,
            "contour_area_threshold": contour_area_threshold,
            **FIXED_PARAMS,
        }
        for hough_threshold, min_line_length, max_line_gap, contour_area_threshold in product(
            HOUGH_THRESHOLDS,
            MIN_LINE_LENGTHS,
            MAX_LINE_GAPS,
            CONTOUR_AREA_THRESHOLDS,
        )
    ]


def find_pairs(input_dir):
    """Finds ``*_prediction.png`` and matching ``*_mask.png`` files."""

    input_dir = Path(input_dir)
    pairs = []
    for prediction_path in sorted(input_dir.glob("*_prediction.png")):
        prefix = prediction_path.name.removesuffix("_prediction.png")
        mask_path = input_dir / f"{prefix}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {prediction_path.name}: {mask_path.name}")
        pairs.append((prediction_path, mask_path))

    if not pairs:
        raise ValueError(f"No *_prediction.png files found in {input_dir}")
    return pairs


def load_binary_mask(path):
    """Loads a PNG as a boolean mask."""

    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def initial_totals(configurations):
    """Creates running confusion counts for raw and postprocessed methods."""

    return [
        {"method": "none", "params": None, "counts": init_conf_counts()},
        *[
            {"method": "postprocess", "params": params, "counts": init_conf_counts()}
            for params in configurations
        ],
    ]


def result_rows(totals, image_count):
    """Converts running totals into CSV rows with aggregate metrics."""

    rows = []
    for index, result in enumerate(totals):
        params = result["params"] or {}
        metrics = metrics_from_conf_counts(result["counts"])
        rows.append(
            {
                "method_id": "none" if result["method"] == "none" else f"postprocess_{index:02d}",
                "method": result["method"],
                **{name: params.get(name, "") for name in (
                    "hough_threshold",
                    "min_line_length",
                    "max_line_gap",
                    "morph_kernel_size",
                    "min_component_size",
                    "contour_filter",
                    "contour_area_threshold",
                )},
                "image_count": image_count,
                **metrics,
            }
        )
    return rows


def write_results(output_csv, totals, image_count):
    """Writes the current aggregate counts and metrics."""

    rows = result_rows(totals, image_count)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_csv)


def run_sweep(input_dir, output_csv):
    """Runs every parameter combination on each validation pair."""

    pairs = find_pairs(input_dir)
    configurations = parameter_grid()
    totals = initial_totals(configurations)

    for image_count, (prediction_path, mask_path) in enumerate(pairs, start=1):
        print(f"{image_count}/{len(pairs)}: {prediction_path.name}", flush=True)
        prediction = load_binary_mask(prediction_path)
        target = load_binary_mask(mask_path)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {prediction_path.name}: {prediction.shape} != {target.shape}"
            )

        update_conf_counts_batch(totals[0]["counts"], conf_counts_from_arrays(prediction, target))
        for result in totals[1:]:
            processed = postprocess_segmentation(prediction, **result["params"])
            counts = conf_counts_from_arrays(processed, target)
            update_conf_counts_batch(result["counts"], counts)

        write_results(output_csv, totals, image_count)

    return result_rows(totals, len(pairs))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing validation predictions and masks.")
    parser.add_argument("--output-csv", required=True, help="CSV receiving aggregate confusion counts and metrics.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args.input_dir, args.output_csv)
