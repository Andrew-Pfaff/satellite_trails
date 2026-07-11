"""Evaluate final full-field PNG segmentation pipelines and write CSV metrics."""

import argparse
import csv
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from segmentation import SatelliteTrailPipeline
from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint
from satellite_trail_segmentation.ml_utils.metrics import conf_counts_from_arrays, metrics_from_conf_counts
from satellite_trail_segmentation.postprocess.pipeline import postprocess_segmentation
from satellite_trail_segmentation.unet_model.unet import UNet


def build_model(model_class, checkpoint_path, allowed_keys):
    """
    Builds a model from checkpoint configuration and loads its weights.

    Args:
        model_class (type): Model class to instantiate.
        checkpoint_path (str or Path): Path to a checkpoint or weights file.
        allowed_keys (tuple): Model configuration keys accepted by the class.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", {}) or {}
    model_kwargs = {key: model_config[key] for key in allowed_keys if key in model_config}
    model = model_class(**model_kwargs)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model


def test_rows(master_split_csv, png_dir, split_id="2"):
    """
    Selects full-field PNG/mask rows for one split from the master split CSV.

    Args:
        master_split_csv (str or Path): CSV containing image name, mask name, and split id.
        png_dir (str or Path): Directory containing PNG images and masks.
        split_id (str): Split id to select. Defaults to "2".

    Returns:
        list[dict]: Rows with image and mask names plus resolved paths.
    """

    rows = []
    with open(master_split_csv, newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 3 or row[2] != str(split_id):
                continue
            image_name, mask_name = row[:2]
            rows.append(
                {
                    "image_name": image_name,
                    "mask_name": mask_name,
                    "image_path": Path(png_dir) / image_name,
                    "mask_path": Path(png_dir) / mask_name,
                }
            )
    return rows


def postprocess_methods(args=None):
    """
    Defines the final full-field postprocessing variant to evaluate.

    Returns:
        dict: Mapping of output method suffixes to postprocessing keyword arguments.
    """

    selected = {
        "hough_threshold": getattr(args, "hough_threshold", 50),
        "min_line_length": getattr(args, "min_line_length", 100),
        "max_line_gap": getattr(args, "max_line_gap", 125),
        "morph_kernel_size": getattr(args, "morph_kernel_size", 3),
        "min_component_size": getattr(args, "min_component_size", 500),
        "contour_filter": True,
        "contour_area_threshold": getattr(args, "contour_area_threshold", 1500),
    }
    return {
        "postprocess_asta": {
            "hough_threshold": 50,
            "min_line_length": 100,
            "max_line_gap": 250,
            "morph_kernel_size": 3,
            "min_component_size": 500,
            "contour_filter": True,
            "contour_area_threshold": 3000,
        },
        "postprocess_selected": selected,
    }


def metric_row(method, image_name, mask_name, prediction, target):
    """
    Builds one per-image metric row for a prediction and target mask.

    Args:
        method (str): Evaluation method name.
        image_name (str): Source image filename.
        mask_name (str): Mask filename.
        prediction (np.ndarray): Binary prediction mask.
        target (np.ndarray): Ground-truth mask.

    Returns:
        dict: Per-image metric row.
    """

    metrics = metrics_from_conf_counts(conf_counts_from_arrays(prediction, target))
    return {
        "method": method,
        "image_name": image_name,
        "mask_name": mask_name,
        **metrics,
    }


def aggregate_rows(rows):
    """
    Aggregates per-image rows into one metric row per method.

    Args:
        rows (list[dict]): Per-image metric rows.

    Returns:
        list[dict]: Aggregate rows grouped by method.
    """

    grouped = {}
    for row in rows:
        method = row["method"]
        grouped.setdefault(method, {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "image_count": 0})
        grouped[method]["image_count"] += 1
        for key in ("tp", "fp", "fn", "tn"):
            grouped[method][key] += float(row[key])

    aggregate = []
    for method, counts in grouped.items():
        metric_counts = {key: counts[key] for key in ("tp", "fp", "fn", "tn")}
        aggregate.append(
            {
                "method": method,
                "image_count": counts["image_count"],
                **metrics_from_conf_counts(metric_counts),
            }
        )
    return aggregate


def write_rows(csv_path, rows):
    """
    Writes metric rows to a CSV file.

    Args:
        csv_path (str or Path): Output path.
        rows (list[dict]): Rows to write. Empty lists create no file.
    """

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_full_fields(args):
    """
    Runs raw and postprocessed full-field evaluations for selected PNG rows.

    Args:
        args (argparse.Namespace): Parsed command-line options.

    Returns:
        dict: Per-method per-image metric rows.
    """

    device = torch.device(args.device) if args.device is not None else None
    unet_model = build_model(
        UNet,
        args.unet_checkpoint,
        ("in_channels", "out_channels", "kernel_size", "base_channels", "dropout", "use_batchnorm"),
    )
    classifier_model = build_model(
        TrailClassifier,
        args.classifier_checkpoint,
        ("in_channels", "kernel_size", "base_channels", "dropout"),
    )
    pipeline = SatelliteTrailPipeline(
        unet_model,
        classifier_model=classifier_model,
        patch_dim=args.patch_dim,
        device=device,
        unet_batch_size=args.unet_batch_size,
        classifier_batch_size=args.classifier_batch_size,
        num_workers=args.num_workers,
    )

    rows_by_method = {
        "unet": [],
        "classifier_unet": [],
    }
    configs = postprocess_methods(args)
    for prefix in ("unet", "classifier_unet"):
        for suffix in configs:
            rows_by_method[f"{prefix}_{suffix}"] = []

    selected_rows = test_rows(args.master_split_csv, args.png_dir, split_id=args.split_id)
    if not selected_rows:
        raise ValueError(f"No rows found for split_id={args.split_id!r}")

    for index, row in enumerate(selected_rows, start=1):
        print(f"Evaluating {index}/{len(selected_rows)}: {row['image_name']}")
        preprocessing = pipeline.preprocessing(row["image_path"], row["mask_path"], normalization=args.normalization)
        target = preprocessing["mask"]

        unet_data, _ = pipeline.segmentation(
            preprocessing["patch_data"],
            use_classifier=False,
            unet_threshold=args.unet_threshold,
            classifier_threshold=args.classifier_threshold,
        )
        classifier_data, _ = pipeline.segmentation(
            preprocessing["patch_data"],
            use_classifier=True,
            unet_threshold=args.unet_threshold,
            classifier_threshold=args.classifier_threshold,
        )
        predictions = {
            "unet": unet_data["segmented_result"],
            "classifier_unet": classifier_data["segmented_result"],
        }

        for method, prediction in predictions.items():
            rows_by_method[method].append(metric_row(method, row["image_name"], row["mask_name"], prediction, target))
            for suffix, config in configs.items():
                postprocess_method = f"{method}_{suffix}"
                postprocessed = postprocess_segmentation(prediction, **config)
                if isinstance(postprocessed, tuple):
                    postprocessed = postprocessed[0]
                rows_by_method[postprocess_method].append(
                    metric_row(postprocess_method, row["image_name"], row["mask_name"], postprocessed, target)
                )

    output_dir = Path(args.output_dir)
    all_rows = []
    for method, rows in rows_by_method.items():
        write_rows(output_dir / f"{method}_per_image_metrics.csv", rows)
        all_rows.extend(rows)
    write_rows(output_dir / "aggregate_metrics.csv", aggregate_rows(all_rows))
    return rows_by_method


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate final full-field PNG segmentation metrics.")
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--master-split-csv", required=True)
    parser.add_argument("--unet-checkpoint", required=True)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--split-id", default="2")
    parser.add_argument("--unet-threshold", type=float, default=0.6)
    parser.add_argument("--classifier-threshold", type=float, default=0.67)
    parser.add_argument("--normalization", choices=("source_zscore", "patch_zscore", "uint8"), default="source_zscore")
    parser.add_argument("--patch-dim", type=int, default=528)
    parser.add_argument("--device", default=None)
    parser.add_argument("--unet-batch-size", type=int, default=None)
    parser.add_argument("--classifier-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hough-threshold", type=int, default=50)
    parser.add_argument("--min-line-length", type=int, default=100)
    parser.add_argument("--max-line-gap", type=int, default=125)
    parser.add_argument("--morph-kernel-size", type=int, default=3)
    parser.add_argument("--min-component-size", type=int, default=500)
    parser.add_argument("--contour-area-threshold", type=float, default=1500)
    parser.add_argument("--output-dir", default="final_eval_outputs")
    return parser.parse_args()


def main():
    evaluate_full_fields(parse_args())


if __name__ == "__main__":
    main()
