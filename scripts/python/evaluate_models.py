import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.evaluation.classifier_evaluate import evaluate_dataset_classifier
from satellite_trail_segmentation.evaluation.unet_evaluate import evaluate_dataset_unet
from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint
from satellite_trail_segmentation.ml_utils.metrics import best_threshold_by_metric, best_threshold_by_penalized_specificity, specificity_with_recall_penalty
from satellite_trail_segmentation.unet_model.attention_unet import AttentionUNet
from satellite_trail_segmentation.unet_model.unet import UNet
from satellite_trail_segmentation.utils.visualizations import plot_roc_curve, plot_threshold_metrics


SEGMENTATION_MODELS = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
}


def checkpoint_model_config(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    return checkpoint.get("model_config", {})


def build_model(model_type, model_path):
    model_config = checkpoint_model_config(model_path)

    if model_type in SEGMENTATION_MODELS:
        allowed_keys = ("in_channels", "out_channels", "kernel_size", "base_channels", "dropout", "use_batchnorm")
        model_kwargs = {key: model_config[key] for key in allowed_keys if key in model_config}
        model = SEGMENTATION_MODELS[model_type](**model_kwargs)
    else:
        allowed_keys = ("in_channels", "kernel_size", "base_channels", "dropout")
        model_kwargs = {key: model_config[key] for key in allowed_keys if key in model_config}
        model = TrailClassifier(**model_kwargs)

    load_checkpoint(model_path, model)
    model.eval()
    return model, model_config


def write_summary(summary_csv_path, row):
    if summary_csv_path is None:
        return

    summary_csv_path = Path(summary_csv_path)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate satellite trail models.")

    parser.add_argument("--model-type", type=str, required=True, choices=["unet", "attention_unet", "classifier"])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--h5-path", type=str, default="/home/anp50/rds/hpc-work/satellite_trails/data/h5s/dataset.h5")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--normalization", type=str, default="source_zscore", choices=["source_zscore", "patch_zscore", "uint8"])
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-count", type=int, default=37)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=0.99)
    parser.add_argument("--recall-penalty", type=float, default=3.0)
    parser.add_argument("--threshold-metrics-save-path", type=str, default=None)
    parser.add_argument("--roc-save-path", type=str, default=None)
    parser.add_argument("--summary-csv-path", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.fixed_threshold is None:
        thresholds = list(np.linspace(args.threshold_min, args.threshold_max, args.threshold_count))
        threshold_mode = "sweep"
    else:
        thresholds = [float(args.fixed_threshold)]
        threshold_mode = "fixed"

    model, model_config = build_model(args.model_type, args.model_path)

    print(f"Loaded {args.model_type} model from {args.model_path}")
    print(f"Model config: {model_config}")
    print(f"Evaluating split={args.split} normalization={args.normalization} threshold_mode={threshold_mode} thresholds={thresholds[0]:.2f}..{thresholds[-1]:.2f} n={len(thresholds)}")

    if args.model_type in SEGMENTATION_MODELS:
        batch_size = args.batch_size or 64
        metrics_by_threshold, fpr, tpr, roc_thresholds, optimal_threshold, roc_auc = evaluate_dataset_unet(
            model, args.h5_path, args.split, thresholds, batch_size, normalization=args.normalization
        )
        best_threshold, best_metrics = best_threshold_by_metric(metrics_by_threshold, "iou")
        ranking_score = best_metrics["iou"]

        print("UNet threshold metrics:")
        print(metrics_by_threshold)
        print(f"Best IOU: {ranking_score:.6f} at threshold {best_threshold:.3f}")
        print(f"ROC AUC: {roc_auc:.6f} | ROC optimal threshold: {optimal_threshold:.6f}")

        if args.roc_save_path:
            plot_roc_curve(fpr, tpr, roc_thresholds, roc_auc, optimal_threshold, save_path=args.roc_save_path)
        if args.threshold_metrics_save_path:
            plot_threshold_metrics(metrics_by_threshold, save_path=args.threshold_metrics_save_path)

        summary = {"model_type": args.model_type, "model_path": args.model_path, "h5_path": args.h5_path,
                   "split": args.split, "normalization": args.normalization, "batch_size": batch_size,
                   "threshold_mode": threshold_mode,
                   "best_threshold": best_threshold, "ranking_metric": "iou", "ranking_score": ranking_score,
                   "roc_auc": roc_auc, "roc_optimal_threshold": optimal_threshold, **best_metrics}
    else:
        batch_size = args.batch_size or 128
        metrics_by_threshold, image_wise_counts = evaluate_dataset_classifier(
            model, args.h5_path, args.split, thresholds, batch_size, normalization=args.normalization
        )
        best_threshold, best_metrics = best_threshold_by_penalized_specificity(metrics_by_threshold, args.min_recall, args.recall_penalty)
        ranking_score = specificity_with_recall_penalty(best_metrics, args.min_recall, args.recall_penalty)

        print("Classifier threshold metrics:")
        print(metrics_by_threshold)
        print()
        print("Classifier image-wise counts:")
        print(image_wise_counts)
        print(f"Best penalized specificity: {ranking_score:.6f} at threshold {best_threshold:.3f}")

        if args.threshold_metrics_save_path:
            plot_threshold_metrics(metrics_by_threshold, save_path=args.threshold_metrics_save_path)

        summary = {"model_type": args.model_type, "model_path": args.model_path, "h5_path": args.h5_path,
                   "split": args.split, "normalization": args.normalization, "batch_size": batch_size,
                   "threshold_mode": threshold_mode,
                   "best_threshold": best_threshold, "ranking_metric": "penalized_specificity",
                   "ranking_score": ranking_score, "min_recall": args.min_recall,
                   "recall_penalty": args.recall_penalty, **best_metrics}

    write_summary(args.summary_csv_path, summary)
