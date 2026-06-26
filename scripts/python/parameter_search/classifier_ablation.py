import argparse
import csv
import gc
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.classifier_model.classifier_train_function import train_classifier
from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.data.sampler import FixedStepWeightedTrailSampler
from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint
from satellite_trail_segmentation.ml_utils.lr_scheduler import create_cos_lr_sched
from satellite_trail_segmentation.ml_utils.metrics import best_threshold_by_penalized_specificity, conf_counts_from_logits, init_conf_counts, metrics_from_conf_counts, specificity_with_recall_penalty, update_conf_counts_batch
from satellite_trail_segmentation.ml_utils.seed import set_seed


LOGGER = logging.getLogger(__name__)
NORMALIZATION = "source_zscore"
SHIFT_VALUES = (0.0, 0.25)
MIN_SHIFT = 15
MAX_SHIFT = 100
GRAD_CLIP_MAX_NORM = 1.0
EARLY_STOPPING_MIN_DELTA = 0.0
BASE_CHANNELS = 16
MIN_RECALL = 0.99
RECALL_PENALTY = 3.0
THRESHOLDS = np.linspace(0.05, 0.95, 37)
REQUIRED_COLUMNS = ("trial_number", "score", "learning_rate", "weight_decay", "fn_penalty_weight", "sampler_pos_fraction", "pos_weight", "dropout_rate")
SUMMARY_FIELDS = ("source_trial_number", "variant", "ranking_score", "optuna_score", "p_shift", "train_final_epoch",
                  "train_best_penalized_specificity", "val_best_threshold", "val_penalized_specificity", "val_iou",
                  "val_precision", "val_recall", "val_specificity", "val_f1", "val_fpr", "val_fnr",
                  "learning_rate", "weight_decay", "fn_penalty_weight", "sampler_pos_fraction", "pos_weight", "dropout_rate")


def default_trial_csv(db_dir, study_name):
    parent_dir = os.path.dirname(os.path.normpath(db_dir)) or "."
    return os.path.join(parent_dir, "summaries", f"{study_name}_trials.csv")


def load_top_trials(trial_csv, top_n):
    if top_n <= 0:
        raise ValueError(f"--top-n must be positive, got {top_n}")
    if not os.path.exists(trial_csv):
        raise FileNotFoundError(f"Trial CSV not found: {trial_csv}")

    with open(trial_csv, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Trial CSV is empty: {trial_csv}")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Trial CSV {trial_csv} is missing required columns: {missing}")

        trials = []
        for row_number, row in enumerate(reader, start=2):
            try:
                parsed = {column: float(row[column]) for column in REQUIRED_COLUMNS if column != "trial_number"}
                parsed["trial_number"] = int(row["trial_number"])
            except ValueError as exc:
                raise ValueError(f"Could not parse numeric values in {trial_csv} row {row_number}: {row}") from exc
            trials.append(parsed)

    if not trials:
        raise ValueError(f"Trial CSV has no data rows: {trial_csv}")

    return sorted(trials, key=lambda row: row["score"], reverse=True)[:top_n]


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.steps_per_epoch <= 0:
        raise ValueError(f"--steps-per-epoch must be positive, got {args.steps_per_epoch}")


def make_eval_loader(dataset, batch_size, num_workers):
    loader_kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": num_workers, "pin_memory": True}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def evaluate_best_checkpoint(model, data_path, batch_size, num_workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataset = H5PatchDataset(data_path, split="val", return_metadata=True, return_masks=False, normalization=NORMALIZATION)
    loader = make_eval_loader(dataset, batch_size, num_workers)
    threshold_counts = {threshold: init_conf_counts() for threshold in THRESHOLDS}

    with torch.no_grad():
        for images, metadata in loader:
            images = images.to(device)
            targets = metadata["patch_has_trail"].to(device=device, dtype=torch.float32).view(-1, 1)
            logits = model(images)

            for threshold in THRESHOLDS:
                batch_counts = conf_counts_from_logits(logits, targets, threshold)
                update_conf_counts_batch(threshold_counts[threshold], batch_counts)

    metrics_by_threshold = {threshold: metrics_from_conf_counts(counts) for threshold, counts in threshold_counts.items()}
    best_threshold, best_metrics = best_threshold_by_penalized_specificity(metrics_by_threshold, MIN_RECALL, RECALL_PENALTY)
    ranking_score = specificity_with_recall_penalty(best_metrics, MIN_RECALL, RECALL_PENALTY)
    return float(best_threshold), float(ranking_score), best_metrics


def run_variant(trial_row, p_shift, args, output_dir):
    trial_number = int(trial_row["trial_number"])
    variant = f"trial_{trial_number}_shift_{str(p_shift).replace('.', 'p')}"
    temp_path = output_dir / "tmp" / f"{variant}.pt"
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    trial_seed = args.seed + trial_number
    set_seed(trial_seed)

    train_ds = H5PatchDataset(args.data_path, split="train", return_metadata=True, return_masks=False, augment=True,
                              p_shift=p_shift, min_shift=MIN_SHIFT, max_shift=MAX_SHIFT, normalization=NORMALIZATION)
    val_ds = H5PatchDataset(args.data_path, split="val", return_metadata=True, return_masks=False, normalization=NORMALIZATION)
    sampler = FixedStepWeightedTrailSampler(train_ds.pos_indices, train_ds.neg_indices, pos_fraction=trial_row["sampler_pos_fraction"], num_samples=args.steps_per_epoch * args.batch_size)
    model = TrailClassifier(base_channels=BASE_CHANNELS, dropout=trial_row["dropout_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial_row["learning_rate"], weight_decay=trial_row["weight_decay"])
    scheduler = create_cos_lr_sched(optimizer, args.epochs, warmup_epochs=args.warmup_epochs, eta_min=trial_row["learning_rate"] * 0.01)

    try:
        LOGGER.info(f"Running {variant} | optuna_score={trial_row['score']:.6f}")
        train_metrics = train_classifier(model, train_ds, val_ds, optimizer, scheduler, args.epochs, args.batch_size,
                                         pos_weight=trial_row["pos_weight"],
                                         fn_penalty_weight=trial_row["fn_penalty_weight"],
                                         pred_thresholds=THRESHOLDS, min_recall=MIN_RECALL,
                                         recall_penalty=RECALL_PENALTY, sampler=sampler,
                                         num_workers=args.num_workers, full_save_path=str(temp_path),
                                         trial=None, seed=trial_seed, grad_clip_max_norm=GRAD_CLIP_MAX_NORM,
                                         early_stopping_patience=args.early_stopping_patience,
                                         early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA)

        if not temp_path.exists():
            raise RuntimeError(f"Training did not create a temporary checkpoint for {variant}: {temp_path}")

        eval_model = TrailClassifier(base_channels=BASE_CHANNELS, dropout=trial_row["dropout_rate"])
        load_checkpoint(str(temp_path), eval_model)
        val_threshold, ranking_score, val_metrics = evaluate_best_checkpoint(eval_model, args.data_path, args.batch_size, args.num_workers)

        return {"source_trial_number": trial_number, "variant": variant, "ranking_score": ranking_score,
                "optuna_score": trial_row["score"], "p_shift": p_shift,
                "train_final_epoch": train_metrics["final_epoch"],
                "train_best_penalized_specificity": train_metrics["best_penalized_specificity"],
                "val_best_threshold": val_threshold, "val_penalized_specificity": ranking_score,
                "val_iou": val_metrics["iou"], "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"], "val_specificity": val_metrics["specificity"],
                "val_f1": val_metrics["dice"], "val_fpr": val_metrics["fpr"],
                "val_fnr": val_metrics["fnr"], "learning_rate": trial_row["learning_rate"],
                "weight_decay": trial_row["weight_decay"],
                "fn_penalty_weight": trial_row["fn_penalty_weight"],
                "sampler_pos_fraction": trial_row["sampler_pos_fraction"],
                "pos_weight": trial_row["pos_weight"], "dropout_rate": trial_row["dropout_rate"]}
    finally:
        if temp_path.exists():
            temp_path.unlink()
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()


def write_summary(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run classifier ablations from top Optuna trial CSV rows")

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--trial-csv", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="results/hyperparameter_tuning/ablations/classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--study-name", type=str, default="classifier_tuning")
    parser.add_argument("--db-dir", type=str, default="results/hyperparameter_tuning/dbs")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    validate_args(args)
    trial_csv = args.trial_csv if args.trial_csv is not None else default_trial_csv(args.db_dir, args.study_name)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "classifier_ablation_summary.csv"
    top_trials = load_top_trials(trial_csv, args.top_n)

    LOGGER.info(f"Loaded {len(top_trials)} top trials from {trial_csv}")
    LOGGER.info(f"Summary CSV: {summary_path}")

    rows = []
    for trial_row in top_trials:
        for p_shift in SHIFT_VALUES:
            rows.append(run_variant(trial_row, p_shift, args, output_dir))
            
    rows = sorted(rows, key=lambda row: row["ranking_score"], reverse=True)
    write_summary(rows, summary_path)

    print("\n" + "=" * 30)
    print("CLASSIFIER ABLATION COMPLETE")
    print(f"Summary CSV: {summary_path}")
    print(f"Best Variant: {rows[0]['variant']}")
    print(f"Best Penalized Specificity: {rows[0]['ranking_score']:.6f}")
    print("=" * 30)
