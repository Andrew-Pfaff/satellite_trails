import argparse
import csv
import gc
import logging
import os

import numpy as np
import optuna
import torch

from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.classifier_model.classifier_train_function import train_classifier
from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.data.sampler import FixedStepWeightedTrailSampler
from satellite_trail_segmentation.ml_utils.lr_scheduler import create_cos_lr_sched
from satellite_trail_segmentation.ml_utils.seed import set_seed


LOGGER = logging.getLogger(__name__)
NORMALIZATION = "source_zscore"
P_SHIFT = 0.0
MIN_SHIFT = 4
MAX_SHIFT = 20
GRAD_CLIP_MAX_NORM = 1.0
EARLY_STOPPING_MIN_DELTA = 0.0
BASE_CHANNELS = 16
MIN_RECALL = 0.99
RECALL_PENALTY = 3.0


def create_objective(data_path, epochs, warmup_epochs, batch_size, num_workers, seed, steps_per_epoch, early_stopping_patience, trial_results_path):
    train_ds = H5PatchDataset(data_path, split="train", return_metadata=True, return_masks=False, augment=True,
                              p_shift=P_SHIFT, min_shift=MIN_SHIFT, max_shift=MAX_SHIFT, normalization=NORMALIZATION)
    val_ds = H5PatchDataset(data_path, split="val", return_metadata=True, return_masks=False, normalization=NORMALIZATION)

    def objective(trial):
        trial_seed = seed + trial.number
        set_seed(trial_seed)

        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
        fn_penalty_weight = trial.suggest_float("fn_penalty_weight", 0.0, 6.0)
        sampler_pos_fraction = trial.suggest_float("sampler_pos_fraction", 0.065, 0.3)
        pos_weight = trial.suggest_float("pos_weight", 1.0, 15.0, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)

        eta_min = learning_rate * 0.01
        pred_thresholds = np.linspace(0.05, 0.95, 37)

        LOGGER.info(f"Trial {trial.number} start | lr={learning_rate:.2e} | weight_decay={weight_decay:.2e} | fn_penalty_weight={fn_penalty_weight:.2f} | sampler_pos_fraction={sampler_pos_fraction:.3f} | pos_weight={pos_weight:.2f} | dropout={dropout_rate:.3f}")

        sampler = FixedStepWeightedTrailSampler(train_ds.pos_indices, train_ds.neg_indices, pos_fraction=sampler_pos_fraction, num_samples=steps_per_epoch * batch_size)
        model = TrailClassifier(base_channels=BASE_CHANNELS, dropout=dropout_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = create_cos_lr_sched(optimizer, epochs, warmup_epochs=warmup_epochs, eta_min=eta_min)

        train_metrics = train_classifier(model, train_ds, val_ds, optimizer, scheduler, epochs, batch_size,
                                         pos_weight=pos_weight, fn_penalty_weight=fn_penalty_weight,
                                         pred_thresholds=pred_thresholds, min_recall=MIN_RECALL,
                                         recall_penalty=RECALL_PENALTY, sampler=sampler,
                                         num_workers=num_workers, trial=trial, seed=trial_seed,
                                         grad_clip_max_norm=GRAD_CLIP_MAX_NORM,
                                         early_stopping_patience=early_stopping_patience,
                                         early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA)

        score = train_metrics["best_penalized_specificity"]
        LOGGER.info(f"Trial {trial.number} complete | score={score:.6f} | final_epoch={train_metrics['final_epoch']}")
        row = {"trial_number": trial.number, "score": score, "final_epoch": train_metrics["final_epoch"], "learning_rate": learning_rate, "weight_decay": weight_decay, "fn_penalty_weight": fn_penalty_weight, "sampler_pos_fraction": sampler_pos_fraction, "pos_weight": pos_weight, "dropout_rate": dropout_rate}
        with open(trial_results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

        return score

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Tune TrailClassifier hyperparameters")

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--study-name", type=str, default="classifier_tuning")
    parser.add_argument("--db-dir", type=str, default="results/hyperparameter_tuning/dbs")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    set_seed(args.seed)
    os.makedirs(args.db_dir, exist_ok=True)

    study = optuna.create_study(study_name=args.study_name, storage=f"sqlite:///{args.db_dir}/{args.study_name}.db",
                                load_if_exists=True, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=10, interval_steps=2))

    LOGGER.info(f"Using fixed steps per epoch: {args.steps_per_epoch}")
    LOGGER.info(f"Study storage: {args.db_dir}/{args.study_name}.db")
    trial_results_path = os.path.join(os.path.dirname(os.path.normpath(args.db_dir)) or ".", "summaries", f"{args.study_name}_trials.csv")
    os.makedirs(os.path.dirname(trial_results_path), exist_ok=True)
    LOGGER.info(f"Trial results CSV: {trial_results_path}")

    LOGGER.info(f"Fixed settings: normalization={NORMALIZATION} | p_shift={P_SHIFT:.3f} | min_shift={MIN_SHIFT} | max_shift={MAX_SHIFT} | grad_clip={GRAD_CLIP_MAX_NORM:.2f} | min_delta={EARLY_STOPPING_MIN_DELTA:.3f} | base_channels={BASE_CHANNELS} | min_recall={MIN_RECALL:.2f} | recall_penalty={RECALL_PENALTY:.2f}")

    objective = create_objective(args.data_path, args.epochs, args.warmup_epochs, args.batch_size, args.num_workers, args.seed,
                                 args.steps_per_epoch, args.early_stopping_patience, trial_results_path)

    study.optimize(objective, n_trials=args.trials)

    print("\n" + "=" * 30)
    print("CLASSIFIER STUDY COMPLETE")
    print(f"Best Penalized Specificity: {study.best_value:.6f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 30)
