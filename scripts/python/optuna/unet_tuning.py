import argparse
import gc
import logging
import os

import numpy as np
import optuna
import torch

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.data.sampler import FixedStepWeightedTrailSampler
from satellite_trail_segmentation.ml_utils.lr_scheduler import create_cos_lr_sched
from satellite_trail_segmentation.ml_utils.seed import set_seed
from satellite_trail_segmentation.unet_model.unet import UNet
from satellite_trail_segmentation.unet_model.unet_train_function import train_unet


LOGGER = logging.getLogger(__name__)
NORMALIZATION = "source_zscore"
P_SHIFT = 0.0
MIN_SHIFT = 4
MAX_SHIFT = 20
GRAD_CLIP_MAX_NORM = 1.0
EARLY_STOPPING_MIN_DELTA = 0.0
LABEL_SMOOTHING = 0.0
USE_BATCHNORM = True


def create_objective(data_path, epochs, warmup_epochs, batch_size, num_workers, seed, steps_per_epoch, early_stopping_patience):
    train_ds = H5PatchDataset(data_path, split="train", augment=True, p_shift=P_SHIFT, min_shift=MIN_SHIFT, max_shift=MAX_SHIFT, normalization=NORMALIZATION)
    val_ds = H5PatchDataset(data_path, split="val", normalization=NORMALIZATION)

    def objective(trial):
        trial_seed = seed + trial.number
        set_seed(trial_seed)

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
        bce_weight_factor = trial.suggest_float("bce_weight_factor", 0.35, 0.65)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.2)
        pos_weight = trial.suggest_categorical("pos_weight", [1.0, 2.0, 4.0])
        sampler_pos_fraction = trial.suggest_float("sampler_pos_fraction", 0.1, 0.5)

        eta_min = learning_rate * 0.01
        iou_thresholds = np.linspace(0.05, 0.95, 37)

        LOGGER.info(f"Trial {trial.number} start | lr={learning_rate:.2e} | bce_weight_factor={bce_weight_factor:.3f} | weight_decay={weight_decay:.2e} | dropout={dropout_rate:.3f} | pos_weight={pos_weight:.1f} | sampler_pos_fraction={sampler_pos_fraction:.3f}")

        sampler = FixedStepWeightedTrailSampler(train_ds.pos_indices, train_ds.neg_indices, pos_fraction=sampler_pos_fraction, num_samples=steps_per_epoch * batch_size)
        model = UNet(dropout=dropout_rate, use_batchnorm=USE_BATCHNORM)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = create_cos_lr_sched(optimizer, epochs, warmup_epochs=warmup_epochs, eta_min=eta_min)

        train_metrics = train_unet(model, train_ds, val_ds, optimizer, scheduler,
                                   epochs, batch_size=batch_size, pos_weight=pos_weight,
                                   bce_weight_factor=bce_weight_factor, label_smoothing=LABEL_SMOOTHING,
                                   iou_thresholds=iou_thresholds, sampler=sampler, num_workers=num_workers,
                                   trial=trial, seed=trial_seed, grad_clip_max_norm=GRAD_CLIP_MAX_NORM,
                                   early_stopping_patience=early_stopping_patience,
                                   early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA)

        score = train_metrics["best_iou"]
        LOGGER.info(f"Trial {trial.number} complete | score={score:.6f} | final_epoch={train_metrics['final_epoch']} | best_threshold={train_metrics['best_threshold']:.3f}")

        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

        return score

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Tune UNet hyperparameters")

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--study-name", type=str, default="unet_tuning")
    parser.add_argument("--db-dir", type=str, default="results/optuna/dbs")
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

    LOGGER.info(f"Fixed settings: normalization={NORMALIZATION} | p_shift={P_SHIFT:.3f} | min_shift={MIN_SHIFT} | max_shift={MAX_SHIFT} | grad_clip={GRAD_CLIP_MAX_NORM:.2f} | min_delta={EARLY_STOPPING_MIN_DELTA:.3f} | use_batchnorm={USE_BATCHNORM}")

    objective = create_objective(args.data_path, args.epochs, args.warmup_epochs, args.batch_size, args.num_workers, args.seed,
                                 args.steps_per_epoch, args.early_stopping_patience)
    study.optimize(objective, n_trials=args.trials)

    print("\n" + "=" * 30)
    print("UNET STUDY COMPLETE")
    print(f"Best Value: {study.best_value:.6f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 30)
