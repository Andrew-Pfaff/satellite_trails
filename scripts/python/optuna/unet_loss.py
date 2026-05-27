import logging
import argparse
import os

import numpy as np
import optuna
import torch

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.data.sampler import BalancedTrailSampler
from satellite_trail_segmentation.unet_model.unet import UNet
from satellite_trail_segmentation.unet_model.unet_train_function import train_unet
from satellite_trail_segmentation.ml_utils.lr_scheduler import create_cos_lr_sched
from satellite_trail_segmentation.ml_utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def create_objective(data_path, epochs, batch_size, num_workers, seed,
                     learning_rate, dropout_rate, sampler_fraction):
    train_ds = H5PatchDataset(data_path, split="train", augment=True, p_flip=0.5, p_rot=0.75, p_shift=0)
    val_ds = H5PatchDataset(data_path, split="val")

    def objective(trial):
        trial_seed = seed + trial.number
        set_seed(trial_seed)

        pos_weight = trial.suggest_float("pos_weight", 1.0, 20.0)
        bce_loss_factor = 1.0
        dice_loss_factor = trial.suggest_float("dice_loss_factor", 0.1, 20.0, log=True)

        sampler = BalancedTrailSampler(train_ds.pos_indices, train_ds.neg_indices, pos_fraction=sampler_fraction)

        model = UNet(dropout=dropout_rate)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = create_cos_lr_sched(optimizer, epochs)

        iou_thresholds = np.linspace(0.4, 0.6, 9)

        train_metrics = train_unet(model, train_ds, val_ds, optimizer, scheduler, 
                                   epochs, batch_size=batch_size, 
                                   pos_weight=pos_weight, bce_loss_factor=bce_loss_factor, dice_loss_factor=dice_loss_factor,
                                   iou_thresholds=iou_thresholds, sampler=sampler,
                                   num_workers=num_workers, trial=trial, seed=trial_seed)
        
        score = train_metrics["best_iou"]
        threshold = train_metrics["best_threshold"]

        LOGGER.info(f"Trial {trial.number} complete after {train_metrics['final_epoch']} epochs. \n Parameters: lr={learning_rate:.2e} | dropout={dropout_rate}\nBest validation IOU: {score} at threshold: {threshold}.")
        
        return score
    return objective



def parse_args():
    parser = argparse.ArgumentParser(description="Train satellite trail segmentation model")
    
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--dropout-rate", type=float, required=True)
    parser.add_argument("--sampler-fraction", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, 
                        format="%(asctime)s %(name)s: %(message)s")
    
    set_seed(args.seed)    

    db_file_dir = "results/optuna"
    os.makedirs(db_file_dir, exist_ok=True)

    study = optuna.create_study(study_name="unet_loss_tuning", storage=f"sqlite:///{db_file_dir}/unet_loss_tuning.db", 
                                load_if_exists=True, 
                                direction="maximize", 
                                sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))

    objective = create_objective(args.data_path, args.epochs, args.batch_size, args.num_workers, args.seed,
                                 args.learning_rate, args.dropout_rate, args.sampler_fraction)
    
    study.optimize(objective, n_trials=args.trials)


    print("\n" + "="*30)
    print("OPTUNA STUDY COMPLETE")
    print(f"Best Value: {study.best_value:.6f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*30)
