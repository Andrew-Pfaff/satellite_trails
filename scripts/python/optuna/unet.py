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


def create_objective(data_path, epochs, batch_size, num_workers, seed):
    train_ds = H5PatchDataset(data_path, split="train", augment=True, p_flip=0.5, p_rot=0.75)
    val_ds = H5PatchDataset(data_path, split="val")

    def objective(trial):
        trial_seed = seed + trial.number
        set_seed(trial_seed)

        learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.01, 0.1)
        sampler_fraction = trial.suggest_float("sampler_fraction", 0.1, 0.5) 
        pos_weight = trial.suggest_float("pos_weight", 5.0, 50.0, log=True)
        bce_loss_factor = 1.0
        dice_loss_factor = trial.suggest_float("dice_loss_factor", 0.5, 3.0)

        LOGGER.info(f"Trial {trial.number} | lr={learning_rate:.2e} | wd={weight_decay:.2e} | dropout={dropout_rate:.3f} | sampler={sampler_fraction:.3f} | pos_weight={pos_weight:.2f} | bce={bce_loss_factor:.1f} | dice={dice_loss_factor:.2f}")

        sampler = BalancedTrailSampler(train_ds.pos_indices, train_ds.neg_indices, pos_fraction=sampler_fraction)

        model = UNet(dropout=dropout_rate)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = create_cos_lr_sched(optimizer, epochs)

        iou_thresholds = np.linspace(0.45, 0.65, 17)

        train_metrics = train_unet(model, train_ds, val_ds, optimizer, scheduler, 
                                   epochs, batch_size=batch_size, 
                                   pos_weight=pos_weight, bce_loss_factor=bce_loss_factor, dice_loss_factor=dice_loss_factor,
                                   iou_thresholds=iou_thresholds, sampler=sampler,
                                   num_workers=num_workers, trial=trial, seed=trial_seed)
        
        score = train_metrics["best_iou"]
        threshold = train_metrics["best_threshold"]

        LOGGER.info(f"Trial {trial.number} complete after {train_metrics['final_epoch']} epochs. \n Parameters: lr={learning_rate:.2e} | dropout={dropout_rate:.3f} | sampler_frac={sampler_fraction:.3f} | pos_weight={pos_weight:.2f} | dice_factor={dice_loss_factor:.2f}\nBest validation IOU: {score:.4f} with threshold: {threshold:.2f}.")
        
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
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, 
                        format="%(asctime)s %(name)s: %(message)s")
    
    set_seed(args.seed)    

    db_file_dir = "results/optuna"
    os.makedirs(db_file_dir, exist_ok=True)

    study = optuna.create_study(study_name="unet_full_tuning", storage=f"sqlite:///{db_file_dir}/unet_full_tuning.db", 
                                load_if_exists=True, 
                                direction="maximize", 
                                sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))

    objective = create_objective(args.data_path, args.epochs, args.batch_size, args.num_workers, args.seed)
    study.optimize(objective, n_trials=args.trials)


    print("\n" + "="*30)
    print("OPTUNA STUDY COMPLETE")
    print(f"Best Value: {study.best_value:.6f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*30)
