import logging
import argparse
import os

import optuna
import torch

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.data.sampler import BalancedTrailSampler
from satellite_trail_segmentation.unet_model.unet import UNet
from satellite_trail_segmentation.unet_model.train import train_unet, create_cos_lr_sched

LOGGER = logging.getLogger(__name__)


def create_objective(data_path, epochs):
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        p_aug = trial.suggest_float("p_aug", 0.1, 0.5)
        sampler_fraction = trial.suggest_float("sampler_fraction", 0.1, 0.5)

        train_ds = H5PatchDataset(data_path, split="train", augment=True, p_flip=p_aug, p_rot=p_aug, p_shift=p_aug)
        val_ds = H5PatchDataset(data_path, split="val")
        sampler = BalancedTrailSampler(train_ds.pos_indices, train_ds.neg_indices, pos_fraction=sampler_fraction)

        model = UNet(dropout=dropout_rate)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = create_cos_lr_sched(optimizer, epochs, warmup_epochs=5)

        train_loss, val_loss, best_loss, total_epochs = train_unet(model, train_ds, val_ds, optimizer, scheduler, epochs, batch_size=64, num_workers=4, sampler=sampler, trial=trial)
        
        score = best_loss

        LOGGER.warning(f'Trial {trial.number} complete after {total_epochs} epochs. \n Parameters: lr={learning_rate:.2e} | dropout={dropout_rate} | p_augmentation={p_aug}\nBest validation loss: {score}.')
        
        
        return score
    return objective



def parse_args():
    parser = argparse.ArgumentParser(description="Train satellite trail segmentation model")
    
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--verbose", action="store_true", default=True)
    
    return parser.parse_args()


if __name__ == "__main__":
    # Configuration
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, 
                        format="%(asctime)s %(name)s: %(message)s")
    
    data_path = args.data_path
    n_trials = args.trials 
    epochs_per_trial = args.epochs

    objective = create_objective(data_path, epochs_per_trial)

    db_file_dir = "results/optuna"
    os.makedirs(db_file_dir, exist_ok=True)
    study = optuna.create_study(study_name="unet_tuning", storage=f"sqlite:///{db_file_dir}/unet_optuna_study.db", 
                                load_if_exists=True, direction="minimize", 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))

    study.optimize(objective, n_trials=n_trials)


    print("\n" + "="*30)
    print("OPTUNA STUDY COMPLETE")
    print(f"Best Value: {study.best_value:.6f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*30)