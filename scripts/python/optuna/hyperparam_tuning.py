import logging
import argparse
import os

import optuna
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.model.unet import UNet
from satellite_trail_segmentation.model import train

LOGGER = logging.getLogger(__name__)


def create_objective(data_path, epochs):
    def objective(trial):
        print()
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        p_aug = trial.suggest_float("p_aug", 0.1, 0.5)
        lr_decay = trial.suggest_float("lr_decay", 10, 10000, log=True)


        train_ds = H5PatchDataset(data_path, split="train", augment=True, p_flip=p_aug, p_rot=p_aug, p_shift=p_aug)
        val_ds = H5PatchDataset(data_path, split="val")

        model = UNet(in_channels=1, out_channels=1, kernel_size=3, base_channels=8, dropout=dropout_rate)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/lr_decay)

        train_loss, val_loss, best_loss, total_epochs = train.train_unet(model, train_ds, val_ds, optimizer, scheduler, epochs, batch_size=16, num_workers=2, trial=trial)
        
        score = best_loss

        LOGGER.warning(f'Trial {trial.number} complete after {total_epochs} epochs. \n Parameters: lr={learning_rate:.2e} | dropout={dropout_rate} | p_augmentation={p_aug} | lr_decay={lr_decay:.2e} \nBest validation loss: {score}.')
        
        
        return score
    return objective



def parse_args():
    parser = argparse.ArgumentParser(description="Train satellite trail segmentation model")
    
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--trials", type=int, default=50)
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
    study = optuna.create_study(study_name="unet_tuning", storage=f"sqlite:///{db_file_dir}/optuna_study.db", load_if_exists=True, direction="minimize", pruner=optuna.pruners.MedianPruner())

    study.optimize(objective, n_trials=n_trials)


    print("\n" + "="*30)
    print("OPTUNA STUDY COMPLETE")
    print(f"Best Value: {study.best_value:.6f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*30)