import logging
import argparse
import os

import optuna
import torch

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.classifier_model.train import train_classifier, create_cos_lr_sched

LOGGER = logging.getLogger(__name__)

def create_objective(data_path, epochs, batch_size, num_workers):
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        p_shift = trial.suggest_float("p_shift", 0.1, 0.5)
        base_channels = trial.suggest_categorical("base_channels", [8, 16, 32])
        pos_weight = trial.suggest_float("pos_weight", 1.0, 20.0)
        fn_penalty_weight = trial.suggest_float("fn_penalty_weight", 1.0, 10.0)
        pred_threshold = trial.suggest_float("pred_threshold", 0.1, 0.5)


        train_ds = H5PatchDataset(data_path, split="train", return_metadata=True, 
                                  return_masks=False, augment=True, 
                                  p_flip=0.5, p_rot=0.75, p_shift=p_shift)
        val_ds = H5PatchDataset(data_path, split="val", return_metadata=True, 
                                return_masks=False)

        model = TrailClassifier(base_channels=base_channels, dropout=dropout_rate)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = create_cos_lr_sched(optimizer, epochs, warmup_epochs=5)


        _, _, best_val_recall, total_epochs = train_classifier(model=model,
                                                               train_ds=train_ds,
                                                               val_ds=val_ds,
                                                               optimizer=optimizer,
                                                               scheduler=scheduler,
                                                               epochs=epochs,
                                                               batch_size=batch_size,
                                                               pos_weight=pos_weight,
                                                               fn_penalty_weight=fn_penalty_weight,
                                                               pred_threshold=pred_threshold,
                                                               num_workers=num_workers,
                                                               save_path=None,
                                                               trial=trial)
        
        score = best_val_recall

        LOGGER.warning(
            f'Trial {trial.number} complete after {total_epochs} epochs. \n'
            f'Params: lr={learning_rate:.2e} | channels={base_channels} | '
            f'pos_w={pos_weight:.2f} | fn_penalty={fn_penalty_weight:.2f}\n'
            f'Best Validation Recall: {score:.4f}'
        )
        return score
    return objective

def parse_args():
    parser = argparse.ArgumentParser(description="Tune TrailClassifier Hyperparameters")
    
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, 
                        format="%(asctime)s %(name)s: %(message)s")
    
    db_file_dir = "results/optuna"
    os.makedirs(db_file_dir, exist_ok=True)
    
    # We use "maximize" because our objective is best_val_recall
    study = optuna.create_study(
        study_name="classifier_tuning", 
        storage=f"sqlite:///{db_file_dir}/classifier_optuna_study.db", 
        load_if_exists=True, 
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    objective = create_objective(args.data_path, args.epochs, args.batch_size, args.num_workers)
    study.optimize(objective, n_trials=args.trials)

    print("\n" + "="*30)
    print("CLASSIFIER TUNING COMPLETE")
    print(f"Best Recall: {study.best_value:.6f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*30)