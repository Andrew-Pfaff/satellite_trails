import logging
import argparse
import os

import optuna
import torch

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.data.sampler import BalancedTrailSampler
from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.classifier_model.classifier_train_function import train_classifier
from satellite_trail_segmentation.ml_utils.lr_scheduler import create_cos_lr_sched
from satellite_trail_segmentation.ml_utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def create_objective(data_path, epochs, batch_size, num_workers, seed):
    train_ds = H5PatchDataset(data_path, split="train", return_metadata=True, return_masks=False, augment=True, p_flip=0.5, p_rot=0.75, p_shift=0)
    val_ds = H5PatchDataset(data_path, split="val", return_metadata=True, return_masks=False)

    def objective(trial):
        trial_seed = seed + trial.number
        set_seed(trial_seed)

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        base_channels = trial.suggest_categorical("base_channels", [8, 16, 32])
        sampler_type = trial.suggest_categorical("sampler_type", ["none", "fraction"])

        if sampler_type == "none":
            sampler_fraction = None
        else:
            sampler_fraction = trial.suggest_float("sampler_fraction", 0.1, 0.5)

        if sampler_fraction is not None:
            sampler = BalancedTrailSampler(train_ds.pos_indices, train_ds.neg_indices, pos_fraction=sampler_fraction)
        else:
            sampler = None

        model = TrailClassifier(base_channels=base_channels, dropout=dropout_rate)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = create_cos_lr_sched(optimizer, epochs)


        train_metrics = train_classifier(model, train_ds, val_ds, optimizer, scheduler, epochs, batch_size,
                                         sampler=sampler, num_workers=num_workers, trial=trial, seed=trial_seed)
        
        score = train_metrics['best_penalized_specificity']

        LOGGER.info(f"Trial {trial.number} complete: score={score:.6f} lr={learning_rate:.2e} dropout={dropout_rate:.3f} base_channels={base_channels} sampler_type={sampler_type} sampler_fraction={f'{sampler_fraction:.3f}' if sampler_fraction is not None else 'None'}")


        del model, optimizer, scheduler
        import gc
        gc.collect()
        torch.cuda.empty_cache()


        return score
    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Tune TrailClassifier Hyperparameters")
    
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
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
    
    study = optuna.create_study(study_name="classifier_arch_tuning", 
                                storage=f"sqlite:///{db_file_dir}/classifier_arch_tuning.db", 
                                load_if_exists=True, 
                                direction="maximize", 
                                sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))

    objective = create_objective(args.data_path, args.epochs, args.batch_size, args.num_workers, args.seed)
    study.optimize(objective, n_trials=args.trials)

    print("\n" + "="*30)
    print("CLASSIFIER ARCHITECTURE STUDY COMPLETE")
    print(f"Best Penalized Specificity: {study.best_value:.6f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*30)
