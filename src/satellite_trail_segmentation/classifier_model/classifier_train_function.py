import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import optuna

from satellite_trail_segmentation.ml_utils.loss_functions import bce_fn_penalty_loss
from satellite_trail_segmentation.ml_utils.metrics import init_conf_counts, update_conf_counts_batch, conf_counts_from_logits, metrics_from_conf_counts, specificity_with_recall_penalty, best_threshold_by_penalized_specificity
from satellite_trail_segmentation.ml_utils.checkpoints import save_checkpoint, save_weights
from satellite_trail_segmentation.ml_utils.seed import make_generator, seed_worker


LOGGER = logging.getLogger(__name__)


def train_classifier(model, train_ds, val_ds, optimizer, scheduler, 
                     epochs, batch_size, pos_weight=1.0, fn_penalty_weight=1.0, 
                     pred_thresholds=None, min_recall=0.98, recall_penalty=2.0, 
                     sampler=None, num_workers=0, full_save_path=None, 
                     weight_save_path=None, trial=None, seed=None, grad_clip_max_norm=1.0,
                     early_stopping_patience=None, early_stopping_min_delta=0.0):
    """
    Trains a classifier model for satellite trail detection over a specified number of epochs.

    Full training workflow: manages data loading, moves tensors to the appropriate device, executes the forward and backward passes, tracks loss profiles across training and validation splits, updates the learning rate schedule, and implements custom checkpoint saving alongside Optuna trial pruning.
    Uses a combo loss of weighted BCE and dice score. Saves on highest penalized specificity (specificity minus a penalty value when recall threshold is not met).

    Args:
        model (torch.nn.Module): The classifier neural network instance to be trained.
        train_ds (torch.utils.data.Dataset): The dataset object containing training images and ground-truth patch labels.
        val_ds (torch.utils.data.Dataset): The dataset object containing validation data.
        optimizer (torch.optim.Optimizer): The torch optimization object used to update network weights.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate schedule object stepped once per epoch.
        epochs (int): The total number of full training cycles to execute.
        batch_size (int): Number of training/validation samples to pass through the network per iteration.
        pos_weight (float): Positive class weighting factor passed to the classifier loss.
        fn_penalty_weight (float): Scaling factor for the soft false-negative penalty term.
        pred_thresholds (list): Probability thresholds used to binarize classifier outputs for metric tracking. 
        min_recall
        recall_penalty
        sampler
        num_workers (int): Number of asynchronous subprocesses to allocate for data loading.
        full_save_path (str): File path for saving the full model weights and config.
        save_path (str): File path for saving the model weights.
        trial (optuna.trial.Trial, optional): An active Optuna study trial hyperparameter hook used for validating metrics reporting and active epoch pruning. Defaults to None.
        seed (int): Random seed.
        grad_clip_max_norm (float, optional): Maximum gradient norm. Set to None or <=0 to disable clipping. Defaults to 1.0.
        early_stopping_patience (int, optional): Consecutive epochs without sufficient penalized-specificity improvement before stopping. Set to None to disable.
        early_stopping_min_delta (float, optional): Minimum penalized-specificity increase required to reset early-stopping patience. Defaults to 0.0.

    Returns:
        dict: training history and best validation summary. Includes:
            - train_loss (list): Train loss per epoch history.
            - val_loss (list): Validation loss per epoch history.
            - best_val_loss (float): Validation loss during epoch with the highest penalized specificity
            - val_penalized_specificity (list): Penalized specificity per epoch history.
            - best_penalized_specificity (float): Best penalized specificity score.
            - final_epoch (int): Number of epochs the model trained for.
    """

    if seed is not None:
        generator = make_generator(seed)
        worker_init_fn = seed_worker
    else:
        generator = None
        worker_init_fn = None


    if full_save_path is not None:
        Path(full_save_path).parent.mkdir(parents=True, exist_ok=True)
    if weight_save_path is not None:
        Path(weight_save_path).parent.mkdir(parents=True, exist_ok=True)

    if pred_thresholds is None:
        pred_thresholds = [0.5]
    pred_thresholds = [float(threshold) for threshold in pred_thresholds]

    sampler_config = None
    if sampler is not None:
        sampler_num_samples = getattr(sampler, "num_samples", None)
        if sampler_num_samples is None:
            sampler_num_samples = len(sampler)
        sampler_config = {
            "class_name": sampler.__class__.__name__,
            "pos_fraction": getattr(sampler, "pos_fraction", None),
            "num_samples": sampler_num_samples,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    model = model.to(device)

    use_workers = num_workers > 0
    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, persistent_workers=use_workers, worker_init_fn=worker_init_fn, generator=generator, prefetch_factor=2 if use_workers else None)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=use_workers, worker_init_fn=worker_init_fn, generator=generator, prefetch_factor=2 if use_workers else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=use_workers, worker_init_fn=worker_init_fn, generator=generator, prefetch_factor=2 if use_workers else None)

    train_loss = []
    val_loss = []
    val_penalized_specificity = []
    best_val_specificity = -float("inf")
    best_val_loss = float("inf")
    final_epoch = 0
    early_stopping_best_specificity = -float("inf")
    early_stopping_wait = 0
    use_early_stopping = early_stopping_patience is not None and early_stopping_patience > 0

    model_config = {"in_channels": model.in_channels, "kernel_size": model.kernel_size, "base_channels": model.base_channels, "dropout": model.dropout,
                    "pos_weight": pos_weight, "fn_penalty_weight": fn_penalty_weight, "pred_thresholds": pred_thresholds,
                    "loss": {"name": "bce_fn_penalty_loss", "pos_weight": pos_weight, "fn_penalty_weight": fn_penalty_weight},
                    "min_recall": min_recall, "recall_penalty": recall_penalty, "batch_size": batch_size, "seed": seed,
                    "normalization": getattr(train_ds, "normalization", None), "sampler": sampler_config,
                    "augmentation": {"p_flip": getattr(train_ds, "p_flip", None), "p_rot": getattr(train_ds, "p_rot", None)},
                    "grad_clip_max_norm": grad_clip_max_norm,
                    "early_stopping": {"patience": early_stopping_patience, "min_delta": early_stopping_min_delta}}
    
    LOGGER.info(f"Starting classifier training for {epochs} epochs on {device} with {len(train_loader)} train batches and {len(val_loader)} val batches.")

    scaler = GradScaler(device=device_type)

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        train_samples = 0
        for images, metadata in train_loader:
            images = images.to(device)
            targets = metadata["patch_has_trail"].to(device=device, dtype=torch.float32).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device_type):
                logits = model(images)
                loss = bce_fn_penalty_loss(logits, targets, pos_weight=pos_weight, fn_penalty_weight=fn_penalty_weight)

            scaler.scale(loss).backward()
            if grad_clip_max_norm is not None and grad_clip_max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()

            batch_size_actual = images.size(0)
            epoch_train_loss += loss.item() * batch_size_actual
            train_samples += batch_size_actual

        epoch_train_loss = epoch_train_loss / train_samples

        with torch.no_grad():
            model.eval()
            epoch_val_loss = 0
            val_samples = 0
            threshold_counts = {t: init_conf_counts() for t in pred_thresholds}

            for images, metadata in val_loader:
                images = images.to(device)
                targets = metadata["patch_has_trail"].to(device=device, dtype=torch.float32).view(-1, 1)

                with autocast(device_type=device_type):
                    logits = model(images)
                    loss = bce_fn_penalty_loss(logits, targets, pos_weight=pos_weight, fn_penalty_weight=fn_penalty_weight)
                
                batch_size_actual = images.size(0)
                epoch_val_loss += loss.item()* batch_size_actual
                val_samples += batch_size_actual

                for t in pred_thresholds:
                    batch_counts = conf_counts_from_logits(logits, targets, t)
                    update_conf_counts_batch(threshold_counts[t], batch_counts)

            epoch_val_loss = epoch_val_loss / val_samples

        scheduler.step()

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

        
        threshold_metrics = {t: metrics_from_conf_counts(counts) for t, counts in threshold_counts.items()}
        best_threshold, best_metrics = best_threshold_by_penalized_specificity(threshold_metrics, min_recall, recall_penalty)
        epoch_val_specificity = specificity_with_recall_penalty(best_metrics, min_recall, recall_penalty)
        val_penalized_specificity.append(epoch_val_specificity)

        is_best_specificity = epoch_val_specificity > best_val_specificity
        is_better_loss_at_max_specificity = (epoch_val_specificity >= best_val_specificity and epoch_val_loss < best_val_loss)
        if is_best_specificity or is_better_loss_at_max_specificity:
            best_val_specificity = epoch_val_specificity
            best_val_loss = epoch_val_loss
            
            if full_save_path is not None:
                save_metrics = {"best_val_specificity": best_val_specificity, "best_val_loss": best_val_loss, "val_recall": best_metrics["recall"], "val_specificity": best_metrics["specificity"]}
                save_checkpoint(full_save_path, model, optimizer, scheduler, sampler, epoch=epoch+1, metrics=save_metrics, model_config=model_config)
            if weight_save_path is not None:
                save_weights(weight_save_path, model, model_config)


        final_epoch = epoch + 1
        LOGGER.info(f"Epoch {epoch + 1}/{epochs} | penalized specificity={epoch_val_specificity:.4f} | val_fnr={best_metrics['fnr']:.4f} | train_loss={epoch_train_loss:.4f} | val_loss={epoch_val_loss:.4f} | val_recall={best_metrics['recall']:.4f} | val_specificity={best_metrics['specificity']:.4f} | best_thr={best_threshold:.2f}")
        

        if trial is not None:
            trial.report(epoch_val_specificity, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if use_early_stopping:
            if epoch_val_specificity > early_stopping_best_specificity + early_stopping_min_delta:
                early_stopping_best_specificity = epoch_val_specificity
                early_stopping_wait = 0
            else:
                early_stopping_wait += 1
                if early_stopping_wait >= early_stopping_patience:
                    LOGGER.info(f"Early stopping after {epoch + 1} epochs. No penalized-specificity improvement greater than {early_stopping_min_delta} for {early_stopping_patience} consecutive epochs.")
                    break
    
    return {"train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": float(best_val_loss),
            "val_penalized_specificity": val_penalized_specificity,
            "best_penalized_specificity": float(best_val_specificity),
            "final_epoch": final_epoch}
