import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import optuna

from satellite_trail_segmentation.ml_utils.loss_functions import bce_fn_penalty_loss
from satellite_trail_segmentation.ml_utils.metrics import init_conf_counts, update_conf_counts_batch, conf_counts_from_logits, metrics_from_conf_counts, specificity_with_recall_penalty
from satellite_trail_segmentation.ml_utils.checkpoints import save_checkpoint, save_weights


LOGGER = logging.getLogger(__name__)


def train_classifier(model, train_ds, val_ds, optimizer, scheduler, 
                     epochs, batch_size, pos_weight=1.0, fn_penalty_weight=1.0, 
                     pred_threshold=0.5, min_recall=0.95, recall_penalty=1.0, 
                     sampler=None, num_workers=0, full_save_path=None, 
                     weight_save_path=None, trial=None):
    """
    Trains a classifier model for satellite trail detection over a specified number of epochs.

    Full training workflow: manages data loading, moves tensors to the appropriate device, executes the forward and backward passes, tracks loss profiles across training and validation splits, updates the learning rate schedule, and implements custom checkpoint saving alongside Optuna trial pruning.

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
        pred_threshold (float): Probability threshold used to binarize classifier outputs for metric tracking.
        num_workers (int): Number of asynchronous subprocesses to allocate for data loading.
        save_path (str): File path for saving the model.
        trial (optuna.trial.Trial, optional): An active Optuna study trial hyperparameter hook used for validating metrics reporting and active epoch pruning. Defaults to None.

    Returns:
        tuple: A 4-element tuple containing runtime performance tracking historical data:
            - train_loss (list of float): Cumulative running training loss calculated per epoch.
            - val_loss (list of float): Cumulative running validation loss calculated per epoch.
            - best_val_recall (float): The highest validation recall achieved across the run.
            - final_epoch (int): The absolute iteration index representing the final epoch reached prior to completion or pruning interception.
    """

    if full_save_path is not None:
        Path(full_save_path).parent.mkdir(parents=True, exist_ok=True)
    if weight_save_path is not None:
        Path(weight_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    model_config = {"in_channels": model.in_channels, "kernel_size": model.kernel_size, "base_channels": model.base_channels, "dropout": model.dropout,
                    "pos_weight": pos_weight, "fn_penalty_weight": fn_penalty_weight, "pred_threshold": pred_threshold}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    model = model.to(device)

    use_workers = num_workers > 0
    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, persistent_workers=use_workers, prefetch_factor=2 if use_workers else None)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=use_workers, prefetch_factor=2 if use_workers else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=use_workers, prefetch_factor=2 if use_workers else None)

    train_loss = []
    val_loss = []
    val_penalized_specificity = []
    best_val_specificity = 0.0
    best_val_loss = float("inf")
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
            threshold_counts = init_conf_counts()

            for images, metadata in val_loader:
                images = images.to(device)
                targets = metadata["patch_has_trail"].to(device=device, dtype=torch.float32).view(-1, 1)

                with autocast(device_type=device_type):
                    logits = model(images)
                    loss = bce_fn_penalty_loss(logits, targets, pos_weight=pos_weight, fn_penalty_weight=fn_penalty_weight)
                
                batch_size_actual = images.size(0)
                epoch_val_loss += loss.item()* batch_size_actual
                val_samples += batch_size_actual

                batch_counts = conf_counts_from_logits(logits, targets, pred_threshold)
                update_conf_counts_batch(threshold_counts, batch_counts)

            epoch_val_loss = epoch_val_loss / val_samples

        scheduler.step()

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

        epoch_metrics = metrics_from_conf_counts(threshold_counts)
        epoch_val_specificity = specificity_with_recall_penalty(epoch_metrics, min_recall, recall_penalty)
        val_penalized_specificity.append(epoch_val_specificity)

        is_best_specificity = epoch_val_specificity > best_val_specificity
        is_better_loss_at_max_specificity = (epoch_val_specificity >= best_val_specificity and epoch_val_loss < best_val_loss)
        if is_best_specificity or is_better_loss_at_max_specificity:
            best_val_specificity = epoch_val_specificity
            best_val_loss = epoch_val_loss
            
            if full_save_path is not None:
                save_metrics = {"best_val_specificity": best_val_specificity, "best_val_loss": best_val_loss, "val_recall": epoch_metrics["recall"], "val_specificity": epoch_metrics["specificity"]}
                save_checkpoint(full_save_path, model, optimizer, scheduler, epoch=epoch+1, metrics=save_metrics, model_config=model_config)
            if weight_save_path is not None:
                save_weights(weight_save_path, model, model_config)


        final_epoch = epoch + 1
        LOGGER.info(f"Epoch {epoch + 1}/{epochs} | val_fnr={epoch_metrics['fnr']:.6f} | train_loss={epoch_train_loss:.6f} | val_loss={epoch_val_loss:.6f} | val_recall={epoch_metrics['recall']:.6f} | val_specificity={epoch_metrics['specificity']:.6f} | val_precision={epoch_metrics['precision']:.6f}")
        

        if trial is not None:
            trial.report(epoch_val_specificity, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    return {"train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": float(best_val_loss),
            "val_penalized_specificity": val_penalized_specificity,
            "best_penalized_specificity": float(best_val_specificity),
            "final_epoch": final_epoch}