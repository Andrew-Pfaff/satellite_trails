import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import optuna

from satellite_trail_segmentation.ml_utils.loss_functions import combo_loss
from satellite_trail_segmentation.ml_utils.metrics import init_conf_counts, update_conf_counts_batch, conf_counts_from_logits, metrics_from_conf_counts, threshold_sweep_from_logits, best_threshold_by_metric
from satellite_trail_segmentation.ml_utils.checkpoints import save_checkpoint, save_weights
from satellite_trail_segmentation.ml_utils.seed import make_generator, seed_worker


LOGGER = logging.getLogger(__name__)


def train_unet(model, train_ds, val_ds, optimizer, scheduler, 
               epochs, batch_size, pos_weight=1.0, bce_loss_factor=0.5, 
               dice_loss_factor=0.5, iou_thresholds = None, 
               sampler=None, num_workers=0, full_save_path=None, 
               weight_save_path=None, trial=None, seed=None):
    """
    Trains a UNet model for satellite trail segmentation over a specified number of epochs.

    Full training workflow: manages data loading, moves tensors to the appropriate device, executes the forward and backward passes, tracks loss profiles across training and validation splits, updates the learning rate schedule, and implements custom checkpoint saving alongside Optuna trial pruning.

    Args:
        model (torch.nn.Module): The UNet neural network instance to be trained.
        train_ds (torch.utils.data.Dataset): The dataset object containing training images and ground-truth semantic masks.
        val_ds (torch.utils.data.Dataset): The dataset object containing validation data.
        optimizer (torch.optim.Optimizer): The torch optimizatier object to update network weights.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate schedule object stepped once per epoch.
        epochs (int): The total number of full training cycles to execute.
        batch_size (int): Number of training/validation samples to pass through the network per iteration.
        pos_weight(float): Positive class weighting factor in BCE loss.
        bce_loss_factor (float): Weight of the BCE loss.
        dice_loss_factor (float): Weight of the Dice loss.
        iou_thresholds (list): List of thresholds at which metrics are calculated during training.
        sampler (torch.utils.data.Sampler, optional): Sampler strategy (e.g., BalancedTrailSampler) to set balance of positive and negative data samples. Defaults to None.
        num_workers (int, optional): Number of asynchronous subprocesses to allocate for data loading. Defaults to 0.
        save_path (str, optional): File path for saving the model. Defaults to None.
        trial (optuna.trial.Trial, optional): An active Optuna study trial hyperparameter hook used for validating metrics reporting and active epoch pruning. Defaults to None.
        seed (int): Random seed.
        
    Returns:
        dict: training history and best validation summary. Includes:
            - train_loss (list): Train loss per epoch history.
            - val_loss (list): Validation loss per epoch history.
            - val_loss_at_best_iou (float): Validation loss during epoch with the best IOU.
            - val_iou (list):  Validation max IOU per epoch history.
            - best_iou (float): Best max validation IOU over the range of thresholds.
            - best_threshold (float): Threshold for the best IOU value in the best IOU epoch.
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
    
    model_config = {"in_channels": model.in_channels, "out_channels": model.out_channels, "kernel_size": model.kernel_size, "base_channels": model.base_channels, "dropout": model.dropout,
                    "pos_weight": pos_weight, "bce_loss_factor": bce_loss_factor, "dice_loss_factor": dice_loss_factor}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type 
    model = model.to(device)
    
    use_workers = num_workers > 0
    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, persistent_workers=use_workers, worker_init_fn=worker_init_fn, generator=generator, prefetch_factor=2 if use_workers else None)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=use_workers, worker_init_fn=worker_init_fn, generator=generator, prefetch_factor=2 if use_workers else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=use_workers, worker_init_fn=worker_init_fn, generator=generator, prefetch_factor=2 if use_workers else None)

    if iou_thresholds is None:
        iou_thresholds = [0.5]
    train_loss = []
    val_loss = []
    val_iou = []
    best_loss = float("inf")
    best_iou = -float("inf")
    best_threshold = 0

    LOGGER.info(f"Starting training for {epochs} epochs on {device} with {len(train_loader)} train batches and {len(val_loader)} val batches.")
    
    scaler = GradScaler(device=device_type)

    for epoch in range(epochs):        
        model.train()
        epoch_train_loss = 0
        train_samples = 0 
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)

            
            with autocast(device_type=device_type):
                logits = model(images)
                loss = combo_loss(logits, masks, pos_weight, bce_loss_factor, dice_loss_factor)

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
            threshold_counts = {threshold: init_conf_counts() for threshold in iou_thresholds}
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                with autocast(device_type=device_type):
                    logits = model(images)
                    loss = combo_loss(logits, masks, pos_weight, bce_loss_factor, dice_loss_factor)

                batch_size_actual = images.size(0)
                epoch_val_loss += loss.item() * batch_size_actual
                val_samples += batch_size_actual

                for threshold in iou_thresholds:
                    batch_counts = conf_counts_from_logits(logits, masks, threshold)
                    update_conf_counts_batch(threshold_counts[threshold], batch_counts)
            
            epoch_val_loss = epoch_val_loss / val_samples

        scheduler.step()

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

        epoch_threshold_metrics = {threshold: metrics_from_conf_counts(counts) for threshold, counts in threshold_counts.items()}
        epoch_best_threshold, epoch_best_metrics =  best_threshold_by_metric(epoch_threshold_metrics, "iou")
        epoch_best_iou = epoch_best_metrics["iou"]
        val_iou.append(epoch_best_iou)
        
        if epoch_best_iou > best_iou:
            best_iou = epoch_best_iou
            best_threshold = epoch_best_threshold
            best_loss = epoch_val_loss
            
            if full_save_path is not None:
                save_metrics = {'best_iou': best_iou, 'best_threshold': best_threshold, 'val_loss_at_best_iou': best_loss}
                save_checkpoint(full_save_path, model, optimizer, scheduler, epoch=epoch+1, metrics=save_metrics, model_config=model_config)
            if weight_save_path is not None:
                save_weights(weight_save_path, model, model_config)
                

        final_epoch = (epoch+1)
        LOGGER.info(f"Epoch {epoch + 1}/{epochs} | train_loss={epoch_train_loss:.4f} | val_loss={epoch_val_loss:.4f} | val_iou={epoch_best_iou:.4f} (thr={epoch_best_threshold:.2f}) | best_iou={best_iou:.4f}")


        if trial is not None:
            trial.report(epoch_best_iou, epoch)
            if trial.should_prune():
                LOGGER.warning(f"Trial pruned by Optuna at epoch {epoch + 1}")
                raise optuna.exceptions.TrialPruned()

    return {"train_loss": train_loss,
            "val_loss": val_loss,
            "val_loss_at_best_iou": float(best_loss),
            "val_iou": val_iou,
            "best_iou": float(best_iou),
            "best_threshold": float(best_threshold),
            "final_epoch": final_epoch}
