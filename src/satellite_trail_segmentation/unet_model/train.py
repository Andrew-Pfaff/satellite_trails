import logging
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import optuna

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.data.sampler import BalancedTrailSampler
from satellite_trail_segmentation.unet_model.unet import UNet
from satellite_trail_segmentation.unet_model.losses import combo_loss
from satellite_trail_segmentation.utils.visualizations import plot_loss_curves

LOGGER = logging.getLogger(__name__)


def train_unet(model, train_ds, val_ds, optimizer, scheduler, epochs, batch_size, sampler=None, num_workers=0, save_path=None, trial=None):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_loss = []
    val_loss = []
    best_loss = torch.inf

    LOGGER.info(f"Starting training for {epochs} epochs on {device} with {len(train_loader)} train batches and {len(val_loader)} val batches.")
    
    for epoch in range(epochs):        
        model.train()
        epoch_train_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = combo_loss(logits, masks)

            loss.backward()
            optimizer.step()
        
            epoch_train_loss += 1/len(train_loader) * loss.item()
        

        with torch.no_grad():
            model.eval()
            epoch_val_loss = 0
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = combo_loss(logits, masks)

                epoch_val_loss += 1/len(val_loader) * loss.item()

        scheduler.step()

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

        if trial is not None:
            trial.report(epoch_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()



        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss

            if save_path is not None:
                torch.save({"epoch": epoch + 1, "best_val_loss": best_loss, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                            "model_config": {"in_channels": model.in_channels, "out_channels": model.out_channels, "kernel_size": model.kernel_size, "base_channels": model.base_channels, "dropout": model.dropout},
                            "train_loss": train_loss, "val_loss": val_loss},
                            save_path)

        final_epoch = (epoch+1)
        LOGGER.info(f"Epoch {epoch + 1}/{epochs} | train loss={epoch_train_loss:.6f} | val_loss={epoch_val_loss:.6f}")

    return train_loss, val_loss, best_loss, final_epoch


def create_cos_lr_sched(optimizer, epochs, warmup_epochs=5, eta_min=1e-6):
    if warmup_epochs is not None and warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cos = CosineAnnealingLR(optimizer, T_max=(epochs-warmup_epochs), eta_min=eta_min)
        return SequentialLR(optimizer, schedulers=[warmup, cos], milestones=[warmup_epochs])
    else:
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)


def main(data_path, epochs, batch_size, learning_rate, dropout_rate, p_aug, num_workers, warmup_epochs, eta_min, sampler_fraction=None, save_path=None): # pragma: no cover.
    train_ds = H5PatchDataset(data_path, split="train", augment=True, p_flip=p_aug, p_rot=p_aug, p_shift=p_aug)
    val_ds = H5PatchDataset(data_path, split="val")

    if sampler_fraction is not None:
        sampler = BalancedTrailSampler(train_ds.pos_indices, train_ds.neg_indices, pos_fraction=sampler_fraction)
    else:
        sampler = None
    
    model = UNet(dropout=dropout_rate)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = create_cos_lr_sched(optimizer, epochs, warmup_epochs=warmup_epochs, eta_min=eta_min)

    train_loss, val_loss, _, _ = train_unet(model, train_ds, val_ds, optimizer, scheduler, epochs, batch_size, sampler=sampler, num_workers=num_workers, save_path=save_path)
    return train_loss, val_loss


def parse_args(): # pragma: no cover.
    parser = argparse.ArgumentParser(description="Train satellite trail segmentation model")
    
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--augmentation-prob", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--sampler-fraction", type=float, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__": # pragma: no cover.
    args = parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, 
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    train_loss, val_loss = main(data_path=args.data_path,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.learning_rate,
                                dropout_rate=args.dropout_rate,
                                p_aug=args.augmentation_prob,
                                warmup_epochs=args.warmup_epochs,
                                eta_min=args.eta_min,
                                sampler_fraction=args.sampler_fraction,
                                save_path=args.save_path,
                                num_workers=args.num_workers)
    
    if args.plot_path is not None:  
        plot_loss_curves(train_loss, val_loss, args.plot_path)