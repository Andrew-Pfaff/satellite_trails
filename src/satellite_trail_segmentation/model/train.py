import logging
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.model.unet import UNet
from satellite_trail_segmentation.model.losses import combo_loss
from satellite_trail_segmentation.utils.visualizations import plot_loss_curves

LOGGER = logging.getLogger(__name__)


def train_unet(model, train_ds, val_ds, optimizer, scheduler, epochs, batch_size, save_path=None):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss

            if save_path is not None:
                torch.save({"epoch": epoch + 1, "best_val_loss": best_loss, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                            "model_config": {"in_channels": model.in_channels, "out_channels": model.out_channels, "kernel_size": model.kernel_size, "base_channels": model.base_channels, "dropout": model.dropout},
                            "train_loss": train_loss, "val_loss": val_loss},
                            save_path)

        LOGGER.info(f"Epoch {epoch + 1}/{epochs} | train loss={epoch_train_loss:.6f} | val_loss={epoch_val_loss:.6f}")

    return train_loss, val_loss


def main(data_path, epochs, batch_size, learning_rate, dropout_rate, lr_decay, save_path=None):
    train_ds = H5PatchDataset(data_path, split="train", augment=True, p_flip=0.1, p_rot=0.1, p_shift=0.1)
    val_ds = H5PatchDataset(data_path, split="val")
    
    model = UNet(in_channels=1, out_channels=1, kernel_size=3, base_channels=8, dropout=dropout_rate)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/lr_decay)

    train_loss, val_loss = train_unet(model, train_ds, val_ds, optimizer, scheduler, epochs, batch_size, save_path=save_path)
    return train_loss, val_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train satellite trail segmentation model")
    
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--lr-decay", type=float, default=1e4)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--plot-path", type=str, default=None)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, 
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    train_loss, val_loss = main(data_path=args.data_path,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.learning_rate,
                                dropout_rate=args.dropout_rate,
                                lr_decay=args.lr_decay,
                                save_path=args.save_path)
    
      
    plot_loss_curves(train_loss, val_loss, args.plot_path)