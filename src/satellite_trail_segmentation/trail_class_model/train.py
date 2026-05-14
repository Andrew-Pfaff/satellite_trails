import logging
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.trail_class_model.classifier import TrailClassifier
from satellite_trail_segmentation.trail_class_model.losses import recall_combo_loss
from satellite_trail_segmentation.trail_class_model.metrics import batch_metrics
from satellite_trail_segmentation.utils.visualizations import plot_loss_curves

LOGGER = logging.getLogger(__name__)


def train_classifier(model, train_ds, val_ds, optimizer, scheduler, epochs, batch_size, pos_weight, num_workers, save_path, trial):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_loss = []
    val_loss = []
    best_val_recall = float("-inf")

    LOGGER.info(f"Starting classifier training for {epochs} epochs on {device} with {len(train_loader)} train batches and {len(val_loader)} val batches.")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for images, metadata in train_loader:
            images = images.to(device)
            targets = metadata["patch_has_trail"].to(device=device, dtype=torch.float32).view(-1, 1)

            optimizer.zero_grad()
            logits = model(images)
            loss = recall_combo_loss(logits, targets, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

            epoch_train_loss += 1/len(train_loader) * loss.item()

        with torch.no_grad():
            model.eval()
            epoch_val_loss = 0
            val_true_positive = 0.0
            val_false_positive = 0.0
            val_false_negative = 0.0

            for images, metadata in val_loader:
                images = images.to(device)
                targets = metadata["patch_has_trail"].to(device=device, dtype=torch.float32).view(-1, 1)

                logits = model(images)
                loss = recall_combo_loss(logits, targets, pos_weight=pos_weight)
                epoch_val_loss += 1/len(val_loader) * loss.item()

                metrics = batch_metrics(logits, targets)
                val_true_positive += metrics["true_positive"]
                val_false_positive += metrics["false_positive"]
                val_false_negative += metrics["false_negative"]

        scheduler.step()

        epoch_val_recall = val_true_positive / (val_true_positive + val_false_negative + 1e-8)
        epoch_val_precision = val_true_positive / (val_true_positive + val_false_positive + 1e-8)
        epoch_val_fnr = val_false_negative / (val_true_positive + val_false_negative + 1e-8)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

        if trial is not None:
            trial.report(epoch_val_recall, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if epoch_val_recall > best_val_recall:
            best_val_recall = epoch_val_recall
            if save_path is not None:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "best_val_recall": best_val_recall,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "model_config": {
                            "in_channels": model.in_channels,
                            "kernel_size": model.kernel_size,
                            "base_channels": model.base_channels,
                            "dropout": model.dropout,
                        },
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    save_path,
                )

        final_epoch = epoch + 1
        LOGGER.info(f"Epoch {epoch + 1}/{epochs} | train_loss={epoch_train_loss:.6f} | val_loss={epoch_val_loss:.6f} | val_recall={epoch_val_recall:.6f} | val_precision={epoch_val_precision:.6f} | val_fnr={epoch_val_fnr:.6f}")

    return train_loss, val_loss, best_val_recall, final_epoch


def main(data_path, epochs, batch_size, learning_rate, dropout_rate, lr_decay, pos_weight, num_workers, save_path=None, base_channels=16, trial=None): # pragma: no cover.
    train_ds = H5PatchDataset(data_path, split="train", return_metadata=True, return_masks=False, augment=True, p_flip=0.1, p_rot=0.1, p_shift=0.1)
    val_ds = H5PatchDataset(data_path, split="val", return_metadata=True, return_masks=False)

    model = TrailClassifier(in_channels=1, base_channels=base_channels, dropout=dropout_rate)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate / lr_decay)

    train_loss, val_loss, best_val_recall, final_epoch = train_classifier(model, train_ds, val_ds, optimizer, scheduler, epochs, batch_size, pos_weight, num_workers, save_path, trial=trial)
    return train_loss, val_loss, best_val_recall, final_epoch


def parse_args():  # pragma: no cover.
    parser = argparse.ArgumentParser(description="Train satellite trail classifier model")

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--lr-decay", type=float, default=1e4)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__": # pragma: no cover.
    args = parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, 
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    train_loss, val_loss, best_val_recall, final_epoch = main(data_path=args.data_path,
                                                              epochs=args.epochs,
                                                              batch_size=args.batch_size,
                                                              learning_rate=args.learning_rate,
                                                              dropout_rate=args.dropout_rate,
                                                              lr_decay=args.lr_decay,
                                                              pos_weight=args.pos_weight,
                                                              num_workers=args.num_workers,
                                                              save_path=args.save_path,
                                                              base_channels=args.base_channels)

    if args.plot_path is not None:
        plot_loss_curves(train_loss, val_loss, args.plot_path)
