import logging
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import optuna

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.classifier_model.classifier import get_classifier_model, TinyGatekeeper
from satellite_trail_segmentation.classifier_model.losses import bce_fn_penalty_loss
from satellite_trail_segmentation.classifier_model.metrics import batch_metrics
from satellite_trail_segmentation.utils.visualizations import plot_loss_curves

LOGGER = logging.getLogger(__name__)


def train_classifier(model, train_ds, val_ds, optimizer, scheduler, epochs, 
                     batch_size, pos_weight, fn_penalty_weight, pred_threshold, 
                     num_workers, save_path, trial=None):
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_loss = []
    val_loss = []
    best_val_recall = 0.0
    min_val_loss = float("inf")


    LOGGER.info(f"Starting classifier training for {epochs} epochs on {device} with {len(train_loader)} train batches and {len(val_loader)} val batches.")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for images, metadata in train_loader:
            images = images.to(device)
            targets = metadata["patch_has_trail"].to(device=device, dtype=torch.float32).view(-1, 1)

            optimizer.zero_grad()
            logits = model(images)
            loss = bce_fn_penalty_loss(logits, targets, pos_weight=pos_weight, fn_penalty_weight=fn_penalty_weight)
            loss.backward()
            optimizer.step()

            epoch_train_loss += 1/len(train_loader) * loss.item()

        with torch.no_grad():
            model.eval()
            epoch_val_loss = 0
            val_true_positive = 0.0
            val_false_positive = 0.0
            val_false_negative = 0.0
            val_true_negative = 0.0

            for images, metadata in val_loader:
                images = images.to(device)
                targets = metadata["patch_has_trail"].to(device=device, dtype=torch.float32).view(-1, 1)

                logits = model(images)
                loss = bce_fn_penalty_loss(logits, targets, pos_weight=pos_weight, fn_penalty_weight=fn_penalty_weight)
                epoch_val_loss += 1/len(val_loader) * loss.item()

                metrics = batch_metrics(logits, targets, threshold=pred_threshold)
                val_true_positive += metrics["true_positive"]
                val_false_positive += metrics["false_positive"]
                val_false_negative += metrics["false_negative"]
                val_true_negative += metrics["true_negative"]

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

        is_best_recall = epoch_val_recall > best_val_recall
        is_better_loss_at_max_recall = (epoch_val_recall >= best_val_recall and epoch_val_loss < min_val_loss)

        if is_best_recall or is_better_loss_at_max_recall:
            best_val_recall = max(best_val_recall, epoch_val_recall)
            min_val_loss = epoch_val_loss
            if save_path is not None:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "best_val_recall": best_val_recall,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "model_config": {
                            "arch": "resnet18",
                            "pos_weight": pos_weight,
                        },
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    save_path,
                )

        final_epoch = epoch + 1
        LOGGER.info(f"Epoch {epoch + 1}/{epochs} | val_fnr={epoch_val_fnr:.6f} | train_loss={epoch_train_loss:.6f} | val_loss={epoch_val_loss:.6f} | val_recall={epoch_val_recall:.6f} | val_precision={epoch_val_precision:.6f}")

    return train_loss, val_loss, best_val_recall, final_epoch


def create_cos_lr_sched(optimizer, epochs, warmup_epochs=5, eta_min=1e-6):
    if warmup_epochs is not None and warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cos = CosineAnnealingLR(optimizer, T_max=(epochs-warmup_epochs), eta_min=eta_min)
        return SequentialLR(optimizer, schedulers=[warmup, cos], milestones=[warmup_epochs])
    else:
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    

def main(data_path, epochs, batch_size, learning_rate, pos_weight, num_workers, warmup_epochs, eta_min, p_shift, fn_penalty_weight, pred_threshold, save_path=None): # pragma: no cover.
    train_ds = H5PatchDataset(data_path, split="train", return_metadata=True, return_masks=False, augment=True, p_flip=0.5, p_rot=0.75, p_shift=p_shift)
    val_ds = H5PatchDataset(data_path, split="val", return_metadata=True, return_masks=False)

    model = TinyGatekeeper()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = create_cos_lr_sched(optimizer, epochs, warmup_epochs=warmup_epochs, eta_min=eta_min)

    train_loss, val_loss, best_val_recall, final_epoch = train_classifier(model, train_ds, val_ds, optimizer, scheduler, epochs, 
                                                                          batch_size, pos_weight, fn_penalty_weight, pred_threshold,
                                                                          num_workers, save_path, trial=None)
    return train_loss, val_loss, best_val_recall, final_epoch


def parse_args():  # pragma: no cover.
    parser = argparse.ArgumentParser(description="Train satellite trail classifier model")

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--p-shift", type=float, default=0.1)
    parser.add_argument("--fn-penalty-weight", type=float, default=1)
    parser.add_argument("--pred-threshold", type=float, default=0.3)
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
                                                              warmup_epochs=args.warmup_epochs,
                                                              eta_min=args.eta_min, 
                                                              pos_weight=args.pos_weight,
                                                              p_shift=args.p_shift,
                                                              fn_penalty_weight=args.fn_penalty_weight,
                                                              pred_threshold=args.pred_threshold,
                                                              num_workers=args.num_workers,
                                                              save_path=args.save_path)

    if args.plot_path is not None:
        plot_loss_curves(train_loss, val_loss, args.plot_path)
