import argparse
import torch
import logging

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.data.sampler import FixedStepWeightedTrailSampler
from satellite_trail_segmentation.utils.visualizations import plot_loss_curves
from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.classifier_model.classifier_train_function import train_classifier
from satellite_trail_segmentation.ml_utils.lr_scheduler import create_cos_lr_sched
from satellite_trail_segmentation.ml_utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def main(data_path, learning_rate, sampler_fraction, base_channels, dropout_rate, warmup_epochs, eta_min, epochs, batch_size,
         pos_weight, fn_penalty_weight, min_recall, recall_penalty, weight_decay,
         num_workers, full_save_path=None, weight_save_path=None, seed=1,
         normalization="source_zscore", steps_per_epoch=800, grad_clip_max_norm=1.0,
         early_stopping_patience=None, early_stopping_min_delta=0.0, p_shift=0.0, min_shift=4, max_shift=20):
    
    set_seed(seed)

    train_ds = H5PatchDataset(data_path, split="train", return_metadata=True, return_masks=False, augment=True, p_shift=p_shift, min_shift=min_shift, max_shift=max_shift, normalization=normalization)
    val_ds = H5PatchDataset(data_path, split="val", return_metadata=True, return_masks=False, normalization=normalization)

    if sampler_fraction is not None:
        sampler = FixedStepWeightedTrailSampler(
            train_ds.pos_indices,
            train_ds.neg_indices,
            pos_fraction=sampler_fraction,
            num_samples=steps_per_epoch * batch_size,
        )
    else:
        sampler = None

    model = TrailClassifier(base_channels=base_channels, dropout=dropout_rate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = create_cos_lr_sched(optimizer, epochs, warmup_epochs=warmup_epochs, eta_min=eta_min)

    train_metrics = train_classifier(model, train_ds, val_ds, optimizer, scheduler, 
                                     epochs, batch_size, pos_weight=pos_weight, 
                                     fn_penalty_weight=fn_penalty_weight, 
                                     min_recall=min_recall, recall_penalty=recall_penalty, 
                                     sampler=sampler, num_workers=num_workers, 
                                     full_save_path=full_save_path, weight_save_path=weight_save_path, 
                                     trial=None, seed=seed, grad_clip_max_norm=grad_clip_max_norm,
                                     early_stopping_patience=early_stopping_patience,
                                     early_stopping_min_delta=early_stopping_min_delta)
    
    LOGGER.info(f"Training completed after {train_metrics['final_epoch']} epochs. Best (recall penalized) specificity: {train_metrics['best_penalized_specificity']:.2f}. Validation loss on that  epoch: {train_metrics['best_val_loss']:.2f}")

    return train_metrics['train_loss'], train_metrics['val_loss']


def parse_args():
    parser = argparse.ArgumentParser(description="Train satellite trail classifier model")

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--sampler-fraction", type=float, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=800)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--pos-weight", type=float, default=3.0)
    parser.add_argument("--fn-penalty-weight", type=float, default=1)
    parser.add_argument("--min-recall", type=float, default=0.98)
    parser.add_argument("--recall-penalty", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--full-save-path", type=str, default=None)
    parser.add_argument("--weight-save-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--normalization", type=str, default="source_zscore", choices=["source_zscore", "patch_zscore", "uint8"])
    parser.add_argument("--grad-clip-max-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--p-shift", type=float, default=0.0)
    parser.add_argument("--min-shift", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, 
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    train_loss, val_loss = main(data_path=args.data_path,
                                learning_rate=args.learning_rate,
                                sampler_fraction=args.sampler_fraction,
                                base_channels=args.base_channels,
                                dropout_rate=args.dropout_rate,
                                warmup_epochs=args.warmup_epochs,
                                eta_min=args.eta_min,
                                epochs=args.epochs,
                                batch_size=args.batch_size,                                            
                                pos_weight=args.pos_weight,
                                fn_penalty_weight=args.fn_penalty_weight,
                                min_recall=args.min_recall,
                                recall_penalty=args.recall_penalty,
                                weight_decay=args.weight_decay,
                                num_workers=args.num_workers,
                                full_save_path=args.full_save_path,
                                weight_save_path=args.weight_save_path,
                                seed=args.seed,
                                normalization=args.normalization,
                                steps_per_epoch=args.steps_per_epoch,
                                grad_clip_max_norm=args.grad_clip_max_norm,
                                early_stopping_patience=args.early_stopping_patience,
                                early_stopping_min_delta=args.early_stopping_min_delta,
                                p_shift=args.p_shift,
                                min_shift=args.min_shift,
                                max_shift=args.max_shift)

    if args.plot_path is not None:
        plot_loss_curves(train_loss, val_loss, args.plot_path)
