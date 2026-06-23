import argparse
import logging

import torch

from satellite_trail_segmentation.data.dataset import H5PatchDataset
from satellite_trail_segmentation.data.sampler import BalancedTrailSampler
from satellite_trail_segmentation.utils.visualizations import plot_loss_curves
from satellite_trail_segmentation.unet_model.unet import UNet
from satellite_trail_segmentation.unet_model.unet_train_function import train_unet
from satellite_trail_segmentation.ml_utils.lr_scheduler import create_cos_lr_sched
from satellite_trail_segmentation.ml_utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def main(data_path, epochs, batch_size, learning_rate, dropout_rate,
         pos_weight, bce_loss_factor, dice_loss_factor, label_smoothing, weight_decay, num_workers, warmup_epochs, 
         eta_min, sampler_fraction=None, full_save_path=None, weight_save_path=None, seed=1,
         normalization="source_zscore"):
    
    set_seed(seed)

    train_ds = H5PatchDataset(data_path, split="train", augment=True, p_flip=0.5, p_rot=0.75, normalization=normalization)
    val_ds = H5PatchDataset(data_path, split="val", normalization=normalization)

    if sampler_fraction is not None:
        sampler = BalancedTrailSampler(train_ds.pos_indices, train_ds.neg_indices, pos_fraction=sampler_fraction)
    else:
        sampler = None
    
    model = UNet(dropout=dropout_rate)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = create_cos_lr_sched(optimizer, epochs, warmup_epochs=warmup_epochs, eta_min=eta_min)

    train_metrics = train_unet(model, train_ds, val_ds, optimizer, scheduler,
                               epochs, batch_size, pos_weight=pos_weight, 
                               bce_loss_factor=bce_loss_factor, dice_loss_factor=dice_loss_factor,
                               label_smoothing=label_smoothing, sampler=sampler, num_workers=num_workers,
                               full_save_path=full_save_path, weight_save_path=weight_save_path, 
                               trial=None, seed=seed)
    
    LOGGER.info(f"Training completed after {train_metrics['final_epoch']} epochs. Best validation IOU: {train_metrics['best_iou']:.2f} with threshold {train_metrics['best_threshold']:.2f}. Validation loss for that epoch: {train_metrics['val_loss_at_best_iou']:.2f}")
    
    return train_metrics['train_loss'], train_metrics['val_loss']


def parse_args(): 
    parser = argparse.ArgumentParser(description="Train satellite trail segmentation model")
    
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--bce-loss-factor", type=float, default=0.5)
    parser.add_argument("--dice-loss-factor", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--sampler-fraction", type=float, default=None)
    parser.add_argument("--normalization", type=str, default="source_zscore", choices=["source_zscore", "patch_zscore", "uint8"])
    parser.add_argument("--full-save-path", type=str, default=None)
    parser.add_argument("--weight-save-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
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
                                pos_weight=args.pos_weight,
                                bce_loss_factor=args.bce_loss_factor,
                                dice_loss_factor=args.dice_loss_factor,
                                label_smoothing=args.label_smoothing,
                                weight_decay=args.weight_decay,
                                num_workers=args.num_workers,
                                warmup_epochs=args.warmup_epochs,
                                eta_min=args.eta_min,
                                sampler_fraction=args.sampler_fraction,
                                full_save_path=args.full_save_path,
                                weight_save_path=args.weight_save_path,
                                seed=args.seed,
                                normalization=args.normalization)
    
    if args.plot_path is not None:  
        plot_loss_curves(train_loss, val_loss, args.plot_path)
        
