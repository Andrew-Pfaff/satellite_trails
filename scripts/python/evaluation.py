import argparse
import logging

from satellite_trail_segmentation.unet_model.evaluate import evaluate_patches, recreate_full_field_pred, evaluate_full_field_pred
from satellite_trail_segmentation.utils.load_model import load_model_weights
from satellite_trail_segmentation.utils.visualizations import plot_full_field, plot_roc_curve, plot_pred_residual

LOGGER = logging.getLogger(__name__)


def main(model_path, data_path, full_field_index, plot_save_path_ff, plot_save_path_roc, plot_save_path_residual, batch_size, subsample_fraction, patch_dim):
    model = load_model_weights(model_path)

    LOGGER.info("Evaluating patches on val set...")
    test_loss, metrics, roc_data = evaluate_patches(
        model=model,
        h5_path=data_path,
        split_type="val",
        batch_size=batch_size,
        subsample_fraction=subsample_fraction,
    )
    LOGGER.info(f"Patch evaluation complete | loss={test_loss:.6f} | AUC={metrics['roc_auc']:.3f} | optimal_threshold={metrics['optimal_threshold']:.3f}")

    LOGGER.info(f"Reconstructing full-field prediction for source_index={full_field_index}...")
    full_image, full_pred, full_mask = recreate_full_field_pred(
        model=model,
        h5_path=data_path,
        split_type="val",
        source_index=full_field_index,
        batch_size=1,
        patch_dim=patch_dim,
    )

    metrics_ff, roc_data_ff, full_image, full_pred, full_mask = evaluate_full_field_pred(
        full_image=full_image,
        full_pred=full_pred,
        full_mask=full_mask,
        threshold=metrics["optimal_threshold"],
    )
    LOGGER.info(f"Full-field evaluation complete | AUC={metrics_ff['roc_auc']:.3f}")

    LOGGER.info("Saving plots...")
    plot_full_field(full_image, full_pred, full_mask, save_path=plot_save_path_ff, threshold=metrics["optimal_threshold"])
    plot_roc_curve(**roc_data, roc_auc=metrics["roc_auc"], optimal_threshold=metrics["optimal_threshold"], save_path=plot_save_path_roc)
    plot_pred_residual(full_pred, full_mask, save_path=plot_save_path_residual)
    LOGGER.info("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate satellite trail segmentation model")

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--full-field-index", type=int, required=True)
    parser.add_argument("--plot-save-path-ff", type=str, required=True)
    parser.add_argument("--plot-save-path-roc", type=str, required=True)
    parser.add_argument("--plot-save-path-residual", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--subsample-fraction", type=float, default=0.01)
    parser.add_argument("--patch-dim", type=int, default=528)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    main(model_path=args.model_path,
         data_path=args.data_path,
         full_field_index=args.full_field_index,
         plot_save_path_ff=args.plot_save_path_ff,
         plot_save_path_roc=args.plot_save_path_roc,
         plot_save_path_residual=args.plot_save_path_residual,
         batch_size=args.batch_size,
         subsample_fraction=args.subsample_fraction,
         patch_dim=args.patch_dim)