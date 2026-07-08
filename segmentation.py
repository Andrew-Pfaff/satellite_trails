import os
import argparse
import csv
import pprint
from pathlib import Path
from time import time

from PIL import Image
Image.MAX_IMAGE_PIXELS = 150000000
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from satellite_trail_segmentation.postprocess.hough import to_numpy_2d
from satellite_trail_segmentation.postprocess.pipeline import postprocess_segmentation
from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.unet_model.unet import UNet
from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint
from satellite_trail_segmentation.ml_utils.metrics import conf_counts_from_arrays, metrics_from_conf_counts
from satellite_trail_segmentation.utils.visualizations import plot_prediction_mask


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_UNET_CHECKPOINT = PROJECT_ROOT / "results" / "models" / "unet" / "unet_weights.pt"
DEFAULT_CLASSIFIER_CHECKPOINT = PROJECT_ROOT / "results" / "models" / "classifier" / "classifier_weights.pt"


class SatelliteTrailPipeline:
    """
    Runs full-field satellite trail segmentation from PNG input through patching,
    optional classifier filtering, U-Net prediction, postprocessing, evaluation,
    and output export.
    """

    def __init__(self, unet_model, classifier_model=None, patch_dim=528, device=None, timing=True, unet_batch_size=None, classifier_batch_size=None, num_workers=0):
        """
        Initializes model placement, patch settings, batch sizes, and DataLoader options.
        """

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            self.unet_batch_size = 32 if unet_batch_size is None else unet_batch_size
            self.classifier_batch_size = 128 if classifier_batch_size is None else classifier_batch_size
        else:
            self.unet_batch_size = 1 if unet_batch_size is None else unet_batch_size
            self.classifier_batch_size = 4 if classifier_batch_size is None else classifier_batch_size
        self.num_workers = num_workers

        self.classifier_model = classifier_model.to(self.device) if classifier_model else None
        self.unet_model = unet_model.to(self.device) if unet_model else None
        
        self.unet_model.eval()
        if self.classifier_model:
            self.classifier_model.eval()
        
        self.patch_dim = patch_dim
        self.timing = timing
        if self.timing:
            self.times = {}


    def _load_and_patch(self, image_path, normalization="source_zscore"):
        """
        Loads a PNG image, validates full-field dimensions, normalizes it, and splits it into model patches.
        """

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image path does not exist: {image_path}")
        valid_normalizations = {"source_zscore", "patch_zscore", "uint8"}
        if normalization not in valid_normalizations:
            raise ValueError(f"normalization must be one of {tuple(sorted(valid_normalizations))}, got {normalization!r}")
        
        with Image.open(image_path) as img:
            raw_img_array = np.array(img, dtype=np.float32)
            img_array = raw_img_array / 255.0
        
        original_shape = img_array.shape

        if (original_shape[0] % self.patch_dim != 0) or (original_shape[1] % self.patch_dim != 0):
            raise ValueError(f"Invalid image dimensions {original_shape} for patch dimension {self.patch_dim}. Both height and width must be perfectly divisible by {self.patch_dim}.")
        
        eps = 1e-6
        source_mean = raw_img_array.mean()
        source_std = raw_img_array.std()
        if source_std < eps:
            source_std = 1.0

        patches = []
        patch_indices = []

        for row in range(0, original_shape[0], self.patch_dim):   
            for col in range(0, original_shape[1], self.patch_dim):
                raw_patch = raw_img_array[row:row+self.patch_dim, col:col+self.patch_dim]
                if normalization == "source_zscore":
                    patch = (raw_patch - source_mean) / source_std
                elif normalization == "patch_zscore":
                    patch = raw_patch
                    patch_mean = patch.mean()
                    patch_std = patch.std()
                    if patch_std < eps:
                        patch_std = 1.0
                    patch = (patch - patch_mean) / patch_std
                else:
                    patch = img_array[row:row+self.patch_dim, col:col+self.patch_dim]
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
                patches.append(patch_tensor)
                patch_indices.append((row, col))
        
        return img_array, patches, patch_indices, original_shape


    def preprocessing(self, source_path, mask_path=None, normalization="source_zscore"):
        """
        Builds source patch data and optionally loads matching mask patch data for evaluation.
        """

        start_time = time()
        
        image, patches, patch_indicies, original_shape = self._load_and_patch(source_path, normalization=normalization)
        
        patch_data = {"source_patches":patches,  "source_indicies":patch_indicies, "shape":original_shape}

        return_dict = {"image": image, "patch_data":patch_data, "time": time()-start_time}

        if mask_path is not None:
            mask, mask_patches, mask_coords, mask_original_shape = self._load_and_patch(mask_path, normalization="uint8")
            if mask_original_shape != original_shape:
                raise ValueError(f"Given mask image does not match the shape of the input image.")
            
            patch_data["mask_patches"] = mask_patches
            patch_data["mask_coords"] = mask_coords
            return_dict["mask"] = mask

        return return_dict

    def segmentation(self, patch_data, use_classifier=True, unet_threshold=0.6, classifier_threshold=0.67, save_intermediate_results=False):
        """
        Runs optional classifier patch filtering followed by U-Net inference over selected full-field patches.
        """

        if use_classifier == True and self.classifier_model==None:
            raise RuntimeError("Classifier cannot be used without classifier model being loaded into the pipeline.")
        
        times = {}
        pred_data = {}
        start_time = time()


        data_loader_time = 0
        image_shape = patch_data["shape"]
        patches_list = patch_data["source_patches"]
        patch_indices = patch_data["source_indicies"]
        total_patches = len(patches_list)
    
        patches_tensor = torch.stack(patches_list, dim=0)
        coords_tensor =torch.tensor(patch_indices, dtype=torch.int32)

        segmentation_prediction = np.zeros(image_shape)
        
        data_loader_time += time() - start_time
        
        # Classifier
        if use_classifier:
            classifier_loader_time = time()
            cls_dataset = TensorDataset(patches_tensor, coords_tensor)
            cls_loader  = DataLoader(cls_dataset, batch_size=self.classifier_batch_size, shuffle=False, pin_memory=(self.device.type == "cuda"), num_workers=self.num_workers)
            
            imgs_to_segment_mask = torch.ones(total_patches, dtype=torch.bool)

            if save_intermediate_results:
                classifier_prediction = np.zeros((image_shape[0]//self.patch_dim,image_shape[1]//self.patch_dim), dtype=np.uint8)  
            
            data_loader_time += time() - classifier_loader_time

            start_classifier_time = time()
            with torch.no_grad():
                for images, coords in cls_loader:
                    logits = self.classifier_model(images.to(self.device))
                    probs = torch.sigmoid(logits).squeeze(1).cpu()
                    preds = (probs >= classifier_threshold).bool()

                    for i in range(len(preds)):
                        row0 = coords[i, 0]//self.patch_dim
                        col0 = coords[i, 1]//self.patch_dim
                        imgs_to_segment_mask[(row0 * image_shape[1]//self.patch_dim) + col0] = preds[i]
                        if save_intermediate_results:
                            classifier_prediction[row0,col0] = int(preds[i])
                            
            classifier_time = time() - start_classifier_time

        data_loader_time_unet = time()


        # UNet
        if use_classifier:
            unet_patches = patches_tensor[imgs_to_segment_mask]
            unet_coords  = coords_tensor[imgs_to_segment_mask]
        else:
            unet_patches = patches_tensor
            unet_coords = coords_tensor


        unet_dataset = TensorDataset(unet_patches, unet_coords)
        unet_loader  = DataLoader(unet_dataset, batch_size=self.unet_batch_size, shuffle=False, pin_memory=(self.device.type == "cuda"), num_workers=self.num_workers)

        data_loader_time += time() - data_loader_time_unet 


        start_unet = time()

        with torch.no_grad():
            for patches, coords in unet_loader:
                logits = self.unet_model(patches.to(self.device))
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                preds = (probs >= unet_threshold).astype(np.uint8)
                
                for i in range(len(preds)):
                    row0 = coords[i, 0]
                    col0 = coords[i, 1]
                    segmentation_prediction[row0 : row0 + self.patch_dim, col0 : col0 + self.patch_dim] = preds[i]

        unet_time = time() - start_unet


        times["segmentation_time"] = time()-start_time
        times["init_data_loaders"] = data_loader_time
        if use_classifier:
            times["classifier_time"] = classifier_time
        times["unet_time"] = unet_time
        
        pred_data["segmented_result"] = segmentation_prediction
        if use_classifier and save_intermediate_results:
            pred_data["classifier_result"] = classifier_prediction

        return pred_data, times


    def postprocessing(self, pred_mask, **postprocess_kwargs):
        """
        Applies postprocessing to a binary segmentation mask.

        Args:
            pred_mask (np.ndarray or torch.Tensor): Binary segmentation prediction mask.
            **postprocess_kwargs: Keyword arguments passed to postprocess_segmentation.

        Returns:
            tuple: Postprocessed mask, timing dictionary, and contour details. Contour
            details are None unless contour_details=True is passed.
        """

        times = {}
        start_time = time()

        hough_start = time()
        postprocess_result = postprocess_segmentation(pred_mask, **postprocess_kwargs)
        hough_time = time() - hough_start
        contour_details = None

        if isinstance(postprocess_result, tuple):
            postprocessed_image, contour_details = postprocess_result
        else:
            postprocessed_image = postprocess_result

        times["postprocessing_time"] = time()-start_time
        times["hough_transform"] = hough_time

        return postprocessed_image, times, contour_details


    def evaluate_masks(self, pred_mask, postprocessed_mask, real_mask):
        """
        Evaluates raw and postprocessed segmentation masks against a real mask.

        Args:
            pred_mask (np.ndarray or torch.Tensor): Raw segmentation prediction mask.
            postprocessed_mask (np.ndarray or torch.Tensor): Postprocessed prediction mask.
            real_mask (np.ndarray or torch.Tensor): Ground-truth binary mask.

        Returns:
            dict: Metrics for raw and postprocessed masks, plus metric deltas.
        """

        pred_mask = to_numpy_2d(pred_mask)
        postprocessed_mask = to_numpy_2d(postprocessed_mask)
        real_mask = to_numpy_2d(real_mask)

        if pred_mask.shape != real_mask.shape:
            raise ValueError(f"pred_mask shape {pred_mask.shape} does not match real_mask shape {real_mask.shape}")
        if postprocessed_mask.shape != real_mask.shape:
            raise ValueError(
                f"postprocessed_mask shape {postprocessed_mask.shape} does not match real_mask shape {real_mask.shape}"
            )

        pred_metrics = metrics_from_conf_counts(conf_counts_from_arrays(pred_mask, real_mask))
        postprocessed_metrics = metrics_from_conf_counts(conf_counts_from_arrays(postprocessed_mask, real_mask))

        metric_deltas = {}
        for key, value in postprocessed_metrics.items():
            if key in pred_metrics:
                metric_deltas[key] = value - pred_metrics[key]

        return {
            "prediction": pred_metrics,
            "postprocessed": postprocessed_metrics,
            "delta": metric_deltas,
        }
    

    def export_outputs(self, image, prediction, mask=None, metrics=None, export_png=False, export_plots=False, metrics_csv=False, output_dir="outputs", output_prefix="prediction"):
        """
        Saves requested postprocessing outputs.

        Args:
            image (np.ndarray): Source image array.
            prediction (np.ndarray): Final predicted mask.
            mask (np.ndarray, optional): Ground-truth mask array. Defaults to None.
            metrics (dict, optional): Metrics from evaluate_masks. Defaults to None.
            export_png (bool): Whether to save the predicted mask PNG. Defaults to False.
            export_plots (bool): Whether to save the image/prediction plot. Defaults to False.
            metrics_csv (bool): Whether to save metrics CSV when a mask is available. Defaults to False.
            output_dir (str or Path): Output directory. Defaults to "outputs".
            output_prefix (str): Output filename prefix. Defaults to "prediction".

        Returns:
            dict: Paths for saved outputs.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {}

        if export_png:
            mask_path = output_dir / f"{output_prefix}_mask.png"
            prediction_png = (to_numpy_2d(prediction) > 0).astype(np.uint8) * 255
            Image.fromarray(prediction_png).save(mask_path)
            output_paths["mask"] = str(mask_path)

        if export_plots:
            plot_path = output_dir / f"{output_prefix}_plot.png"
            plot_prediction_mask(image, prediction, mask=mask, save_path=plot_path)
            output_paths["plot"] = str(plot_path)

        if metrics_csv and mask is not None and metrics is not None:
            metrics_path = output_dir / f"{output_prefix}_metrics.csv"
            metric_stages = ["prediction", "postprocessed", "delta"]
            metric_names = sorted(
                {
                    metric_name
                    for stage in metric_stages
                    for metric_name in metrics.get(stage, {})
                }
            )
            with metrics_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["stage", *metric_names])
                writer.writeheader()
                for stage in metric_stages:
                    metric_group = metrics.get(stage)
                    if metric_group is not None:
                        row = {"stage": stage}
                        row.update(metric_group)
                        writer.writerow(row)
            output_paths["metrics"] = str(metrics_path)

        return output_paths


def main(unet_model, source_path, mask_path=None, classifier_model=None, 
         timing=True, use_classifier=True, normalization="source_zscore",
         unet_threshold=0.6, classifier_threshold=0.67, patch_dim=528,
         device=None, unet_batch_size=None, classifier_batch_size=None, num_workers=0,
         postprocess_config=None, export_png=True, export_plots=True, 
         metrics_csv=True, output_dir="outputs", output_prefix="prediction"):
    """
    Runs the complete single-image segmentation workflow and returns timing plus optional results.
    """

    times = {}
    pipeline = SatelliteTrailPipeline(unet_model, classifier_model=classifier_model, patch_dim=patch_dim, device=device, timing=timing, unet_batch_size=unet_batch_size, classifier_batch_size=classifier_batch_size, num_workers=num_workers)
    postprocess_config = {} if postprocess_config is None else postprocess_config
    
    preprocessing_data = pipeline.preprocessing(source_path, mask_path, normalization=normalization)
    times["preprocessing"] = preprocessing_data["time"]
    
    pred_data, segmentation_times = pipeline.segmentation(preprocessing_data["patch_data"], use_classifier=use_classifier, unet_threshold=unet_threshold, classifier_threshold=classifier_threshold, save_intermediate_results=True)
    unet_pred_image = pred_data["segmented_result"]
    times["segmentation"] = segmentation_times

    postprocessed_image, postprocess_time, contour_details = pipeline.postprocessing(
        unet_pred_image,
        **postprocess_config,
    )
    times["postprocessing"] = postprocess_time

    if mask_path is not None:
        results = pipeline.evaluate_masks(unet_pred_image, postprocessed_image, preprocessing_data["mask"])
        if contour_details is not None:
            results["contour_details"] = contour_details
        
        export_paths = pipeline.export_outputs(preprocessing_data["image"], postprocessed_image, mask=preprocessing_data["mask"], metrics=results, export_png=export_png, export_plots=export_plots, metrics_csv=metrics_csv, output_dir=output_dir, output_prefix=output_prefix)
        
        if export_paths:
            results["export_paths"] = export_paths
        
        return times, results

    export_paths = pipeline.export_outputs(preprocessing_data["image"], postprocessed_image, export_png=export_png, export_plots=export_plots, metrics_csv=False, output_dir=output_dir, output_prefix=output_prefix)

    if contour_details is not None:
        results = {"contour_details": contour_details}
        if export_paths:
            results["export_paths"] = export_paths
        return times, results
    
    if export_paths:
        return times, {"export_paths": export_paths}
    
    return times


def parse_args():
    """
    Parses command-line arguments for model loading, inference, postprocessing, and exports.
    """

    parser = argparse.ArgumentParser(description="Run satellite trail segmentation on one full-field PNG.")

    # Inputs
    parser.add_argument("--source-path", required=True, help="Path to the source full-field PNG.")
    parser.add_argument("--mask-path", default=None, help="Optional path to the ground-truth mask PNG.")

    # Models
    parser.add_argument("--unet-checkpoint", default=DEFAULT_UNET_CHECKPOINT, help="Path to the U-Net checkpoint or weights file.")
    parser.add_argument("--classifier-checkpoint", default=DEFAULT_CLASSIFIER_CHECKPOINT, help="Path to the classifier checkpoint or weights file.")
    parser.add_argument("--use-classifier", action="store_true", help="Run classifier patch filtering before U-Net.")

    # Inference config
    parser.add_argument("--normalization", choices=("source_zscore", "patch_zscore", "uint8"), default="source_zscore", help="Input image normalization mode.")
    parser.add_argument("--patch-dim", type=int, default=528, help="Patch size used for full-field tiling.")
    parser.add_argument("--device", default=None, help="Torch device string. Defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--unet-batch-size", type=int, default=None, help="Optional U-Net inference batch size.")
    parser.add_argument("--classifier-batch-size", type=int, default=None, help="Optional classifier inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--unet-threshold", type=float, default=0.6, help="U-Net probability threshold.")
    parser.add_argument("--classifier-threshold", type=float, default=0.67, help="Classifier probability threshold.")

    # Export config
    parser.add_argument("--export-png", dest="export_png", action="store_true", default=True, help="Save the final predicted mask PNG.")
    parser.add_argument("--no-export-png", dest="export_png", action="store_false", help="Do not save the final predicted mask PNG.")
    parser.add_argument("--export-plots", dest="export_plots", action="store_true", default=True, help="Save the image/prediction plot.")
    parser.add_argument("--no-export-plots", dest="export_plots", action="store_false", help="Do not save the image/prediction plot.")
    parser.add_argument("--metrics-csv", dest="metrics_csv", action="store_true", default=True, help="Save metrics CSV when a mask is provided.")
    parser.add_argument("--no-metrics-csv", dest="metrics_csv", action="store_false", help="Do not save metrics CSV.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    parser.add_argument("--output-prefix", default="prediction", help="Output filename prefix.")

    # Postprocess config
    parser.add_argument("--line-mode", choices=("asta", "centerline"), default="asta", help="Hough line drawing mode.")
    parser.add_argument("--width-mode", choices=("none", "contour_width", "median_sampled_width"), default="none", help="Width (when using centerline) or gap-fill (when using asta) mode.")
    parser.add_argument("--foreground-value", type=int, default=255, help="Foreground value for returned postprocessed mask.")
    parser.add_argument("--hough-threshold", type=int, default=50, help="Minimum Hough accumulator threshold.")
    parser.add_argument("--min-line-length", type=int, default=100, help="Minimum Hough line length.")
    parser.add_argument("--max-line-gap", type=int, default=250, help="Maximum Hough line gap and gap-fill length.")
    parser.add_argument("--line-cluster-angle-degrees", type=float, default=3, help="Maximum cluster orientation difference in degrees.")
    parser.add_argument("--line-cluster-distance", type=float, default=8, help="Maximum line cluster perpendicular distance in pixels.")
    parser.add_argument("--line-cluster-max-along-gap", type=float, default=250, help="Maximum along-line gap for grouping Hough segments into one centerline cluster.")
    parser.add_argument("--width-samples", type=int, default=9, help="Number of sampled width positions.")
    parser.add_argument("--max-width-search", type=int, default=25, help="Perpendicular width search radius in pixels.")
    parser.add_argument("--max-contour-distance", type=float, default=20, help="Maximum line-to-contour distance for contour width.")
    parser.add_argument("--min-fill-gap", type=int, default=10, help="Minimum gap length to fill, if the gap is smaller, it will not be filled.")
    parser.add_argument("--fallback-width", type=float, default=1, help="Fallback width when width estimation fails.")
    parser.add_argument("--morph-kernel-size", type=int, default=3, help="Morphological closing kernel size.")
    parser.add_argument("--min-component-size", type=int, default=500, help="Minimum connected component size to keep.")
    parser.add_argument("--no-contour-filter", dest="contour_filter", action="store_false", default=True, help="Disable ASTA-style final contour filtering.")
    parser.add_argument("--contour-area-threshold", type=float, default=3000, help="Minimum contour area kept by ASTA-style final contour filtering.")
    parser.add_argument("--contour-details", action="store_true", help="Return PNG-plane contour details.")
    parser.add_argument("--contour-min-area", type=float, default=10, help="Minimum contour area included in contour details.")

    return parser.parse_args()


def cli():
    """
    Command-line entry point for running the full segmentation pipeline on one PNG image.
    """

    args = parse_args()
    if args.use_classifier and args.classifier_checkpoint is None:
        raise ValueError("--classifier-checkpoint is required when --use-classifier is enabled")

    # Models
    device = None if args.device is None else torch.device(args.device)
    unet_model = UNet()
    load_checkpoint(args.unet_checkpoint, unet_model)

    classifier_model = None
    if args.use_classifier:
        classifier_model = TrailClassifier(base_channels=16)
        load_checkpoint(args.classifier_checkpoint, classifier_model)

    # Postprocess config
    postprocess_config = {
        "line_mode": args.line_mode,
        "width_mode": args.width_mode,
        "foreground_value": args.foreground_value,
        "hough_threshold": args.hough_threshold,
        "min_line_length": args.min_line_length,
        "max_line_gap": args.max_line_gap,
        "line_cluster_angle_degrees": args.line_cluster_angle_degrees,
        "line_cluster_distance": args.line_cluster_distance,
        "line_cluster_max_along_gap": args.line_cluster_max_along_gap,
        "width_samples": args.width_samples,
        "max_width_search": args.max_width_search,
        "max_contour_distance": args.max_contour_distance,
        "min_fill_gap": args.min_fill_gap,
        "fallback_width": args.fallback_width,
        "morph_kernel_size": args.morph_kernel_size,
        "min_component_size": args.min_component_size,
        "contour_filter": args.contour_filter,
        "contour_area_threshold": args.contour_area_threshold,
        "contour_details": args.contour_details,
        "contour_min_area": args.contour_min_area,
    }

    # Pipeline execution
    result = main(
        unet_model,
        args.source_path,
        mask_path=args.mask_path,
        classifier_model=classifier_model,
        use_classifier=args.use_classifier,
        normalization=args.normalization,
        unet_threshold=args.unet_threshold,
        classifier_threshold=args.classifier_threshold,
        patch_dim=args.patch_dim,
        device=device,
        unet_batch_size=args.unet_batch_size,
        classifier_batch_size=args.classifier_batch_size,
        num_workers=args.num_workers,
        postprocess_config=postprocess_config,
        export_png=args.export_png,
        export_plots=args.export_plots,
        metrics_csv=args.metrics_csv,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )

    pprint.pprint(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
