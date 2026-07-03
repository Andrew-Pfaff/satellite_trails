import os
from time import time

from PIL import Image
Image.MAX_IMAGE_PIXELS = 150000000
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from satellite_trail_segmentation.postprocess.hough import to_numpy_2d
from satellite_trail_segmentation.postprocess.pipeline import postprocess_segmentation
from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.unet_model.unet import UNet
from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint
from satellite_trail_segmentation.ml_utils.metrics import conf_counts_from_arrays, metrics_from_conf_counts
from satellite_trail_segmentation.utils.visualizations import plot_segmentation_postprocess_comparison

class SatelliteTrailPipeline:
    def __init__(self, unet_model, classifier_model=None, patch_dim=528, device=None, timing=True, unet_batch_size=None, classifier_batch_size=None, num_workers=0):
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


    def postprocessing(self, pred_mask):
        times = {}
        start_time = time()

        hough_start = time()
        postprocessed_image = postprocess_segmentation(pred_mask)
        hough_time = time() - hough_start

        times["postprocessing_time"] = time()-start_time
        times["hough_transform"] = hough_time
        
        return postprocessed_image, times


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
    

    # def export_outputs(self, export_png=True, export_plots=True, output_dir="outputs"):
    #     pass


def main(unet_model,source_path, mask_path=None, classifier_model=None, timing=True, use_classifier=True, normalization="source_zscore"):
    times = {}
    pipeline = SatelliteTrailPipeline(unet_model, classifier_model=classifier_model, timing=timing)
    
    preprocessing_data = pipeline.preprocessing(source_path, mask_path, normalization=normalization)
    times["preprocessing"] = preprocessing_data["time"]
    
    pred_data, segmentation_times = pipeline.segmentation(preprocessing_data["patch_data"], use_classifier=use_classifier, save_intermediate_results=True)
    unet_pred_image = pred_data["segmented_result"]
    times["segmentation"] = segmentation_times

    postprocessed_image, postprocess_time = pipeline.postprocessing(unet_pred_image)
    times["postprocessing"] = postprocess_time

    # pipeline.export_outputs(preprocessing_data["image"], pred_data, postprocessed_image show_classifier=use_classifier)

    if mask_path is not None:
        plot_segmentation_postprocess_comparison(preprocessing_data["image"], unet_pred_image, postprocessed_image, preprocessing_data["mask"], save_path='postprocess.png')
        return times, pipeline.evaluate_masks(unet_pred_image, postprocessed_image, preprocessing_data["mask"])

    return times

if __name__ == "__main__":
    source_path ="/home/andrew/project/satellite_trails/data/png/ML1_20200318_024625_red.fits_full.png"
    mask_path ="/home/andrew/project/satellite_trails/data/png/ML1_20200318_024625_red_mask.png"

    unet_model = UNet()
    load_checkpoint("/home/andrew/project/satellite_trails/results/models/unet/unet_weights.pt", unet_model)

    classifier_model = TrailClassifier(base_channels=16)
    load_checkpoint("/home/andrew/project/satellite_trails/results/models/classifier/classifier_weights.pt", classifier_model)

    times, results = main(unet_model, source_path, mask_path=mask_path, classifier_model=classifier_model, timing=True, use_classifier=False)

    print(times)
    print()
    print(results)
