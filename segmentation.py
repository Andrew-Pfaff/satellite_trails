import os
from time import time

from PIL import Image
Image.MAX_IMAGE_PIXELS = 150000000
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# from satellite_trail_segmentation.
from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.unet_model.unet import UNet
from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint

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


    def _load_and_patch(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image path does not exist: {image_path}")
        
        with Image.open(image_path) as img:
            img_array = np.array(img, dtype=np.float32) / 255.0
        
        original_shape = img_array.shape

        if (original_shape[0] % self.patch_dim != 0) or (original_shape[1] % self.patch_dim != 0):
            raise ValueError(f"Invalid image dimensions {original_shape} for patch dimension {self.patch_dim}. Both height and width must be perfectly divisible by {self.patch_dim}.")
        
        patches = []
        patch_indices = []

        for row in range(0, original_shape[0], self.patch_dim):   
            for col in range(0, original_shape[1], self.patch_dim):
                patch = img_array[row:row+self.patch_dim, col:col+self.patch_dim]
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
                patches.append(patch_tensor)
                patch_indices.append((row, col))
        
        return img_array, patches, patch_indices, original_shape


    def preprocessing(self, source_path, mask_path=None):
        start_time = time()
        
        image, patches, patch_indicies, original_shape = self._load_and_patch(source_path)
        
        patch_data = {"source_patches":patches,  "source_indicies":patch_indicies, "shape":original_shape}

        if mask_path is not None:
            mask_patches, mask_coords, mask_original_shape = self._load_and_patch(mask_path)
            if mask_original_shape != original_shape:
                raise ValueError(f"Given mask image does not match the shape of the input image.")
            
            patch_data["mask_patches"] = mask_patches
            patch_data["mask_coords"] = mask_coords

        return image, patch_data, time()-start_time


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


    def postprocessing(self):
        times = {}
        start_time = time()

        times["postprocessing_time"] = time()-start_time
        # times["hough_transform"] = hough_time
        # times["contour"] = contour_time
        return times

    # def export_outputs(self, export_png=True, export_plots=True, output_dir="outputs"):
    #     pass


def main(unet_model, source_path, mask_path=None, classifier_model=None, timing=True, use_classifier=True):
    times = {}
    pipeline = SatelliteTrailPipeline(unet_model, classifier_model=classifier_model, timing=timing)
    
    image, patch_data, preprocess_time = pipeline.preprocessing(source_path, mask_path)
    times["preprocessing"] = preprocess_time
    
    pred_data, segmentation_times = pipeline.segmentation(patch_data, use_classifier=use_classifier, save_intermediate_results=True)
    times["segmentation"] = segmentation_times

    postprocess_time = pipeline.postprocessing()
    times["postprocessing"] = postprocess_time

    pipeline.export_outputs(image, pred_data, show_classifier=use_classifier)

    return times

if __name__ == "__main__":
    source_path ="/home/andrew/project/satellite_trails/data/png/ML1_20200318_024625_red.fits_full.png"

    unet_model = UNet()
    load_checkpoint("/home/andrew/project/satellite_trails/results/models/small_train/unet_weights.pt", unet_model)

    classifier_model = TrailClassifier(base_channels=16)
    load_checkpoint("/home/andrew/project/satellite_trails/results/models/classifier/archive_noisy_data/classifier_weights.pt", classifier_model)

    times = main(unet_model, source_path, mask_path=None, classifier_model=classifier_model, timing=True)

    print(times)