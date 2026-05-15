import os
import sys
import h5py
import numpy as np
import torch
from pathlib import Path
import time

from satellite_trail_segmentation.unet_model.unet import UNet
from satellite_trail_segmentation.unet_model.evaluate import recreate_full_field_pred as unet_recreate_full_field
from satellite_trail_segmentation.classifier_model.classifier import get_classifier_model, TrailClassifier
from satellite_trail_segmentation.classifier_model.evaluate import recreate_full_field as classifier_recreate_full_field


def create_synthetic_benchmark_h5(file_path):
    """
    Creates a dummy H5 file matching your 528x528 / 20x20 patch grid.
    """
    num_patches = 400  # 20x20 grid
    patch_size = 528   # Standard patch size
    full_field_dim = 10560
    
    print(f"Creating synthetic H5 for {num_patches} patches...")
    with h5py.File(file_path, "w") as f:
        f.attrs["full_shape"] = (full_field_dim, full_field_dim)
        
        # CHANGED: 'patches' -> 'images' to match dataset.py
        f.create_dataset("images", data=np.random.randn(num_patches, patch_size, patch_size).astype(np.float32))
        
        # ADDED: 'masks' because unet_recreate_full_field expects them in the loop
        f.create_dataset("masks", data=np.zeros((num_patches, patch_size, patch_size), dtype=np.float32))
        
        # REQUIRED: patch labels
        f.create_dataset("patch_has_trail", data=np.zeros(num_patches, dtype=int))
        
        # Source tracking
        f.create_dataset("source_index", data=np.zeros(num_patches, dtype=int))
        
        # Split tracking
        splits = np.zeros(10, dtype=int)
        splits[0] = 1 
        f.create_dataset("source_split", data=splits)
        
        # Coordinates for stitching
        coords = np.linspace(0, full_field_dim - patch_size, 20).astype(int)
        xv, yv = np.meshgrid(coords, coords)
        f.create_dataset("patch_x0", data=xv.flatten())
        f.create_dataset("patch_y0", data=yv.flatten())

# --- 4. BENCHMARK ENGINE ---
def run_benchmark():
    # 1. Detection Logic
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    
    device_name = torch.cuda.get_device_name(0) if is_cuda else "System CPU"
    
    print(f"\n" + "="*65)
    print(f"BENCHMARK DEVICE: {device_name}")
    print(f"Field Geometry: 528x528 | 20x20 Grid (400 patches)")
    print("="*65)
    
    temp_h5 = "benchmark_temp.h5"
    create_synthetic_benchmark_h5(temp_h5)

    # Initialize Models
    unet = UNet(in_channels=1, out_channels=1, base_channels=8).to(device).eval()
    resnet_gate = get_classifier_model().to(device).eval()
    custom_gate = TrailClassifier().to(device).eval()

    def time_field_reconstruction(fn, name, batch_size):
        # Warmup
        with torch.no_grad():
            _ = fn()
        
        if is_cuda:
            torch.cuda.synchronize()
        
        # Cross-platform timing using perf_counter
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = fn()
            
        if is_cuda:
            torch.cuda.synchronize()
            
        elapsed_s = time.perf_counter() - start_time
        
        print(f"{name:.<40} (BS={batch_size:<2}) {elapsed_s:.4f}s")
        return elapsed_s

    print("\nRunning full-field inference loops...")

    # 1. UNet Reconstruction (Baseline - usually BS=1 on laptop)
    t_unet = time_field_reconstruction(
        lambda: unet_recreate_full_field(unet, temp_h5, split_type="val", source_index=0, batch_size=1),
        "UNet Full-Field", batch_size=1
    )

    # 2. ResNet18 Gatekeeper (Use a manageable BS for laptop, e.g. 16 or 32)
    resnet_bs = 32 if is_cuda else 8 
    t_resnet = time_field_reconstruction(
        lambda: classifier_recreate_full_field(resnet_gate, temp_h5, split_type="val", source_index=0, batch_size=resnet_bs),
        "ResNet18 Gatekeeper", batch_size=resnet_bs
    )

    # 3. Custom TrailClassifier
    t_custom = time_field_reconstruction(
        lambda: classifier_recreate_full_field(custom_gate, temp_h5, split_type="val", source_index=0, batch_size=resnet_bs),
        "Custom TrailClassifier", batch_size=resnet_bs
    )

    # --- 5. RESULTS SUMMARY ---
    print("\n" + "="*65)
    print("PERFORMANCE COMPARISON")
    print(f"  Custom vs. ResNet-18 Speedup: {t_resnet/t_custom:.2f}x")
    print(f"  Custom vs. UNet Speedup:      {t_unet/t_custom:.2f}x")
    print("-" * 65)
    
    trail_pos_rate = 0.08
    pipeline_cost = t_custom + (trail_pos_rate * t_unet)
    print(f"  Projected Pipeline Speedup:   {t_unet/pipeline_cost:.2f}x faster than UNet alone")
    print("="*65)

    if os.path.exists(temp_h5):
        os.remove(temp_h5)


if __name__ == "__main__":
    run_benchmark()