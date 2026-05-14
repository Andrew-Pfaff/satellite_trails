import os
import csv
import glob
import argparse
import random
from PIL import Image
Image.MAX_IMAGE_PIXELS = 150000000

import h5py
import numpy as np


def _sort_images(image_dir, input_suffix=".fits_full.png", mask_suffix="_mask.png"):
    """
    Sorts the target/mask pairs into separate lists of file paths.

    Parameters:
        image_dir (str): Directory containing PNG files
        input_suffix (str): Suffix identifying image files
        mask_suffix (str): Suffix identifying mask files

    Returns:
        input_paths (list): Sorted image file paths
        mask_paths (list): Corresponding mask file paths
        
    """
    input_files = sorted(glob.glob(os.path.join(image_dir, f"*{input_suffix}")))
    target_files = sorted(glob.glob(os.path.join(image_dir, f"*{mask_suffix}")))

    if len(input_files) != len(target_files):
        raise ValueError(f"Found {len(input_files)} input files and {len(target_files)} target files in {image_dir}")

    print(f"Found {len(input_files)} input files and target files in {image_dir}")

    for input_file, target_file in zip(input_files, target_files):
        input_name = os.path.basename(input_file)
        target_name = os.path.basename(target_file)

        if not input_name.endswith(input_suffix):
            raise ValueError(f"Input file does not end with expected suffix '{input_suffix}': {input_file}")
        if not target_name.endswith(mask_suffix):
            raise ValueError(f"Target file does not end with expected suffix '{mask_suffix}': {target_file}")

        input_stem = input_name[: -len(input_suffix)]
        target_stem = target_name[: -len(mask_suffix)]
        if input_stem != target_stem:
            raise ValueError(f"Target and input file names do not match: {input_file} vs {target_file}")

    return input_files, target_files


def define_split(image_dir, val_split, test_split, output_path, seed=1):
    if not 0 <= val_split < 1:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    if not 0 <= test_split < 1:
        raise ValueError(f"test_split must be in [0, 1), got {test_split}")
    if (val_split + test_split) >= 1:
        raise ValueError(f"val_split + test_split must be less than 1, got {val_split + test_split}")
    
    input_files, target_files = _sort_images(image_dir) 

    data_len = len(input_files)

    rng = np.random.default_rng(seed)
    shuffled_indices = np.arange(data_len)
    rng.shuffle(shuffled_indices)

    num_val = int(data_len * val_split)
    num_test = int(data_len * test_split)

    split_mask = np.zeros(data_len, dtype=np.uint8)
    split_mask[shuffled_indices[:num_val]] = 1
    split_mask[shuffled_indices[num_val : num_val + num_test]] = 2

    train_count = int(np.sum(split_mask == 0))
    val_count = int(np.sum(split_mask == 1))
    test_count = int(np.sum(split_mask == 2))
    print(f"Split counts: train: {train_count} | val: {val_count} | test: {test_count}")


    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for input, target, split in zip(input_files, target_files, split_mask):
            writer.writerow([os.path.basename(input), os.path.basename(target), split])


def load_split(path):
    train = []
    val = []
    test = []

    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            pair = (row[0], row[1])
            if int(row[2]) == 0:
                train.append(pair)
            elif int(row[2]) == 1:
                val.append(pair)
            elif int(row[2]) == 2:
                test.append(pair)

    return train, val, test


def prepare_subset_data(train_files, test_files, val_files, num_images, val_split, test_split, seed=1):
    num_val = int(val_split*num_images)
    num_test = int(test_split*num_images)
    num_train = num_images - (num_val + num_test) 


    if num_val > len(val_files):
        raise ValueError(f"Validation split ({val_split}) requests more val images ({num_val}) than allowed by the initial data split ({len(val_files)}).")
    if num_test > len(test_files):
        raise ValueError(f"Test split ({test_split}) requests more test images ({num_test}) than allowed by the initial data split ({len(test_files)}).")
    if num_train < 1:
        raise ValueError(f"There must be train images within the split.")


    random.seed(seed)
    if num_train < len(train_files):
        train_files = random.sample(train_files, num_train)
    if num_val < len(val_files):
        val_files = random.sample(val_files, num_val)
    if num_test < len(test_files):
        test_files = random.sample(test_files, num_test)

    
    input_files = []
    target_files = []
    for file_pair in train_files:
        input_files.append(file_pair[0])
        target_files.append(file_pair[1])
    for file_pair in val_files:
        input_files.append(file_pair[0])
        target_files.append(file_pair[1])
    for file_pair in test_files:
        input_files.append(file_pair[0])
        target_files.append(file_pair[1])

    split_mask = np.concatenate([np.zeros(num_train, dtype=int), np.ones(num_val, dtype=int), np.full(num_test, 2)])

    return input_files, target_files, split_mask


def _create_patches(image, patch_dim=528):
    """
    Creates non-overlapping square patches from a 2D image.

    Parameters:
        image (np.ndarray): 2D image array
        patch_dim (int): Side length of each square patch in pixels

    Returns:
        patches (np.ndarray): Array of image patches with shape
            (num_patches, patch_dim, patch_dim)
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image array, got shape {image.shape}")
    if patch_dim <= 0:
        raise ValueError(f"patch_dim must be positive, got {patch_dim}")

    height, width = image.shape
    if height % patch_dim != 0 or width % patch_dim != 0:
        raise ValueError(f"Image shape {image.shape} is not evenly divisible by patch_dim={patch_dim}")

    patches = image.reshape(height // patch_dim, patch_dim, width // patch_dim, patch_dim).swapaxes(1, 2).reshape(-1, patch_dim, patch_dim)

    return patches


def create_h5(input_files, mask_files, split_mask, output_path, patch_dim=528, overwrite=False):
    """
    Creates an HDF5 file containing image patches, mask patches, and metadata.

    Parameters:
        input_files (list): Sorted image file paths
        mask_files (list): Corresponding mask file paths
        split_mask (np.ndarray): Full-image split labels where 0=train, 1=val, 2=test
        output_path (str): Path to the output HDF5 file
        patch_dim (int): Side length of each square patch in pixels
        overwrite (bool): If True, overwrite an existing HDF5 file
    """
    input_files = list(input_files)
    mask_files = list(mask_files)
    split_mask = np.asarray(split_mask, dtype=np.uint8)

    if len(input_files) != len(mask_files):
        raise ValueError(f"Found {len(input_files)} input files and {len(mask_files)} target files")
    if len(input_files) == 0:
        raise ValueError("No image and mask pairs were provided")
    if len(split_mask) != len(input_files):
        raise ValueError(f"split_mask length {len(split_mask)} does not match number of image pairs {len(input_files)}")
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output HDF5 already exists: {output_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    string_dtype = h5py.string_dtype(encoding="utf-8")

    with Image.open(input_files[0]) as image_file:
        first_image = np.asarray(image_file, dtype=np.uint8)
    with Image.open(mask_files[0]) as mask_file:
        first_mask = np.asarray(mask_file, dtype=np.uint8)

    if first_image.shape != first_mask.shape:
        raise ValueError(f"Image and mask shapes do not match: {input_files[0]} {first_image.shape} vs {mask_files[0]} {first_mask.shape}")

    full_shape = first_image.shape
    if len(full_shape) != 2:
        raise ValueError(f"Expected 2D source images, got shape {full_shape}")
    if full_shape[0] % patch_dim != 0 or full_shape[1] % patch_dim != 0:
        raise ValueError(f"Image shape {full_shape} is not evenly divisible by patch_dim={patch_dim}")

    patches_per_source = (full_shape[0] // patch_dim) * (full_shape[1] // patch_dim)
    patches_per_row = full_shape[1] // patch_dim
    patch_y0 = (np.arange(patches_per_source, dtype=np.int32) // patches_per_row) * patch_dim
    patch_x0 = (np.arange(patches_per_source, dtype=np.int32) % patches_per_row) * patch_dim
    total_patches = len(input_files) * patches_per_source

    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["patch_dim"] = patch_dim
        h5_file.attrs["full_shape"] = np.asarray(full_shape, dtype=np.int32)

        image_dataset = h5_file.create_dataset("images", shape=(total_patches, patch_dim, patch_dim), dtype=np.uint8)
        mask_dataset = h5_file.create_dataset("masks", shape=(total_patches, patch_dim, patch_dim), dtype=np.uint8)
        source_index_dataset = h5_file.create_dataset("source_index", shape=(total_patches,), dtype=np.int32)
        patch_has_trail_dataset = h5_file.create_dataset("patch_has_trail", shape=(total_patches,), dtype=np.uint8)
        patch_y0_dataset = h5_file.create_dataset("patch_y0", shape=(total_patches,), dtype=np.int32)
        patch_x0_dataset = h5_file.create_dataset("patch_x0", shape=(total_patches,), dtype=np.int32)
        h5_file.create_dataset("source_files", data=np.asarray([os.path.basename(path) for path in input_files], dtype=object), dtype=string_dtype)
        h5_file.create_dataset("source_split", data=split_mask)

        for source_index, (input_path, mask_path) in enumerate(zip(input_files, mask_files)):
            with Image.open(input_path) as image_file:
                image = np.asarray(image_file, dtype=np.uint8)
            with Image.open(mask_path) as mask_file:
                mask = np.asarray(mask_file, dtype=np.uint8)

            if image.shape != mask.shape:
                raise ValueError(f"Image and mask shapes do not match: {input_path} {image.shape} vs {mask_path} {mask.shape}")
            if image.shape != full_shape:
                raise ValueError(f"All images must share the same shape. Expected {full_shape}, got {image.shape} for {input_path}")

            image_patches = _create_patches(image, patch_dim=patch_dim)
            mask_patches = _create_patches(mask, patch_dim=patch_dim)

            num_patches = image_patches.shape[0]
            if num_patches != patches_per_source:
                raise ValueError(f"Expected {patches_per_source} patches per source, got {num_patches} for {input_path}")
            patch_has_trail = np.any(mask_patches > 0, axis=(1, 2)).astype(np.uint8)
            start = source_index * patches_per_source
            end = start + num_patches

            image_dataset[start:end] = image_patches
            mask_dataset[start:end] = mask_patches
            source_index_dataset[start:end] = source_index
            patch_has_trail_dataset[start:end] = patch_has_trail
            patch_y0_dataset[start:end] = patch_y0
            patch_x0_dataset[start:end] = patch_x0

    print(f"Wrote {total_patches} patches to {output_path}")


def parse_args(): # pragma: no cover.
    parser = argparse.ArgumentParser(description="Train satellite trail segmentation model")
    
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)  
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--num-images", type=int, default=None)

    
    return parser.parse_args()


if __name__ == "__main__": # pragma: no cover.
    args = parse_args()

    image_dir = args.data_path
    output_path = args.output_path
    master_split_mask_path = "data/master_split.csv"


    if not os.path.exists(master_split_mask_path):
        define_split(image_dir, args.val_split, args.test_split, master_split_mask_path)
    
    train_raw, val_raw, test_raw = load_split(master_split_mask_path)

    train_files = [(os.path.join(image_dir, f[0]), os.path.join(image_dir, f[1])) for f in train_raw]
    val_files = [(os.path.join(image_dir, f[0]), os.path.join(image_dir, f[1])) for f in val_raw]    
    test_files = [(os.path.join(image_dir, f[0]), os.path.join(image_dir, f[1])) for f in test_raw]


    if args.num_images == None:
        num_images = len(train_files) + len(val_files) + len(test_files)
    else:
        num_images = args.num_images

    input_files, target_files, split_mask = prepare_subset_data(train_files, test_files, val_files, num_images, args.val_split, args.test_split)


    create_h5(input_files, target_files, split_mask, args.output_path)