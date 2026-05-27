import os
import csv
import glob
import argparse
import random
import logging
from PIL import Image
Image.MAX_IMAGE_PIXELS = 150000000

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def _sort_images(image_dir, input_suffix=".fits_full.png", mask_suffix="_mask.png"):
    """
    Sorts the target/mask pairs into separate lists of file paths.

    Args:
        image_dir (str): Directory containing PNG files
        input_suffix (str): Suffix identifying image files
        mask_suffix (str): Suffix identifying mask files

    Returns:
        input_files (list): Sorted image file paths
        target_files (list): Corresponding target file paths
    """

    input_files = sorted(glob.glob(os.path.join(image_dir, f"*{input_suffix}")))
    target_files = sorted(glob.glob(os.path.join(image_dir, f"*{mask_suffix}")))

    if len(input_files) != len(target_files):
        raise ValueError(f"Found {len(input_files)} input files and {len(target_files)} target files in {image_dir}")

    LOGGER.info(f"Found {len(input_files)} input files and target files in {image_dir}")

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
    """
    Defines a stratified train/test/val split for dataset, and writes to csv which category each example fits into.

    Args:
        image_dir (str): Path to the dir with example data
        val_split (float): Ratio of validation data. Valid range: [0,1). Uses floor rounding when counting val examples
        test_split (float): Ratio of test data. Valid range: [0,1). Uses floor rounding when counting test examples. val_split+test_split must be less than 1
        output_path (str): Path to the written csv
        seed (int): Random seed for shuffling

    Returns:
        None. Writes split labels to a CSV at output_path.
    """

    if not 0 <= val_split < 1:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    if not 0 <= test_split < 1:
        raise ValueError(f"test_split must be in [0, 1), got {test_split}")
    if (val_split + test_split) >= 1:
        raise ValueError(f"val_split + test_split must be less than 1, got {val_split + test_split}")
    
    input_files, target_files = _sort_images(image_dir) 
    data_len = len(input_files)

    trail_counts = []
    for mask_path in target_files:
        with Image.open(mask_path) as mask_file:
            mask = np.asarray(mask_file, dtype=np.uint8)
        mask_patches = _create_patches(mask, patch_dim=528)
        trail_counts.append(int(np.sum(np.any(mask_patches > 0, axis=(1, 2)))))

    trail_counts = np.array(trail_counts)
    strata = np.digitize(trail_counts, np.percentile(trail_counts, [25, 50, 75]))

    indices = np.arange(data_len)
    num_val = int(data_len * val_split)
    num_test = int(data_len * test_split)
    num_temp = num_test + num_val

    train_idx, temp_idx, _, temp_strata = train_test_split(indices, strata, test_size=num_temp, random_state=seed, stratify=strata)
    val_idx, test_idx = train_test_split(temp_idx, test_size=num_test, random_state=seed, stratify=temp_strata)

    split_mask = np.zeros(data_len, dtype=np.uint8)
    split_mask[val_idx] = 1
    split_mask[test_idx] = 2

    train_count = int(np.sum(split_mask == 0))
    val_count = int(np.sum(split_mask == 1))
    test_count = int(np.sum(split_mask == 2))
    LOGGER.info(f"Split counts: train: {train_count} | val: {val_count} | test: {test_count}")


    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for input_f, target, split in zip(input_files, target_files, split_mask):
            writer.writerow([os.path.basename(input_f), os.path.basename(target), split])


def load_split(path):
    """
    Loads data split csv that defines which example data belong to test/train/val.

    Args:
        path (str): Path to the split defining csv

    Returns:
        train (list): List of the train images
        val (list): List of the val images
        test (list): List of the test images
    """
    
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
    """
    Takes a master list of train, test, and validation files, and prepares a subset of that data of defined size. 

    Args:
        train_files (list): List of train file names
        test_files (list): List of test file names
        val_files (list): List of validation file names
        num_images (int): Number of images to include in the subset. Must satisfy num_images<train_files+test_files+val_files
        val_split (float): Ratio of validation data. Uses floor rounding when counting val examples. Must satisfy len(val_files)>num_images*val_split
        test_split (float): Ratio of test data. Uses floor rounding when counting test examples. Must satisfy len(test_split)>num_images*test_split

    Returns:
        input_files (list): List of all input files included in the created data subset
        target_files (list): List of all target files included in the created data subset
        split_mask (list): List of the split label of each example. 0-train, 1-val, 2-test
    """
    
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

    Args:
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

    Reads paired image and mask PNG files, slices them into square patches, and writes the patches along with split labels and spatial metadata into a structured HDF5 file for use in training pipelines.
    Metadata includes:
    Datasets:
    - ``images`` (N, patch_dim, patch_dim) uint8: image patches
    - ``masks`` (N, patch_dim, patch_dim) uint8: mask patches
    - ``source_index`` (N,) int32: index of the full-field source_file for each patch
    - ``patch_has_trail`` (N,) uint8: 1 if patch contains trail pixels, else 0
    - ``patch_y0`` (N,) int32: top-left y coordinate of patch in source image
    - ``patch_x0`` (N,) int32: top-left x coordinate of patch in source image
    - ``source_files`` (n_images,) str: basenames of source image files
    - ``source_split`` (n_images,) uint8: split label (0=train, 1=val, 2=test)
    Attributes:
    - ``patch_dim``: patch side length in pixels
    - ``full_shape``: (H, W) shape of the source images

    Args:
        input_files (list): Sorted image file paths
        mask_files (list): Corresponding mask file paths
        split_mask (np.ndarray): Full-image split labels where 0=train, 1=val, 2=test
        output_path (str): Path to the output HDF5 file
        patch_dim (int): Side length of each square patch in pixels
        overwrite (bool): If True, overwrite an existing HDF5 file

    Returns:
        None. Writes H5 file to specified output_path

    Raises:
        FileExistsError: If output_path exists and overwrite is False.
        ValueError: If input_files and mask_files have different lengths.
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

    LOGGER.info(f"Wrote {total_patches} patches to {output_path}")


def parse_args(): # pragma: no cover.
    parser = argparse.ArgumentParser(description="Train satellite trail segmentation model")
    
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)  
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")

    
    return parser.parse_args()


if __name__ == "__main__": # pragma: no cover.
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    image_dir = args.data_path
    output_path = args.output_path
    master_split_mask_path = "data/h5s/master_split.csv"


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
