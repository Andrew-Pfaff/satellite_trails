from PIL import Image

import pytest
import numpy as np

from satellite_trail_segmentation.data.preprocessing import _sort_images, define_split, load_split, prepare_subset_data, _create_patches, create_h5


def test_sort_images(dummy_image_dir):
    input_files, mask_files = _sort_images(str(dummy_image_dir))
    assert len(input_files) == 3
    assert len(mask_files) == 3
    assert all("fits_full.png" in f for f in input_files)
    assert all("_mask.png" in f for f in mask_files)


def test_sort_images_mismatch(dummy_image_dir):
    # Add an extra input image without a mask to trigger mismatch
    extra_img = dummy_image_dir / "img_3.fits_full.png"
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(extra_img)
    
    with pytest.raises(ValueError, match="Found 4 input files and 3 target files"):
        _sort_images(str(dummy_image_dir))


def test_create_patches():
    # 1056 x 1056 image with patch_dim = 528 should yield 4 patches
    dummy_image = np.zeros((1056, 1056), dtype=np.uint8)
    patches = _create_patches(dummy_image, patch_dim=528)
    assert patches.shape == (4, 528, 528)


def test_create_patches_invalid_shape():
    # 1000 x 1000 is not evenly divisible by 528
    dummy_image = np.zeros((1000, 1000), dtype=np.uint8)
    with pytest.raises(ValueError, match="is not evenly divisible by patch_dim"):
        _create_patches(dummy_image, patch_dim=528)


def test_define_and_load_split(dummy_image_dir, tmp_path):
    csv_path = tmp_path / "master_split.csv"
    
    # Using 3 images. Split ratio: ~33% val, ~33% test
    define_split(str(dummy_image_dir), val_split=0.34, test_split=0.34, output_path=str(csv_path), seed=1)
    
    assert csv_path.exists()
    
    train, val, test = load_split(str(csv_path))
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1
    assert len(train[0]) == 2 # tuple of (input, target)


def test_prepare_subset_data():
    # Dummy data
    train_files = [("t_in1", "t_out1"), ("t_in2", "t_out2")]
    val_files = [("v_in1", "v_out1")]
    test_files = [("te_in1", "te_out1")]
    
    input_files, target_files, split_mask = prepare_subset_data(
        train_files, test_files, val_files, 
        num_images=4, val_split=0.25, test_split=0.25
    )
    
    assert len(input_files) == 4
    assert len(target_files) == 4
    assert len(split_mask) == 4
    # Check mask content (train=0, val=1, test=2)
    assert np.sum(split_mask == 0) == 2
    assert np.sum(split_mask == 1) == 1
    assert np.sum(split_mask == 2) == 1


def test_create_h5(dummy_image_dir, tmp_path):
    input_files, mask_files = _sort_images(str(dummy_image_dir))
    split_mask = np.array([0, 1, 2]) # train, val, test
    h5_output = tmp_path / "output.h5"
    
    create_h5(input_files, mask_files, split_mask, str(h5_output), patch_dim=528)
    
    assert h5_output.exists()
    
    import h5py
    with h5py.File(h5_output, "r") as f:
        assert "images" in f
        assert "masks" in f
        assert "source_index" in f
        
        # 3 images * 4 patches per image (1056x1056 -> 4 patches of 528x528)
        assert f["images"].shape == (12, 528, 528)
        assert f["source_split"][:].tolist() == [0, 1, 2]