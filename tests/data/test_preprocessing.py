import h5py
import numpy as np
import pytest
from PIL import Image

from satellite_trail_segmentation.data.preprocessing import (
    _clean_mask,
    _create_patches,
    _sort_images,
    create_h5,
    define_split,
    load_split,
    prepare_subset_data,
)


def _write_png(path, array):
    Image.fromarray(array.astype(np.uint8)).save(path)


def test_sort_images_pairs_by_basename(dummy_image_dir):
    input_files, mask_files = _sort_images(str(dummy_image_dir))
    assert [f.split("/")[-1].replace(".fits_full.png", "") for f in input_files] == [
        f.split("/")[-1].replace("_mask.png", "") for f in mask_files
    ]


def test_sort_images_mismatched_names(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(img_dir / "a.fits_full.png")
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(img_dir / "b_mask.png")
    with pytest.raises(ValueError, match="do not match"):
        _sort_images(str(img_dir))


def test_define_split_validates_and_is_deterministic(split_image_dir_528, tmp_path):
    csv_path = tmp_path / "split.csv"
    define_split(str(split_image_dir_528), val_split=0.25, test_split=0.25, output_path=str(csv_path), seed=7)
    first = csv_path.read_text()
    define_split(str(split_image_dir_528), val_split=0.25, test_split=0.25, output_path=str(csv_path), seed=7)
    assert csv_path.read_text() == first
    train, val, test = load_split(str(csv_path))
    assert len(train) + len(val) + len(test) == 4
    assert (len(train), len(val), len(test)) == (2, 1, 1)

    with pytest.raises(ValueError, match="val_split must be in"):
        define_split(str(split_image_dir_528), val_split=-0.1, test_split=0.2, output_path=str(csv_path))
    with pytest.raises(ValueError, match="test_split must be in"):
        define_split(str(split_image_dir_528), val_split=0.1, test_split=1.0, output_path=str(csv_path))
    with pytest.raises(ValueError, match="must be less than 1"):
        define_split(str(split_image_dir_528), val_split=0.6, test_split=0.4, output_path=str(csv_path))


def test_create_patches_preserves_order_and_values():
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    patches = _create_patches(image, patch_dim=2)
    expected = np.array(
        [
            [[0, 1], [4, 5]],
            [[2, 3], [6, 7]],
            [[8, 9], [12, 13]],
            [[10, 11], [14, 15]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(patches, expected)


def test_create_patches_rejects_bad_input():
    with pytest.raises(ValueError, match="Expected a 2D image"):
        _create_patches(np.zeros((2, 2, 2), dtype=np.uint8), patch_dim=2)
    with pytest.raises(ValueError, match="patch_dim must be positive"):
        _create_patches(np.zeros((2, 2), dtype=np.uint8), patch_dim=0)
    with pytest.raises(ValueError, match="not evenly divisible"):
        _create_patches(np.zeros((3, 4), dtype=np.uint8), patch_dim=2)


def test_clean_mask_removes_small_objects():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1, 1] = 1
    mask[3:6, 3:6] = 1
    cleaned = _clean_mask(mask, max_size=2)
    assert cleaned[1, 1] == 0
    assert cleaned[4, 4] == 1
    np.testing.assert_array_equal(_clean_mask(np.zeros((3, 3), dtype=np.uint8)), np.zeros((3, 3), dtype=np.uint8))


def test_clean_mask_preserves_diagonal_connectivity():
    mask = np.eye(4, dtype=np.uint8) * 255
    cleaned = _clean_mask(mask, max_size=3)

    np.testing.assert_array_equal(cleaned, mask)


def test_prepare_subset_data_is_deterministic():
    train_files = [("t0", "m0"), ("t1", "m1"), ("t2", "m2")]
    val_files = [("v0", "mv0"), ("v1", "mv1")]
    test_files = [("x0", "mx0"), ("x1", "mx1")]
    first_inputs, first_targets, first_split = prepare_subset_data(train_files, test_files, val_files, num_images=5, val_split=0.2, test_split=0.2, seed=3)
    second_inputs, second_targets, second_split = prepare_subset_data(train_files, test_files, val_files, num_images=5, val_split=0.2, test_split=0.2, seed=3)
    assert first_inputs == second_inputs
    assert first_targets == second_targets
    np.testing.assert_array_equal(first_split, second_split)


def test_prepare_subset_data_validation_errors():
    with pytest.raises(ValueError, match="Validation split"):
        prepare_subset_data([("t", "m")], [("x", "m")], [], num_images=3, val_split=0.34, test_split=0.0)
    with pytest.raises(ValueError, match="Test split"):
        prepare_subset_data([("t", "m")], [], [("v", "m")], num_images=3, val_split=0.0, test_split=0.34)
    with pytest.raises(ValueError, match="There must be train images"):
        prepare_subset_data([("t", "m")], [("x", "m")], [("v", "m")], num_images=2, val_split=0.5, test_split=0.5)


def test_create_h5_writes_expected_metadata(dummy_image_dir, tmp_path):
    input_files, mask_files = _sort_images(str(dummy_image_dir))
    split_mask = np.array([0, 1, 2, 0], dtype=np.uint8)
    h5_output = tmp_path / "output.h5"
    create_h5(input_files, mask_files, split_mask, str(h5_output), patch_dim=16, overwrite=True)

    with h5py.File(h5_output, "r") as f:
        assert set(f.keys()) == {
            "images",
            "masks",
            "patch_has_trail",
            "patch_x0",
            "patch_y0",
            "source_files",
            "source_index",
            "source_mean",
            "source_split",
            "source_std",
        }
        assert f["images"].shape == (16, 16, 16)
        assert f["masks"].shape == (16, 16, 16)
        assert f["images"].dtype == np.uint8
        assert f["masks"].dtype == np.uint8
        assert f["source_index"].dtype == np.int32
        assert f["source_mean"].dtype == np.float32
        assert f["source_std"].dtype == np.float32
        assert f["patch_y0"].dtype == np.int32
        assert f["patch_x0"].dtype == np.int32
        assert f["source_split"].dtype == np.uint8
        assert f["source_files"].shape == (4,)
        assert f["source_mean"].shape == (4,)
        assert f["source_split"].shape == (4,)
        assert f["source_std"].shape == (4,)
        assert f["patch_has_trail"].shape == (16,)
        assert f["source_split"][:].tolist() == [0, 1, 2, 0]
        assert f.attrs["patch_dim"] == 16
        assert tuple(f.attrs["full_shape"]) == (32, 32)
        assert f["source_files"].asstr()[:].tolist() == [f"img_{idx}.fits_full.png" for idx in range(4)]
        assert f["source_index"][:].tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
        assert f["patch_y0"][:].tolist() == [0, 0, 16, 16] * 4
        assert f["patch_x0"][:].tolist() == [0, 16, 0, 16] * 4
        assert f["patch_has_trail"].dtype == np.uint8
        assert f["patch_has_trail"][:].tolist() == [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
        expected_means = []
        expected_stds = []
        for input_file in input_files:
            with Image.open(input_file) as image_file:
                image = np.asarray(image_file, dtype=np.uint8)
            expected_means.append(float(image.mean()))
            expected_stds.append(float(image.std()))
        np.testing.assert_allclose(f["source_mean"][:], expected_means, rtol=1e-6)
        np.testing.assert_allclose(f["source_std"][:], expected_stds, rtol=1e-6)

        with Image.open(input_files[0]) as image_file:
            first_image = np.asarray(image_file, dtype=np.uint8)
        expected_mask_top_left = np.zeros((16, 16), dtype=np.uint8)
        expected_mask_top_left[:8, :8] = 255
        np.testing.assert_array_equal(f["images"][0], first_image[:16, :16])
        np.testing.assert_array_equal(f["images"][1], first_image[:16, 16:32])
        np.testing.assert_array_equal(f["masks"][0], expected_mask_top_left)
        np.testing.assert_array_equal(f["masks"][5], np.zeros((16, 16), dtype=np.uint8))
        np.testing.assert_array_equal(f["masks"][7], np.full((16, 16), 255, dtype=np.uint8))

    with pytest.raises(FileExistsError, match="already exists"):
        create_h5(input_files, mask_files, split_mask, str(h5_output), patch_dim=16)

    create_h5(input_files, mask_files, split_mask, str(h5_output), patch_dim=16, overwrite=True)


def test_create_h5_rejects_bad_inputs(tmp_path):
    image = np.zeros((16, 16), dtype=np.uint8)
    mask = np.zeros_like(image)
    image_path = tmp_path / "image.fits_full.png"
    mask_path = tmp_path / "image_mask.png"
    _write_png(image_path, image)
    _write_png(mask_path, mask)

    with pytest.raises(ValueError, match="Found 1 input files and 0 target files"):
        create_h5([str(image_path)], [], np.array([0], dtype=np.uint8), str(tmp_path / "mismatch.h5"), patch_dim=16)
    with pytest.raises(ValueError, match="No image and mask pairs"):
        create_h5([], [], np.array([], dtype=np.uint8), str(tmp_path / "empty.h5"), patch_dim=16)
    with pytest.raises(ValueError, match="split_mask length"):
        create_h5([str(image_path)], [str(mask_path)], np.array([0, 1], dtype=np.uint8), str(tmp_path / "split_len.h5"), patch_dim=16)

    bad_mask_path = tmp_path / "bad_shape_mask.png"
    _write_png(bad_mask_path, np.zeros((8, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="Image and mask shapes do not match"):
        create_h5([str(image_path)], [str(bad_mask_path)], np.array([0], dtype=np.uint8), str(tmp_path / "shape.h5"), patch_dim=16)

    second_image_path = tmp_path / "second.fits_full.png"
    second_mask_path = tmp_path / "second_mask.png"
    _write_png(second_image_path, np.zeros((32, 16), dtype=np.uint8))
    _write_png(second_mask_path, np.zeros((32, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="All images must share the same shape"):
        create_h5(
            [str(image_path), str(second_image_path)],
            [str(mask_path), str(second_mask_path)],
            np.array([0, 1], dtype=np.uint8),
            str(tmp_path / "inconsistent.h5"),
            patch_dim=16,
        )

    nondiv_image_path = tmp_path / "nondiv.fits_full.png"
    nondiv_mask_path = tmp_path / "nondiv_mask.png"
    _write_png(nondiv_image_path, np.zeros((17, 16), dtype=np.uint8))
    _write_png(nondiv_mask_path, np.zeros((17, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="not evenly divisible"):
        create_h5([str(nondiv_image_path)], [str(nondiv_mask_path)], np.array([0], dtype=np.uint8), str(tmp_path / "nondiv.h5"), patch_dim=16)
