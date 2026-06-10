import h5py
import numpy as np
import pytest


@pytest.fixture
def dummy_h5_file(tmp_path):
    h5_path = tmp_path / "dummy_dataset.h5"
    patch_dim = 4
    full_shape = (8, 8)

    images = np.array(
        [
            np.zeros((4, 4), dtype=np.uint8),
            np.full((4, 4), 5, dtype=np.uint8),
            np.array([[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 0, 0], [4, 4, 4, 4]], dtype=np.uint8),
            np.full((4, 4), 9, dtype=np.uint8),
            np.array([[10, 10, 10, 10], [10, 20, 20, 10], [10, 20, 20, 10], [10, 10, 10, 10]], dtype=np.uint8),
            np.full((4, 4), 12, dtype=np.uint8),
            np.array([[13, 13, 13, 13], [13, 30, 30, 13], [13, 30, 30, 13], [13, 13, 13, 13]], dtype=np.uint8),
            np.full((4, 4), 15, dtype=np.uint8),
            np.array([[16, 16, 16, 16], [16, 40, 40, 16], [16, 40, 40, 16], [16, 16, 16, 16]], dtype=np.uint8),
            np.full((4, 4), 18, dtype=np.uint8),
            np.array([[19, 19, 19, 19], [19, 50, 50, 19], [19, 50, 50, 19], [19, 19, 19, 19]], dtype=np.uint8),
            np.full((4, 4), 21, dtype=np.uint8),
        ]
    )
    masks = np.array(
        [
            np.zeros((4, 4), dtype=np.uint8),
            np.ones((4, 4), dtype=np.uint8),
            np.eye(4, dtype=np.uint8),
            np.flipud(np.eye(4, dtype=np.uint8)),
            np.zeros((4, 4), dtype=np.uint8),
            np.ones((4, 4), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
            np.ones((4, 4), dtype=np.uint8),
            np.triu(np.ones((4, 4), dtype=np.uint8)),
            np.zeros((4, 4), dtype=np.uint8),
            np.ones((4, 4), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
        ]
    )
    source_split = np.array([0, 1, 2], dtype=np.uint8)
    source_index = np.array([0] * 4 + [1] * 4 + [2] * 4, dtype=np.int32)
    patch_has_trail = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    patch_y0 = np.array([0, 0, 4, 4] * 3, dtype=np.int32)
    patch_x0 = np.array([0, 4, 0, 4] * 3, dtype=np.int32)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("masks", data=masks)
        f.create_dataset("source_split", data=source_split)
        f.create_dataset("source_index", data=source_index)
        f.create_dataset("patch_has_trail", data=patch_has_trail)
        f.create_dataset("patch_y0", data=patch_y0)
        f.create_dataset("patch_x0", data=patch_x0)
        f.create_dataset("source_files", data=np.array([b"source_0.png", b"source_1.png", b"source_2.png"]))
        f.attrs["patch_dim"] = patch_dim
        f.attrs["full_shape"] = full_shape

    return str(h5_path)
