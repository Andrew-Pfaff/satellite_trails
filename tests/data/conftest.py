import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_patch():
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    return image, mask


@pytest.fixture
def dummy_image_dir(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    rng = np.random.default_rng(123)
    base = np.zeros((32, 32), dtype=np.uint8)

    for idx in range(4):
        image = base.copy()
        mask = np.zeros_like(image)
        image[:, idx * 4 : idx * 4 + 4] = rng.integers(10, 240, size=(32, 4), dtype=np.uint8)
        if idx % 2 == 0:
            mask[:8, :8] = 1
            mask[20, 1] = 1
        else:
            mask[16:, 16:] = 1
            mask[1, 20] = 1

        Image.fromarray(image).save(img_dir / f"img_{idx}.fits_full.png")
        Image.fromarray(mask * 255).save(img_dir / f"img_{idx}_mask.png")

    return img_dir


@pytest.fixture
def split_image_dir_528(tmp_path):
    img_dir = tmp_path / "split_images"
    img_dir.mkdir()

    for idx in range(4):
        image = np.full((528, 528), idx * 20, dtype=np.uint8)
        mask = np.zeros_like(image)
        mask[idx : idx + 64, idx : idx + 64] = 255

        Image.fromarray(image).save(img_dir / f"field_{idx}.fits_full.png")
        Image.fromarray(mask).save(img_dir / f"field_{idx}_mask.png")

    return img_dir
