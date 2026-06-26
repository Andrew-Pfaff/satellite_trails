import numpy as np

import satellite_trail_segmentation.data.augmentation as augmentation
from satellite_trail_segmentation.data.augmentation import augment_image, flip, rotate, shift


def test_flip_matches_numpy(monkeypatch):
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    mask = (image % 2).astype(np.uint8)
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)

    aug_image, aug_mask = flip(image, mask)

    np.testing.assert_array_equal(aug_image, np.flip(image, axis=0))
    np.testing.assert_array_equal(aug_mask, np.flip(mask, axis=0))
    assert aug_image.dtype == image.dtype
    assert aug_mask.dtype == mask.dtype


def test_rotate_matches_numpy(monkeypatch):
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    mask = (image % 2).astype(np.uint8)

    for k in (1, 2, 3):
        monkeypatch.setattr(np.random, "randint", lambda *args, k=k, **kwargs: k)
        aug_image, aug_mask = rotate(image, mask)
        np.testing.assert_array_equal(aug_image, np.rot90(image, k))
        np.testing.assert_array_equal(aug_mask, np.rot90(mask, k))


def test_shift_matches_expected_direction(monkeypatch):
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 1
    fill_value = 99

    def expected_shift(direction):
        expected_image = np.full_like(image, fill_value)
        expected_mask = np.zeros_like(mask)
        if direction == 0:
            expected_image[:, 1:] = image[:, :-1]
            expected_mask[:, 1:] = mask[:, :-1]
        elif direction == 1:
            expected_image[1:, :] = image[:-1, :]
            expected_mask[1:, :] = mask[:-1, :]
        elif direction == 2:
            expected_image[:, :-1] = image[:, 1:]
            expected_mask[:, :-1] = mask[:, 1:]
        else:
            expected_image[:-1, :] = image[1:, :]
            expected_mask[:-1, :] = mask[1:, :]
        return expected_image, expected_mask

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 1)
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: np.full(kwargs["size"], fill_value))

    for direction in range(4):
        monkeypatch.setattr(np.random, "choice", lambda values, direction=direction: direction)
        aug_image, aug_mask = shift(image, mask, min_shift=1, max_shift=1)
        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape
        expected_image, expected_mask = expected_shift(direction)
        np.testing.assert_array_equal(aug_image, expected_image)
        np.testing.assert_array_equal(aug_mask, expected_mask)
        assert aug_mask.sum() <= mask.sum()


def test_shift_returns_original_when_no_safe_direction(monkeypatch):
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 1] = 1
    mask[1, 0] = 1
    mask[2, -1] = 1
    mask[-1, 2] = 1

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 1)

    aug_image, aug_mask = shift(image, mask, min_shift=1, max_shift=1)

    np.testing.assert_array_equal(aug_image, image)
    np.testing.assert_array_equal(aug_mask, mask)


def test_augment_image_composes_in_order(monkeypatch):
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    mask = (image > 3).astype(np.uint8)
    calls = []

    rand_values = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr(np.random, "rand", lambda *args, **kwargs: next(rand_values))

    def fake_flip(image, mask):
        calls.append("flip")
        return image + 10, mask + 1

    def fake_rotate(image, mask):
        calls.append("rotate")
        return image * 2, mask + 2

    def fake_shift(image, mask, min_shift, max_shift):
        calls.append(("shift", min_shift, max_shift))
        return image - 3, mask + 3

    monkeypatch.setattr(augmentation, "flip", fake_flip)
    monkeypatch.setattr(augmentation, "rotate", fake_rotate)
    monkeypatch.setattr(augmentation, "shift", fake_shift)

    aug_image, aug_mask = augment_image(image, mask, p_flip=1, p_rot=1, p_shift=1)

    assert calls == ["flip", "rotate", ("shift", 15, 100)]
    np.testing.assert_array_equal(aug_image, (image + 10) * 2 - 3)
    np.testing.assert_array_equal(aug_mask, mask + 1 + 2 + 3)


def test_augment_image_identity_when_probability_zero(sample_patch):
    image, mask = sample_patch
    aug_image, aug_mask = augment_image(image, mask, p_flip=0, p_rot=0, p_shift=0)
    np.testing.assert_array_equal(aug_image, image)
    np.testing.assert_array_equal(aug_mask, mask)
