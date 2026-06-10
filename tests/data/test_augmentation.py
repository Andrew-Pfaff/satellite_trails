import numpy as np

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

    def expected_shift(direction):
        fill = np.full_like(image, image.mean())
        expected_image = np.zeros_like(image)
        expected_mask = np.zeros_like(mask)
        if direction == 0:
            expected_image[:, 1:] = image[:, :-1]
            expected_image[:, 0] = fill[:, 0]
            expected_mask[:, 1:] = mask[:, :-1]
        elif direction == 1:
            expected_image[1:, :] = image[:-1, :]
            expected_image[0, :] = fill[0, :]
            expected_mask[1:, :] = mask[:-1, :]
        elif direction == 2:
            expected_image[:, :-1] = image[:, 1:]
            expected_image[:, -1] = fill[:, -1]
            expected_mask[:, :-1] = mask[:, 1:]
        else:
            expected_image[:-1, :] = image[1:, :]
            expected_image[-1, :] = fill[-1, :]
            expected_mask[:-1, :] = mask[1:, :]
        return expected_image.astype(np.uint8), expected_mask

    for direction in range(4):
        values = iter([1, direction])
        monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: next(values))
        aug_image, aug_mask = shift(image, mask, min_shift=1, max_shift=1)
        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape
        expected_image, expected_mask = expected_shift(direction)
        np.testing.assert_array_equal(aug_image, expected_image)
        np.testing.assert_array_equal(aug_mask, expected_mask)
        assert aug_mask.sum() <= mask.sum()


def test_augment_image_composes_in_order(monkeypatch):
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    mask = (image > 3).astype(np.uint8)

    rand_values = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr(np.random, "rand", lambda *args, **kwargs: next(rand_values))
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0 if args[:2] == (0, 2) else 1)

    aug_image, aug_mask = augment_image(image, mask, p_flip=1, p_rot=1, p_shift=1)
    flip_image, flip_mask = flip(image, mask)
    rot_image, rot_mask = rotate(flip_image, flip_mask)
    expected_image, expected_mask = shift(rot_image, rot_mask, min_shift=1, max_shift=1)
    np.testing.assert_array_equal(aug_image, expected_image)
    np.testing.assert_array_equal(aug_mask, expected_mask)


def test_augment_image_identity_when_probability_zero(sample_patch):
    image, mask = sample_patch
    aug_image, aug_mask = augment_image(image, mask, p_flip=0, p_rot=0, p_shift=0)
    np.testing.assert_array_equal(aug_image, image)
    np.testing.assert_array_equal(aug_mask, mask)
