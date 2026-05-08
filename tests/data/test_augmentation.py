import numpy as np

from satellite_trail_segmentation.data.augmentation import flip, rotate, shift, augment_image


#Flip:

def test_flip_output_shape(sample_patch):
    image, mask = sample_patch
    aug_image, aug_mask = flip(image, mask)
    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape


def test_same_flip():
    n = 10 #sample size to reduce false pass /// n=1 -> FP=0.5 | n=N -> FP=(0.5)**N
    for _ in range(n): 
        image = np.random.randint(0, 256, (528, 528), dtype=np.uint8)
        mask = np.random.randint(0, 2, (528, 528), dtype=np.uint8)
        
        aug_image, aug_mask = flip(image, mask)

        img_flip_ax0 = np.flip(image, axis=0)
        img_flip_ax1 = np.flip(image, axis=1)

        if np.array_equal(aug_image, img_flip_ax0):
            flip_ax = 0
        elif np.array_equal(aug_image, img_flip_ax1):
            flip_ax = 1
        else:
            assert False

        mask_same_flip = np.flip(mask, axis=flip_ax)

        assert np.array_equal(mask_same_flip, aug_mask)


#Rotate:

def test_rotate_output_shape(sample_patch):
    image, mask = sample_patch
    aug_image, aug_mask = rotate(image, mask)
    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape


def test_same_rotate():
    n = 5  # sample size to reduce false pass /// n=1 -> FP=0.25 | n=N -> FP=(0.25)**N
    for _ in range(n):
        image = np.random.randint(0, 256, (528, 528), dtype=np.uint8)
        mask = np.random.randint(0, 2, (528, 528), dtype=np.uint8)

        aug_image, aug_mask = rotate(image, mask)

        rot_match = None
        for k in [1, 2, 3]:
            if np.array_equal(aug_image, np.rot90(image, k)):
                rot_match = k
                break
        if rot_match is None:
            assert False

        mask_same_rot = np.rot90(mask, rot_match)

        assert np.array_equal(mask_same_rot, aug_mask)


# Shift:

def test_shift_output_shape(sample_patch):
    image, mask = sample_patch
    aug_image, aug_mask = shift(image, mask)
    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape

def test_shift_mask_no_new_trail_pixels():
    """Shifting can only move trail pixels out of frame, never creates new ones."""
    n = 10
    for _ in range(n):
        image = np.random.randint(0, 256, (528, 528), dtype=np.uint8)
        mask = np.zeros((528, 528), dtype=np.uint8)
        mask[100, :] = 1  # horizontal trail
        aug_image, aug_mask = shift(image, mask)
        assert aug_mask.sum() <= mask.sum()

def test_shift_mask_no_new_trail_pixels_vertical():
    """Vertical trail catches missing mask fill for directions 2 and 3."""
    n = 10
    for _ in range(n):
        image = np.random.randint(0, 256, (528, 528), dtype=np.uint8)
        mask = np.zeros((528, 528), dtype=np.uint8)
        mask[:, 100] = 1  # vertical trail
        aug_image, aug_mask = shift(image, mask)
        assert aug_mask.sum() <= mask.sum()


def test_shift_image_background_fill():
    n = 10
    for _ in range(n):
        image = np.random.randint(0, 256, (528, 528), dtype=np.uint8)
        mask = np.zeros((528, 528), dtype=np.uint8)
        aug_image, aug_mask = shift(image, mask, min_shift=10, max_shift=10)
        expected_fill = np.mean(image).astype(np.uint8)
        assert np.all(aug_image[:, :10] == expected_fill) or \
               np.all(aug_image[:10, :] == expected_fill) or \
               np.all(aug_image[:, -10:] == expected_fill) or \
               np.all(aug_image[-10:, :] == expected_fill)
        

def test_shift_same_direction_image_and_mask():
    n = 10
    for _ in range(n):
        image = np.random.randint(0, 256, (528, 528), dtype=np.uint8)
        mask = np.zeros((528, 528), dtype=np.uint8)
        mask[100, :] = 1  # horizontal trail
        mask[:, 100] = 1  # vertical trail

        aug_image, aug_mask = shift(image, mask, min_shift=10, max_shift=10)
        expected_fill = np.mean(image).astype(np.uint8)

        if np.all(aug_image[:, :10] == expected_fill):
            assert aug_mask[:, :10].sum() == 0
        elif np.all(aug_image[:10, :] == expected_fill):
            assert aug_mask[:10, :].sum() == 0
        elif np.all(aug_image[:, -10:] == expected_fill):
            assert aug_mask[:, -10:].sum() == 0
        elif np.all(aug_image[-10:, :] == expected_fill):
            assert aug_mask[-10:, :].sum() == 0
        else:
            assert False
        

# Augment:

def test_augment_output_shape(sample_patch):
    image, mask = sample_patch
    aug_image, aug_mask = augment_image(image, mask, p_flip=1, p_rot=1, p_shift=1)
    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape


def test_augment_no_ops(sample_patch):
    image, mask = sample_patch
    aug_image, aug_mask = augment_image(image, mask, p_flip=0, p_rot=0, p_shift=0)
    assert np.array_equal(aug_image, image)
    assert np.array_equal(aug_mask, mask)