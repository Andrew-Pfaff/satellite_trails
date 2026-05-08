import numpy as np

def flip(image, mask):
    flip_dir = np.random.randint(0,2)
    aug_image = np.flip(image, axis=flip_dir)
    aug_mask = np.flip(mask, axis=flip_dir)
    return aug_image, aug_mask


def rotate(image, mask):
    rot_dir = np.random.randint(1,4)
    aug_image = np.rot90(image, rot_dir)
    aug_mask = np.rot90(mask, rot_dir)
    return aug_image, aug_mask


def shift(image, mask, min_shift=4, max_shift=20):
    shift_total = np.random.randint(min_shift,max_shift+1)
    shift_dir = np.random.randint(0,4)

    aug_image = np.zeros_like(image)
    aug_mask = np.zeros_like(mask)
    fill_value = np.mean(image).astype(np.uint8)

    if shift_dir == 0:
        aug_image[:,shift_total:] = image[:,:-shift_total]
        aug_image[:,:shift_total] = fill_value
        aug_mask[:,shift_total:] = mask[:,:-shift_total]
    elif shift_dir == 1:
        aug_image[shift_total:,:] = image[:-shift_total,:]
        aug_image[:shift_total,:] = fill_value
        aug_mask[shift_total:,:] = mask[:-shift_total,:]
    elif shift_dir == 2:
        aug_image[:,:-shift_total] = image[:,shift_total:]
        aug_image[:,-shift_total:] = fill_value
        aug_mask[:,:-shift_total] = mask[:,shift_total:]
    else:
        aug_image[:-shift_total,:] = image[shift_total:,:]
        aug_image[-shift_total:,:] = fill_value
        aug_mask[:-shift_total,:] = mask[shift_total:,:]

    return aug_image, aug_mask


def augment_image(image, mask, p_flip=0, p_rot=0, p_shift=0):
    flip_img = (np.random.rand() < p_flip)
    rot_img = (np.random.rand() < p_rot)
    shift_img = (np.random.rand() < p_shift)

    if flip_img:
        image, mask = flip(image, mask)

    if rot_img:
        image, mask = rotate(image, mask)

    if shift_img:
        image, mask = shift(image, mask)

    return image, mask