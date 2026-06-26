import numpy as np

def flip(image, mask):
    """
    Randomly flips an image and mask pair along a single uniform-randomly selected axis.

    Args:
        image (np.ndarray): 2D image array
        mask (np.ndarray): 2D mask array matching image shape

    Returns:
        aug_image (np.ndarray): Flipped image
        aug_mask (np.ndarray): Flipped mask
    """

    flip_dir = np.random.randint(0,2)
    aug_image = np.flip(image, axis=flip_dir)
    aug_mask = np.flip(mask, axis=flip_dir)
    return aug_image, aug_mask


def rotate(image, mask):
    """
    Uniform-randomly rotates an image and mask pair by either 90, 180, or 270 degrees.

    Args:
        image (np.ndarray): 2D image array
        mask (np.ndarray): 2D mask array

    Returns:
        aug_image (np.ndarray): Rotated image
        aug_mask (np.ndarray): Rotated mask
    """
        
    rot_dir = np.random.randint(1,4)
    aug_image = np.rot90(image, rot_dir)
    aug_mask = np.rot90(mask, rot_dir)
    return aug_image, aug_mask


def shift(image, mask, min_shift=15, max_shift=100):
    """
    Randomly shifts an image and mask pair in one of four cardinal directions. Shift distance is selected randomly between min_shift and max_shift pixels.
    Vacated pixels in the image are filled with normal noise centered at the median with the image's standard deviation. Mask pixels are filled with zero.
    
    Args:
        image (np.ndarray): 2D image array
        mask (np.ndarray): 2D mask array
        min_shift (int): Minimum shift in pixels. Defaults to 15.
        max_shift (int): Maximum shift in pixels inclusive. Defaults to 100.
    Returns:
        aug_image (np.ndarray): Shifted image
        aug_mask (np.ndarray): Shifted mask
    """

    shift_total = np.random.randint(min_shift, max_shift + 1)

    valid_dirs = []
    if not np.any(mask[:, 0]):
        valid_dirs.append(0)  
    if not np.any(mask[0, :]):
        valid_dirs.append(1)  
    if not np.any(mask[:, -1]):
        valid_dirs.append(2)  
    if not np.any(mask[-1, :]):
        valid_dirs.append(3)
    if not valid_dirs:
        return image, mask
    
    
    shift_total = np.random.randint(min_shift, max_shift + 1)
    shift_dir = np.random.choice(valid_dirs)

    aug_image = np.random.normal(loc=np.median(image), scale=np.std(image), size=image.shape).astype(image.dtype)
    aug_mask = np.zeros_like(mask)
    

    if shift_dir == 0:
        aug_image[:,shift_total:] = image[:,:-shift_total]
        aug_mask[:,shift_total:] = mask[:,:-shift_total]
    elif shift_dir == 1:
        aug_image[shift_total:,:] = image[:-shift_total,:]
        aug_mask[shift_total:,:] = mask[:-shift_total,:]
    elif shift_dir == 2:
        aug_image[:,:-shift_total] = image[:,shift_total:]
        aug_mask[:,:-shift_total] = mask[:,shift_total:]
    else:
        aug_image[:-shift_total,:] = image[shift_total:,:]
        aug_mask[:-shift_total,:] = mask[shift_total:,:]

    return aug_image, aug_mask


def augment_image(image, mask, p_flip=0, p_rot=0, p_shift=0, min_shift=15, max_shift=100):
    """
    Applies random augmentations to an image and mask pair.

    Each augmentation is applied independently with its given probability.
    Augmentations are applied in order: flip, rotate, shift.

    Args:
        image (np.ndarray): 2D image array
        mask (np.ndarray): 2D mask array
        p_flip (float): Probability of applying a random flip. Defaults to 0.
        p_rot (float): Probability of applying a random rotation. Defaults to 0.
        p_shift (float): Probability of applying a random shift. Defaults to 0.
        min_shift (int): Minimum shift in pixels. Defaults to 15.
        max_shift (int): Maximum shift in pixels inclusive. Defaults to 100.

    Returns:
        image (np.ndarray): Augmented image
        mask (np.ndarray): Augmented mask
    """

    flip_img = (np.random.rand() < p_flip)
    rot_img = (np.random.rand() < p_rot)
    shift_img = (np.random.rand() < p_shift)

    if flip_img:
        image, mask = flip(image, mask)

    if rot_img:
        image, mask = rotate(image, mask)

    if shift_img:
        image, mask = shift(image, mask, min_shift=min_shift, max_shift=max_shift)

    return image, mask
