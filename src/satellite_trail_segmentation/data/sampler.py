import numpy as np
from torch.utils.data import Sampler


class BalancedTrailSampler(Sampler):
    """
    A custom sampler that balances positive (trail-containing) and negative (trail-free) patches in each epoch.

    All positive samples are used every epoch, and negatives are sampled to satisfy the desired class ratio. This ensures the model sees every trail example while controlling class imbalance.

    Attributes:
        pos_indices (list): Indices of positive (trail-containing) samples in the dataset.
        neg_indices (list): Indices of negative (trail-free) samples in the dataset.
        pos_fraction (float): Target fraction of positive samples per epoch.
        n_pos (int): Total number of positive samples.
        n_neg (int): Number of negative samples drawn per epoch to satisfy pos_fraction.
    """

    def __init__(self, pos_indices, neg_indices, pos_fraction=0.5):
        """
        Initialises the sampler and computes the number of negatives needed per epoch.

        Args:
            pos_indices (array-like): Indices of positive samples in the dataset.
            neg_indices (array-like): Indices of negative samples in the dataset.
            pos_fraction (float, optional): Desired fraction of positive samples in each epoch. Defaults to 0.5. For example, 0.3 means 30% of each epoch will be positive patches.
        """

        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.pos_fraction = pos_fraction
        
        # Calculate N negatives needed to satisfy the ratio
        self.n_pos = len(pos_indices)
        self.n_neg = int(self.n_pos * (1 - pos_fraction) / pos_fraction)

    def __iter__(self):
        """
        Generates a shuffled sequence of indices for one epoch.

        All positive indices are included and shuffled. Negatives are sampled to match the target ratio, with replacement if the number of required negatives exceeds the available supply. The combined set is then shuffled before iteration.

        Yields:
            int: Dataset index for each sample in the epoch.
        """

        # 1. Take all positives
        pos_samples = self.pos_indices.copy()
        np.random.shuffle(pos_samples)
        
        # 2. Sample negatives (with replacement if negatives are scarce)
        replace = self.n_neg > len(self.neg_indices)
        neg_samples = np.random.choice(self.neg_indices, self.n_neg, replace=replace)
        
        # 3. Combine and shuffle for the epoch
        epoch_indices = np.concatenate([pos_samples, neg_samples])
        np.random.shuffle(epoch_indices)
        return iter(epoch_indices.tolist())

    def __len__(self):
        """
        Returns the total number of samples per epoch.

        Returns:
            int: Sum of all positive samples and the computed number of negative samples.
        """

        return self.n_pos + self.n_neg