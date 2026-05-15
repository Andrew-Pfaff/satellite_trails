import numpy as np
from torch.utils.data import Sampler


class BalancedTrailSampler(Sampler):
    def __init__(self, pos_indices, neg_indices, pos_fraction=0.5):
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.pos_fraction = pos_fraction
        
        # Calculate N negatives needed to satisfy the ratio
        self.n_pos = len(pos_indices)
        self.n_neg = int(self.n_pos * (1 - pos_fraction) / pos_fraction)

    def __iter__(self):
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
        return self.n_pos + self.n_neg