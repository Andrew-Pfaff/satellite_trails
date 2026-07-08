import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Seeds Python, NumPy, and PyTorch random number generators.

    Args:
        seed (int): Seed value to apply.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Seeds a DataLoader worker from PyTorch's worker seed.

    Args:
        worker_id (int): DataLoader worker id supplied by PyTorch.
    """

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int):
    """
    Creates a manually seeded PyTorch generator.

    Args:
        seed (int): Seed value for the generator.

    Returns:
        torch.Generator: Seeded generator for DataLoader shuffling or sampling.
    """

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
