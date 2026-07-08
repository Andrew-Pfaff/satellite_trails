from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def create_cos_lr_sched(optimizer, epochs, warmup_epochs=5, eta_min=1e-6):
    """
    Creates a sequential learning rate scheduler with a linear warmup phase followed by a cosine annealing decay phase.

    Warmup phase, the learning rate scales linearly from 10% of its initial value up to 100%.
    Then cosine wave decay that smoothly pulls the learning rate down to the minimum specified value (`eta_min`) over the remaining epochs.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the learning rate schedule will be applied.
        epochs (int): Total number of training epochs planned for the run.
        warmup_epochs (int, optional): The number of initial epochs dedicated to the linear warmup phase. If set to 0 or None, the warmup is skipped, and a standard CosineAnnealingLR is returned. Defaults to 5.
        eta_min (float, optional): The minimum bounding value that the learning rate is allowed to decay down to during the cosine annealing phase. Defaults to 1e-6.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: A unified PyTorch learning rate scheduler object (either a `SequentialLR` or a standalone `CosineAnnealingLR`).
            
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = create_cos_lr_sched(optimizer, epochs=50, warmup_epochs=5)
        >>> # Epochs 0-4: LR climbs from 1e-4 to 1e-3
        >>> # Epochs 5-49: LR decays along a cosine curve from 1e-3 down to 1e-6
    """


    if warmup_epochs is not None and warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cos = CosineAnnealingLR(optimizer, T_max=(epochs-warmup_epochs), eta_min=eta_min)
        return SequentialLR(optimizer, schedulers=[warmup, cos], milestones=[warmup_epochs])
    else:
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
