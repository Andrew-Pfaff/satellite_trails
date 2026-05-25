import torch


def save_checkpoint(save_path, model, optimizer, scheduler, epoch, metrics, model_config):
    """Full resumable checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,         
        "model_config": model_config,
    }, save_path)


def save_weights(save_path, model, model_config):
    """Lightweight weights-only file."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
    }, save_path)


def load_checkpoint(load_path, model, optimizer=None, scheduler=None):
    """
    Load a checkpoint. If optimizer/scheduler are None, loads weights only.
    Returns the checkpoint dict so the caller can retrieve metrics/epoch.
    """
    checkpoint = torch.load(load_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint