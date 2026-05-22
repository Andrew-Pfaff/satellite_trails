import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.unet_model.unet import UNet


def load_model_weights(save_path):
    """
    Loads a saved UNet checkpoint and returns the model with weights restored.

    Args:
        save_path (str): Path to the saved checkpoint file.

    Returns:
        torch.nn.Module: UNet model with the checkpoint weights loaded.
    """

    ckpt = torch.load(save_path, map_location="cpu")
    model = UNet(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def load_model_weights_classifier(save_path):
    """
    Loads a saved classifier checkpoint and returns the model with weights restored.

    Args:
        save_path (str): Path to the saved checkpoint file.

    Returns:
        torch.nn.Module: TrailClassifier model with the checkpoint weights loaded.
    """

    ckpt = torch.load(save_path, map_location="cpu")
    model = TrailClassifier()
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def load_full_model(save_path, learning_rate, epochs, lr_decay=1e4):
    """
    Loads a saved UNet checkpoint along with its optimizer and scheduler state.

    Recreates the model, optimizer, and cosine annealing scheduler from the saved checkpoint, then returns all restored training state in a dictionary.

    Args:
        save_path (str): Path to the saved checkpoint file.
        learning_rate (float): Learning rate used to recreate the optimizer.
        epochs (int): Total number of training epochs used to recreate the scheduler.
        lr_decay (float, optional): Factor controlling the scheduler minimum learning rate. Defaults to 1e4.

    Returns:
        dict: A dictionary containing the restored model, optimizer, scheduler, and saved training metadata.
    """

    ckpt = torch.load(save_path, map_location="cpu")

    model = UNet(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate / lr_decay,
    )
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": ckpt["epoch"],
        "best_val_loss": ckpt["best_val_loss"],
        "train_loss": ckpt["train_loss"],
        "val_loss": ckpt["val_loss"],
        "model_config": ckpt["model_config"],
    }
