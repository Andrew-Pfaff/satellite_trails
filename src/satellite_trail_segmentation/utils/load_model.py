import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from satellite_trail_segmentation.model.unet import UNet


def load_model_weights(save_path):
    ckpt = torch.load(save_path, map_location="cpu")
    model = UNet(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def load_full_model(save_path, learning_rate, epochs, lr_decay=1e4):
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
