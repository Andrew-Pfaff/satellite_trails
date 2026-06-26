import math

import torch

from satellite_trail_segmentation.unet_model.attention_unet import AttentionUNet
from satellite_trail_segmentation.unet_model.unet_train_function import train_unet


def test_train_unet_runs_and_saves(monkeypatch, tiny_seg_dataset, tiny_unet_model, tmp_path):
    optimizer = torch.optim.Adam(tiny_unet_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    checkpoint_calls = []
    weight_calls = []

    def fake_checkpoint(save_path, model, optimizer, scheduler, sampler, epoch, metrics, model_config):
        checkpoint_calls.append(
            {
                "save_path": save_path,
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "sampler": sampler,
                "epoch": epoch,
                "metrics": metrics,
                "model_config": model_config,
            }
        )

    def fake_weights(save_path, model, model_config):
        weight_calls.append({"save_path": save_path, "model": model, "model_config": model_config})

    monkeypatch.setattr("satellite_trail_segmentation.unet_model.unet_train_function.save_checkpoint", fake_checkpoint)
    monkeypatch.setattr("satellite_trail_segmentation.unet_model.unet_train_function.save_weights", fake_weights)

    history = train_unet(
        tiny_unet_model,
        tiny_seg_dataset,
        tiny_seg_dataset,
        optimizer,
        scheduler,
        epochs=1,
        batch_size=2,
        iou_thresholds=[0.3, 0.7],
        num_workers=0,
        full_save_path=str(tmp_path / "full.pt"),
        weight_save_path=str(tmp_path / "weights.pt"),
        seed=0,
    )

    assert set(history) == {"train_loss", "val_loss", "val_loss_at_best_iou", "val_iou", "best_iou", "best_threshold", "final_epoch"}
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert len(history["val_iou"]) == 1
    assert history["final_epoch"] == 1
    assert history["best_threshold"] in {0.3, 0.7}
    assert math.isfinite(history["best_iou"])
    assert math.isfinite(history["val_loss_at_best_iou"])
    assert scheduler.last_epoch == 1
    assert len(checkpoint_calls) == 1
    assert len(weight_calls) == 1

    checkpoint = checkpoint_calls[0]
    assert checkpoint["save_path"] == str(tmp_path / "full.pt")
    assert checkpoint["model"] is tiny_unet_model
    assert checkpoint["optimizer"] is optimizer
    assert checkpoint["scheduler"] is scheduler
    assert checkpoint["sampler"] is None
    assert checkpoint["epoch"] == 1
    assert set(checkpoint["metrics"]) == {"best_iou", "best_threshold", "val_loss_at_best_iou"}
    assert checkpoint["metrics"]["best_threshold"] in {0.3, 0.7}
    assert all(math.isfinite(float(value)) for value in checkpoint["metrics"].values())
    config = checkpoint["model_config"]
    assert config["model_name"] == "UNet"
    assert config["model_class"] == "UNet"
    assert config["in_channels"] == 1
    assert config["out_channels"] == 1
    assert config["kernel_size"] == 3
    assert config["base_channels"] == 4
    assert config["dropout"] == 0.0
    assert config["use_batchnorm"] is True
    assert config["normalization"] is None
    assert config["pos_weight"] == 1.0
    assert config["bce_weight_factor"] == 0.5
    assert config["label_smoothing"] == 0.0
    assert config["loss"] == {"name": "combo_loss", "pos_weight": 1.0, "bce_weight_factor": 0.5, "label_smoothing": 0.0}
    assert config["batch_size"] == 2
    assert config["seed"] == 0
    assert config["sampler"] is None
    assert config["iou_thresholds"] == [0.3, 0.7]
    assert config["grad_clip_max_norm"] == 1.0
    assert config["early_stopping"] == {"patience": None, "min_delta": 0.0}
    assert config["augmentation"] == {"p_flip": None, "p_rot": None, "p_shift": None, "min_shift": None, "max_shift": None}
    assert "bce_loss_factor" not in config
    assert "dice_loss_factor" not in config

    weights = weight_calls[0]
    assert weights["save_path"] == str(tmp_path / "weights.pt")
    assert weights["model"] is tiny_unet_model
    assert weights["model_config"] == checkpoint["model_config"]


def test_train_unet_runs_and_saves_attention_unet_metadata(monkeypatch, tiny_seg_dataset, tmp_path):
    model = AttentionUNet(in_channels=1, out_channels=1, kernel_size=3, base_channels=4, dropout=0.0, use_batchnorm=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    checkpoint_calls = []

    def fake_checkpoint(save_path, model, optimizer, scheduler, sampler, epoch, metrics, model_config):
        checkpoint_calls.append(
            {
                "save_path": save_path,
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "sampler": sampler,
                "epoch": epoch,
                "metrics": metrics,
                "model_config": model_config,
            }
        )

    monkeypatch.setattr("satellite_trail_segmentation.unet_model.unet_train_function.save_checkpoint", fake_checkpoint)

    history = train_unet(
        model,
        tiny_seg_dataset,
        tiny_seg_dataset,
        optimizer,
        scheduler,
        epochs=1,
        batch_size=2,
        iou_thresholds=[0.5],
        num_workers=0,
        full_save_path=str(tmp_path / "attention_full.pt"),
        seed=0,
    )

    assert history["final_epoch"] == 1
    assert len(history["train_loss"]) == 1
    assert len(checkpoint_calls) == 1

    config = checkpoint_calls[0]["model_config"]
    assert config["model_name"] == "attention_unet"
    assert config["model_class"] == "AttentionUNet"
    assert config["base_channels"] == 4
    assert config["use_batchnorm"] is True
