import math

import torch

from satellite_trail_segmentation.classifier_model.classifier_train_function import train_classifier


def test_train_classifier_runs_and_saves(monkeypatch, tiny_classifier_dataset, tiny_classifier, tmp_path):
    optimizer = torch.optim.Adam(tiny_classifier.parameters(), lr=1e-3)
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

    monkeypatch.setattr("satellite_trail_segmentation.classifier_model.classifier_train_function.save_checkpoint", fake_checkpoint)
    monkeypatch.setattr("satellite_trail_segmentation.classifier_model.classifier_train_function.save_weights", fake_weights)

    history = train_classifier(
        tiny_classifier,
        tiny_classifier_dataset,
        tiny_classifier_dataset,
        optimizer,
        scheduler,
        epochs=1,
        batch_size=2,
        pred_thresholds=[0.2, 0.8],
        min_recall=0.0,
        recall_penalty=0.0,
        num_workers=0,
        full_save_path=str(tmp_path / "full.pt"),
        weight_save_path=str(tmp_path / "weights.pt"),
        seed=0,
    )

    assert set(history) == {"train_loss", "val_loss", "best_val_loss", "val_penalized_specificity", "best_penalized_specificity", "final_epoch"}
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert len(history["val_penalized_specificity"]) == 1
    assert history["final_epoch"] == 1
    assert math.isfinite(history["best_val_loss"])
    assert math.isfinite(history["best_penalized_specificity"])
    assert scheduler.last_epoch == 1
    assert len(checkpoint_calls) == 1
    assert len(weight_calls) == 1

    checkpoint = checkpoint_calls[0]
    assert checkpoint["save_path"] == str(tmp_path / "full.pt")
    assert checkpoint["model"] is tiny_classifier
    assert checkpoint["optimizer"] is optimizer
    assert checkpoint["scheduler"] is scheduler
    assert checkpoint["sampler"] is None
    assert checkpoint["epoch"] == 1
    assert set(checkpoint["metrics"]) == {"best_val_specificity", "best_val_loss", "val_recall", "val_specificity"}
    assert all(math.isfinite(float(value)) for value in checkpoint["metrics"].values())
    config = checkpoint["model_config"]
    assert config["in_channels"] == 1
    assert config["kernel_size"] == 3
    assert config["base_channels"] == 4
    assert config["dropout"] == 0.2
    assert config["pos_weight"] == 1.0
    assert config["fn_penalty_weight"] == 1.0
    assert config["pred_thresholds"] == [0.2, 0.8]
    assert config["loss"] == {"name": "bce_fn_penalty_loss", "pos_weight": 1.0, "fn_penalty_weight": 1.0}
    assert config["min_recall"] == 0.0
    assert config["recall_penalty"] == 0.0
    assert config["batch_size"] == 2
    assert config["seed"] == 0
    assert config["normalization"] is None
    assert config["sampler"] is None
    assert config["grad_clip_max_norm"] == 1.0
    assert config["early_stopping"] == {"patience": None, "min_delta": 0.0}
    assert config["augmentation"] == {"p_flip": None, "p_rot": None, "p_shift": None, "min_shift": None, "max_shift": None}

    weights = weight_calls[0]
    assert weights["save_path"] == str(tmp_path / "weights.pt")
    assert weights["model"] is tiny_classifier
    assert weights["model_config"] == checkpoint["model_config"]
