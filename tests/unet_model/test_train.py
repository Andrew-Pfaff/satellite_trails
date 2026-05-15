import pytest
import torch
import sys
from unittest.mock import patch, MagicMock
from torch.optim.lr_scheduler import CosineAnnealingLR

from satellite_trail_segmentation.unet_model.train import train_unet, parse_args


def test_parse_args():
    """Verify that command-line arguments are parsed correctly."""
    test_args = [
        "train.py",
        "--data-path", "/fake/data.h5",
        "--epochs", "10",
        "--batch-size", "32",
        "--learning-rate", "0.001"
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.data_path == "/fake/data.h5"
        assert args.epochs == 10
        assert args.batch_size == 32
        assert args.learning_rate == 0.001
        assert args.dropout_rate == 0.0  


@patch("satellite_trail_segmentation.model.train.DataLoader")
@patch("satellite_trail_segmentation.model.train.Path")
@patch("satellite_trail_segmentation.model.train.torch.save")
def test_train_unet(mock_torch_save, mock_path, MockDataLoader, dummy_unet, dummy_dataset_batch):
    """Mock the datasets and ensure train_unet runs fully for 1 epoch."""
    
    optimizer = torch.optim.Adam(dummy_unet.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-4)
    MockDataLoader.return_value = dummy_dataset_batch
    
    train_loss, val_loss, best_loss, final_epoch = train_unet(
        model=dummy_unet,
        train_ds=MagicMock(), 
        val_ds=MagicMock(),
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=1,
        batch_size=2,
        save_path="fake/path.pt"
    )

    assert len(train_loss) == 1
    assert len(val_loss) == 1
    assert final_epoch == 1
    assert isinstance(best_loss, float)
    
    assert scheduler.last_epoch == 1
    mock_torch_save.assert_called_once()