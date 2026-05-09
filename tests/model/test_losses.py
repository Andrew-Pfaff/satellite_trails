import pytest
import torch
from satellite_trail_segmentation.model.losses import bce_loss, dice_loss, combo_loss

@pytest.fixture
def prediction_and_target():
    pred = torch.tensor([[10.0, -10.0], [10.0, -10.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    return pred, mask

def test_bce_loss(prediction_and_target):
    pred, mask = prediction_and_target
    loss = bce_loss(pred, mask)
    assert loss.item() < 1e-4

def test_dice_loss(prediction_and_target):
    pred, mask = prediction_and_target
    loss = dice_loss(pred, mask)
    assert loss.item() < 1e-4

def test_combo_loss(prediction_and_target):
    pred, mask = prediction_and_target
    loss = combo_loss(pred, mask, bce_weight=0.7, dice_weight=0.3)
    assert loss.item() < 1e-4

    bad_mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    high_loss = combo_loss(pred, bad_mask)
    assert high_loss.item() > 1.0