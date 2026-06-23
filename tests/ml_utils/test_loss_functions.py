import pytest
import torch
import torch.nn.functional as F

from satellite_trail_segmentation.ml_utils.loss_functions import bce_fn_penalty_loss, combo_loss, dice_loss, weighted_bce_loss


def test_weighted_bce_loss_matches_torch():
    logits = torch.tensor([[2.0, -2.0]])
    target = torch.tensor([1.0, 0.0])
    expected = F.binary_cross_entropy_with_logits(logits, target.view_as(logits), pos_weight=torch.tensor([3.0]))
    assert torch.isclose(weighted_bce_loss(logits, target, pos_weight=3.0), expected)


def test_weighted_bce_loss_rejects_incompatible_shapes():
    logits = torch.zeros(1, 2)
    target = torch.zeros(3)
    with pytest.raises(ValueError, match="incompatible"):
        weighted_bce_loss(logits, target)


def test_dice_loss_extremes():
    assert dice_loss(torch.full((1, 1, 4, 4), 20.0), torch.ones((1, 1, 4, 4))).item() < 1e-5
    assert dice_loss(torch.full((1, 1, 4, 4), -20.0), torch.ones((1, 1, 4, 4))).item() > 0.999

    empty_mask = torch.zeros((1, 1, 4, 4))
    empty_logits = torch.full_like(empty_mask, -20.0)
    epsilon = 10e-5
    expected_empty_loss = 1 - epsilon / (torch.sigmoid(empty_logits).sum() + epsilon)
    assert torch.isclose(dice_loss(empty_logits, empty_mask), expected_empty_loss)
    assert dice_loss(empty_logits, empty_mask).item() < 0.001
    assert dice_loss(torch.full_like(empty_mask, 20.0), empty_mask).item() > 0.99999


def test_combo_loss_combines_terms_and_smoothing():
    logits = torch.tensor([[1.0, -1.0]])
    mask = torch.tensor([[1.0, 0.0]])
    loss = combo_loss(logits, mask, pos_weight=2.0, bce_weight_factor=0.7, label_smoothing=0.1)
    smoothed = mask * 0.9 + 0.05
    expected = 0.7 * weighted_bce_loss(logits, smoothed, pos_weight=2.0) + 0.3 * dice_loss(logits, smoothed)
    assert torch.isclose(loss, expected)


def test_bce_fn_penalty_loss_penalizes_false_negatives():
    pos_logits = torch.tensor([[5.0], [5.0]])
    neg_logits = torch.tensor([[-5.0], [-5.0]])
    targets = torch.tensor([1.0, 1.0])
    assert bce_fn_penalty_loss(neg_logits, targets, fn_penalty_weight=2.0) > bce_fn_penalty_loss(pos_logits, targets, fn_penalty_weight=2.0)


def test_bce_fn_penalty_loss_normalizes_target_shape():
    logits = torch.tensor([[0.0], [0.0]])
    target = torch.tensor([1.0, 0.0])
    loss = bce_fn_penalty_loss(logits, target)
    assert torch.isfinite(loss)
