import torch
from torch.nn.functional import binary_cross_entropy_with_logits


def weighted_bce_loss(logits, target, pos_weight=1.0):
    """
    Calculates a weighted binary cross-entropy loss.

    Args:
        logits (torch.Tensor): Raw model outputs.
        target (torch.Tensor): Ground truth binary labels or masks with the same shape as ``logits``.
        pos_weight (float, optional): Positive class weighting factor. Defaults to 1.0 (same as no weight).

    Returns:
        torch.Tensor: Scalar tensor representing the weighted BCE loss.
    """

    target = target.to(device=logits.device, dtype=logits.dtype)
    if target.shape != logits.shape:
        if target.numel() != logits.numel():
            raise ValueError(f"Target shape {tuple(target.shape)} is incompatible with logits shape {tuple(logits.shape)}")
        target = target.reshape_as(logits)

    weight = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    
    loss =  binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    return loss


def dice_loss(pred, mask):
    """
    Calculates the Soft Dice Loss to evaluate overlap between predictions and ground truth. This function applies a sigmoid activation to the input predictions to convert them into probabilities before computing the Dice coefficient.

    Args:
        pred (torch.Tensor): Raw, unnormalized model predictions (logits).
        mask (torch.Tensor): Ground truth binary mask with the same shape as `pred`. Values should be 0 or 1. 

    Returns:
        loss (torch.Tensor): Scalar tensor representing the Dice loss, bounded between 0 and 1.
    """
    
    epsilon = 10e-5
    prob = torch.sigmoid(pred)
    loss = 1 - (2*torch.sum(prob*mask) + epsilon) / (torch.sum(prob) + torch.sum(mask) + epsilon)
    return loss


def combo_loss(pred, mask, pos_weight=1.0, bce_weight=0.5, dice_weight=0.5, label_smoothing=0.0):
    """
    Computes a weighted combination of Binary Cross-Entropy and Soft Dice Loss.

    Args:
        pred (torch.Tensor): Raw, unnormalized model predictions (logits).
        mask (torch.Tensor): Ground truth binary mask with the same shape as `pred`. 
        bce_weight (float, optional): Scaling factor for the BCE loss component. Defaults to 0.5.
        dice_weight (float, optional): Scaling factor for the Dice loss component. Defaults to 0.5.
        pos_weight (float, optional): Positive class weighting factor for BCE loss. Defaults to 1.0 (same as no weight).


    Returns:
        loss (torch.Tensor): Scalar tensor representing the total combined loss.
    """
    if label_smoothing > 0.0:
        mask = mask * (1 - label_smoothing) + 0.5 * label_smoothing
    
    loss = bce_weight*weighted_bce_loss(pred,mask,pos_weight) + dice_weight*dice_loss(pred,mask)
    return loss


def bce_fn_penalty_loss(logits, target, pos_weight=1.0, fn_penalty_weight=1.0):
    """
    Computes a weighted BCE loss with an additional soft false-negative penalty term.

    Args:
        logits (torch.Tensor): Raw classifier outputs with shape (batch_size, 1).
        target (torch.Tensor): Ground truth binary labels with the same batch shape.
        pos_weight (float, optional): Positive class weighting factor for BCE. Defaults to 1.0.
        fn_penalty_weight (float, optional): Scaling factor for the soft false-negative penalty. Defaults to 1.0.

    Returns:
        torch.Tensor: Scalar tensor representing the total combined loss.
    """

    target = target.to(device=logits.device, dtype=torch.float32).view(-1, 1)
    bce = weighted_bce_loss(logits, target, pos_weight=pos_weight)

    prob = torch.sigmoid(logits)
    soft_fn = torch.sum((1.0 - prob) * target)
    fn_penalty = soft_fn / (target.sum() + 1e-8)

    loss = bce + fn_penalty_weight* fn_penalty
    return loss
