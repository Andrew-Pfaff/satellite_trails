import torch

_bce = torch.nn.BCEWithLogitsLoss() #Loss function that applies sigmoid and calculates BCE

def bce_loss(pred, mask):
    """
    Calculates the Binary Cross-Entropy (BCE) loss between predictions and target masks.

    Args:
        pred (torch.Tensor): Raw, unnormalized model predictions (logits) or probability maps depending on the underlying `_bce` implementation. 
        mask (torch.Tensor): Ground truth binary mask with the same shape as `pred`. Values should be 0 or 1.

    Returns:
        loss (torch.Tensor): Scalar tensor representing the computed BCE loss.
    """

    loss = _bce(pred,mask)
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


def combo_loss(pred, mask, bce_weight=0.5, dice_weight=0.5):
    """
    Computes a weighted combination of Binary Cross-Entropy and Soft Dice Loss.

    Args:
        pred (torch.Tensor): Raw, unnormalized model predictions (logits).
        mask (torch.Tensor): Ground truth binary mask with the same shape as `pred`. 
        bce_weight (float, optional): Scaling factor for the BCE loss component. Defaults to 0.5.
        dice_weight (float, optional): Scaling factor for the Dice loss component. Defaults to 0.5.

    Returns:
        loss (torch.Tensor): Scalar tensor representing the total combined loss.
    """

    loss = bce_weight*bce_loss(pred,mask) + dice_weight*dice_loss(pred,mask)
    return loss