def bce_loss(image, mask):
    return loss

def dice_loss(image, mask):
    return loss

def combo_loss(image, mask, bce_weight=0.5, dice_weight=0.5):
    loss = bce_weight*bce_loss(image,mask) + dice_weight*dice_loss(image,mask)
    return loss