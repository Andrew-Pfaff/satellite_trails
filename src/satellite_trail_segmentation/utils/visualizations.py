from pathlib import Path

import matplotlib.pyplot as plt
from satellite_trail_segmentation.model.evaluate import image_threshold


def plot_loss_curves(train_loss, val_loss, save_path):
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_full_field(full_image, full_pred, full_mask, save_path, threshold=0.5, title1="Input Image", title2=f"Prediction", title3="Ground Truth Mask"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    axes[0].imshow(full_image, cmap="gray", origin="lower")
    axes[0].set_title(title1)
    
    if threshold is not None:
        full_pred = image_threshold(full_pred, threshold=threshold)
    axes[1].imshow(full_pred, cmap="gray", origin="lower")
    axes[1].set_title(title2)

    axes[2].imshow(full_mask, cmap="gray", origin="lower")
    axes[2].set_title(title3)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()