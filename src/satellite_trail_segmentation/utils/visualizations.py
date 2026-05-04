from pathlib import Path

import matplotlib.pyplot as plt


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

def plot_full_field(full_image, full_pred, full_mask, source_index, save_path, threshold=0.8):
    fig, axes = plt.subplots(1, 3, figsize=(18, 12))

    axes[0].imshow(full_image, cmap="gray", origin="lower")
    axes[0].set_title(f"Input Image (source {source_index})")

    axes[1].imshow(full_pred > threshold, cmap="gray", origin="lower")
    axes[1].set_title(f"Prediction (threshold={threshold})")

    axes[2].imshow(full_mask, cmap="gray", origin="lower")
    axes[2].set_title("Ground Truth Mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()