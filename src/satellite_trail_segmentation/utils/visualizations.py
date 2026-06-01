from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from satellite_trail_segmentation.unet_model.unet_evaluate import image_threshold

def _save_plot(save_path=None):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def plot_loss_curves(train_loss, val_loss, save_path):
    """
    Plots training and validation loss curves and saves the figure to disk.

    Generates a simple line plot of the loss history over epochs, writes it to the requested output path, and closes the figure after saving.

    Args:
        train_loss (list of float): Training loss values recorded per epoch.
        val_loss (list of float): Validation loss values recorded per epoch.
        save_path (str): File path for saving the plot image.
    """

    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    _save_plot(save_path)
    plt.close()


def plot_full_field(full_image, full_pred, full_mask, save_path=None, threshold=0.5, title1="Input Image", title2=f"Prediction", title3="Ground Truth Mask"):
    """
    Plots the reconstructed full image, prediction, and ground truth mask side by side.

    Optionally thresholds the prediction map before display, then saves the resulting figure to the requested output path.

    Args:
        full_image (np.ndarray): Reconstructed input image array.
        full_pred (np.ndarray): Reconstructed prediction array.
        full_mask (np.ndarray): Reconstructed ground truth mask array.
        save_path (str): File path for saving the plot image.
        threshold (float, optional): Threshold applied to `full_pred` before plotting. Defaults to 0.5.
        title1 (str, optional): Title for the input image panel. Defaults to "Input Image".
        title2 (str, optional): Title for the prediction panel. Defaults to "Prediction".
        title3 (str, optional): Title for the mask panel. Defaults to "Ground Truth Mask".
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    axes[0].imshow(full_image, cmap="gray")
    axes[0].set_title(title1)
    
    if threshold is not None:
        full_pred = image_threshold(full_pred, threshold=threshold)
    axes[1].imshow(full_pred, cmap="gray")
    axes[1].set_title(title2)

    axes[2].imshow(full_mask, cmap="gray")
    axes[2].set_title(title3)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    _save_plot(save_path)
    plt.close()


def plot_roc_curve(fpr, tpr, thresholds, roc_auc, optimal_threshold, save_path):
    """
    Plots an ROC curve and highlights the optimal operating threshold.

    Marks the threshold that is closest to the supplied optimal threshold, then saves the figure to the requested output path.

    Args:
        fpr (np.ndarray): False positive rates for the ROC curve.
        tpr (np.ndarray): True positive rates for the ROC curve.
        thresholds (np.ndarray): Threshold values associated with the ROC points.
        roc_auc (float): Area under the ROC curve.
        optimal_threshold (float): Threshold chosen as optimal for classification.
        save_path (str): File path for saving the plot image.
    """

    # Find the index of the optimal threshold in the thresholds array
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        label=f"Optimal threshold = {optimal_threshold:.3f}",
        zorder=5
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    _save_plot(save_path)
    plt.close()


def plot_pred_residual(full_pred, full_mask, save_path):
    """
    Plots the prediction, ground truth mask, and residual map for a full-field reconstruction.

    Displays the prediction probabilities, the corresponding binary mask, and their difference as a residual heatmap before saving the figure to disk.

    Args:
        full_pred (np.ndarray): Reconstructed prediction probability map.
        full_mask (np.ndarray): Reconstructed ground truth mask.
        save_path (str): File path for saving the plot image.
    """

    residual = full_pred - full_mask.astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(full_pred, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[0].set_title("Prediction (probability)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(full_mask, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(residual, cmap="RdBu_r", origin="lower", vmin=-1, vmax=1)
    axes[2].set_title("Residual (pred − mask)")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    _save_plot(save_path)
    plt.close()