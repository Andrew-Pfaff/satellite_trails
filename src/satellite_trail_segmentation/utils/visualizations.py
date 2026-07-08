from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from satellite_trail_segmentation.evaluation.unet_evaluate import image_threshold

def _save_plot(save_path=None):
    """
    Helper function to save the current Matplotlib figure or displays it interactively if save_path is None.

    Args:
        save_path (str or Path, optional): Destination path. If None, displays the figure.
    """

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
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


def _binary_mask(array, threshold=None):
    """
    Converts an array to a boolean mask.

    Args:
        array (np.ndarray): Input prediction or mask array.
        threshold (float, optional): Threshold for probability-like predictions.

    Returns:
        np.ndarray: Boolean mask.
    """

    array = np.asarray(array)
    if threshold is not None:
        return array > threshold
    return array > 0


def _normalize_grayscale(array):
    """
    Normalizes an array to the [0, 1] grayscale range.

    Args:
        array (np.ndarray): Input image-like array.

    Returns:
        np.ndarray: Float32 normalized array.
    """

    array = np.asarray(array, dtype=np.float32)
    min_value = np.nanmin(array)
    max_value = np.nanmax(array)
    if np.isclose(max_value, min_value):
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def _error_color_image(prediction, mask, threshold=None):
    """
    Builds an RGB error overlay for false negatives and false positives.

    Args:
        prediction (np.ndarray): Prediction array or mask.
        mask (np.ndarray): Ground-truth mask array.
        threshold (float, optional): Threshold for probability-like predictions.

    Returns:
        np.ndarray: RGB image with false negatives in red and false positives in green.
    """

    pred_bin = _binary_mask(prediction, threshold=threshold)
    mask_bin = _binary_mask(mask)

    base = _normalize_grayscale(prediction)
    rgb = np.repeat(base[..., np.newaxis], 3, axis=2)

    false_negatives = mask_bin & ~pred_bin
    false_positives = pred_bin & ~mask_bin

    rgb[false_negatives] = [1.0, 0.0, 0.0]
    rgb[false_positives] = [0.0, 1.0, 0.0]
    return rgb


def plot_segmentation_postprocess_comparison(
    image,
    mask,
    prediction,
    postprocessed,
    save_path=None,
    threshold=None,
    title="Segmentation Postprocessing Comparison",
):
    """
    Plots an image, ground-truth mask, raw prediction, and postprocessed mask.

    Args:
        image (np.ndarray): Input image array.
        mask (np.ndarray or None): Ground-truth mask array. If None, the mask panel is omitted.
        prediction (np.ndarray): Raw segmentation prediction or mask.
        postprocessed (np.ndarray): Postprocessed segmentation mask.
        save_path (str or Path, optional): File path for saving the plot image.
        threshold (float, optional): Threshold applied to prediction before plotting.
        title (str, optional): Figure title.
    """

    panels = [("Image", image, "gray", None)]
    if mask is not None:
        panels.append(("Mask", mask, "gray", None))

    if threshold is not None:
        prediction = image_threshold(prediction, threshold=threshold)

    if mask is not None:
        prediction_panel = _error_color_image(prediction, mask)
        postprocessed_panel = _error_color_image(postprocessed, mask)
        prediction_cmap = None
        postprocessed_cmap = None
        prediction_subtitle = "FN red, FP green"
        postprocessed_subtitle = "FN red, FP green"
    else:
        prediction_panel = prediction
        postprocessed_panel = postprocessed
        prediction_cmap = "gray"
        postprocessed_cmap = "gray"
        prediction_subtitle = None
        postprocessed_subtitle = None

    panels.extend(
        [
            ("Prediction", prediction_panel, prediction_cmap, prediction_subtitle),
            ("Postprocessed", postprocessed_panel, postprocessed_cmap, postprocessed_subtitle),
        ]
    )

    if len(panels) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (panel_title, array, cmap, subtitle) in zip(axes, panels):
        if cmap is None:
            ax.imshow(array)
        else:
            ax.imshow(array, cmap=cmap)
        if subtitle is not None:
            panel_title = f"{panel_title}\n{subtitle}"
        ax.set_title(panel_title, pad=10)
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, y=1.04)

    if mask is not None:
        legend_handles = [
            Patch(facecolor="red", edgecolor="red", label="False negatives"),
            Patch(facecolor="green", edgecolor="green", label="False positives"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)

    _save_plot(save_path)
    plt.close(fig)


def plot_prediction_mask(image, prediction, mask=None, save_path=None):
    """
    Plots an input image, prediction mask, and optional ground-truth mask.

    Args:
        image (np.ndarray): Input image array.
        prediction (np.ndarray): Predicted binary mask.
        mask (np.ndarray, optional): Ground-truth mask array. Defaults to None.
        save_path (str or Path, optional): File path for saving the plot image.
    """

    panels = [("Image", image), ("Prediction", prediction)]
    if mask is not None:
        panels.append(("True Mask", mask))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, array) in zip(axes, panels):
        ax.imshow(array, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    _save_plot(save_path)
    plt.close(fig)


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


def plot_threshold_metrics(threshold_metrics, save_path=None, title='Performance Metrics Across Threshold Levels'):
    """
    Plots IoU, precision, recall, and Dice/F1 across threshold values.

    Args:
        threshold_metrics (dict): Dictionary mapping thresholds to metric dictionaries.
        save_path (str or Path, optional): File path for saving the plot image.
        title (str): Figure title.
    """

    thresholds = sorted(threshold_metrics.keys())
    
    iou = [threshold_metrics[t]["iou"] for t in thresholds]
    precision = [threshold_metrics[t]["precision"] for t in thresholds]
    recall = [threshold_metrics[t]["recall"] for t in thresholds]
    dice = [threshold_metrics[t]["dice"] for t in thresholds] 

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, iou, label='IoU', color='royalblue', marker='.')
    plt.plot(thresholds, precision, label='Precision', color='darkorange', marker='.')
    plt.plot(thresholds, recall, label='Recall', color='green', marker='.')
    plt.plot(thresholds, dice, label='F1', color='crimson', marker='.')

    plt.xlabel('Threshold')
    plt.ylabel('Metric Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, 1.0)
    
    plt.tight_layout()
    _save_plot(save_path)
    plt.close()
