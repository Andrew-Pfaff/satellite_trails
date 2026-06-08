import numpy as np
import matplotlib.pyplot as plt

from satellite_trail_segmentation.evaluation.unet_evaluate import evaluate_dataset_unet
from satellite_trail_segmentation.evaluation.classifier_evaluate import evaluate_dataset_classifier
from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint
from satellite_trail_segmentation.unet_model.unet import UNet
from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
from satellite_trail_segmentation.utils.visualizations import plot_threshold_metrics, plot_roc_curve


if __name__ == "__main__":
    h5_path = "/home/anp50/rds/hpc-work/satellite_trails/data/h5s/dataset.h5"
    split = "val"

    unet_batch = 64
    classifier_batch = 128 

    unet_model_path = "/home/anp50/rds/hpc-work/satellite_trails/results/models/unet2/unet_weights.pt"
    classifier_model_path = "/home/anp50/rds/hpc-work/satellite_trails/results/models/classifier/classifier_weights.pt"

    unet_model = UNet()
    load_checkpoint(unet_model_path, unet_model)
    unet_model.eval()

    classifier_model = TrailClassifier()
    load_checkpoint(classifier_model_path, classifier_model)
    classifier_model.eval()

    test_thresholds = list(np.linspace(0.05, 0.95, 19))

    unet_metrics_counts, unet_patch_metrics, fpr, tpr, thresholds, optimal_threshold, roc_auc = evaluate_dataset_unet(unet_model, h5_path, split, test_thresholds, unet_batch)

    print("Unet metrics: ")
    print(unet_metrics_counts)

    classifier_metrics, classifier_image_wise_counts = evaluate_dataset_classifier(classifier_model, h5_path, split, test_thresholds, classifier_batch)

    print("\nClassifier metrics: ")
    print(classifier_metrics)
    print()
    print(classifier_image_wise_counts)


    plot_roc_curve(fpr, tpr, thresholds, roc_auc, optimal_threshold, save_path="/home/anp50/rds/hpc-work/satellite_trails/results/models/unet2/unet_val_roc_metrics.png")
    plot_threshold_metrics(unet_metrics_counts, save_path="/home/anp50/rds/hpc-work/satellite_trails/results/models/unet2/unet_val_metrics.png")
    plot_threshold_metrics(unet_patch_metrics, save_path="/home/anp50/rds/hpc-work/satellite_trails/results/models/unet2/unet_val_metrics_patch.png")
    plot_threshold_metrics(classifier_metrics, save_path="/home/anp50/rds/hpc-work/satellite_trails/results/models/classifier/classifier_val_metrics.png")
    