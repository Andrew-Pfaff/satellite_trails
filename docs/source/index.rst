Satellite Trail Segmentation
============================

This project implements and evaluates deep-learning pipelines for detecting satellite trails in astronomical images. It reproduces ASTA's patch-based U-Net and probabilistic-Hough workflow using the released PNG-derived images and masks, and adds classifier gating plus validation-selected postprocessing parameters.

The repository contains:

* H5 dataset preprocessing for full-field PNG images and masks.
* A patch classifier for deciding which patches contain trails.
* A baseline U-Net segmentation model.
* An Attention U-Net segmentation model.
* Training, hyperparameter tuning, ablation, evaluation, and CSD3 SLURM wrappers.
* Postprocessing utilities for converting raw segmentation masks into cleaner line detections.

Reproducibility scope
---------------------

The original ASTA paper used full-field FITS images. This project only has PNG-derived images and masks, so it cannot exactly reproduce the paper's FITS-intensity training conditions or the final large-scale survey/catalogue application. The report and results should therefore be read as a reproducibility study under a reduced-data setting, not as a direct pixel-for-pixel reproduction of the original pipeline.

Quick example
-------------

The public pipeline entry point is currently :mod:`segmentation.py`. A typical use is:

.. code-block:: python

   from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
   from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint
   from satellite_trail_segmentation.unet_model.unet import UNet
   from segmentation import SatelliteTrailPipeline

   unet = UNet()
   load_checkpoint("results/models/unet/unet_weights.pt", unet)

   classifier = TrailClassifier(base_channels=16)
   load_checkpoint("results/models/classifier/classifier_weights.pt", classifier)

   pipeline = SatelliteTrailPipeline(unet_model=unet, classifier_model=classifier)
   prepared = pipeline.preprocessing("data/png/example_full.png", normalization="source_zscore")
   prediction, times = pipeline.segmentation(
       prepared["patch_data"],
       use_classifier=True,
       unet_threshold=0.65,
       classifier_threshold=0.725,
   )
   postprocessed, postprocess_times, contour_details = pipeline.postprocessing(
       prediction["segmented_result"],
       hough_threshold=50,
       min_line_length=100,
       max_line_gap=125,
       contour_area_threshold=1500,
   )

This interface is useful for full-field PNG inference. The final evaluation scripts use the package evaluation utilities directly on H5 datasets.

Final results
-------------

The held-out test results, classifier-only patch metrics, and full-field timing experiment are summarized in :doc:`results`. The final full-field evaluation used ``UNET_THRESHOLD = 0.65`` and ``CLASSIFIER_THRESHOLD = 0.725`` and compared released ASTA postprocessing defaults against the validation-selected configuration.

Documentation map
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   full_field_pipeline
   data
   training
   parameter_search
   evaluation
   postprocessing
   results
   reproduction
   csd3
   colab
   limitations

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
