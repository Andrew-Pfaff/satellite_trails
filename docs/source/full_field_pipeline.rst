Full-Field Pipeline
===================

``segmentation.py`` is the main user-facing entry point for running the trained models on one full-field PNG image. It loads a source image, tiles it into ``528 x 528`` patches by default, optionally filters patches with the classifier, runs U-Net segmentation, applies postprocessing, and optionally exports masks, plots, and metrics.

Use this page when you want to apply the pipeline to a PNG image. Use :doc:`evaluation` when you want split-level model metrics from H5 datasets or final full-field report CSVs.

Input requirements
------------------

The source image must be a 2D PNG image whose height and width are both divisible by ``patch_dim``. The default ``patch_dim`` is ``528``. If a mask is provided, it must have the same full-field shape as the source image.

The pipeline supports three normalization modes:

``source_zscore``
   Standardize each patch using the source image mean and standard deviation. This is the default mode, as it is the mode the models within the repo have been trained with.

``patch_zscore``
   Standardize each patch using its own mean and standard deviation.

``uint8``
   Scale PNG pixel values to ``[0, 1]``.

Python usage
------------

Load the model checkpoints, construct ``SatelliteTrailPipeline``, and call the pipeline stages directly when you want control over intermediate outputs.

.. code-block:: python

   from segmentation import SatelliteTrailPipeline
   from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
   from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint
   from satellite_trail_segmentation.unet_model.unet import UNet

   unet = UNet()
   load_checkpoint("results/models/unet/unet_weights.pt", unet)

   classifier = TrailClassifier(base_channels=16)
   load_checkpoint("results/models/classifier/classifier_weights.pt", classifier)

   pipeline = SatelliteTrailPipeline(
       unet_model=unet,
       classifier_model=classifier,
       patch_dim=528,
   )

   prepared = pipeline.preprocessing(
       "data/png/example_full.fits_full.png",
       mask_path="data/png/example_full_mask.png",
       normalization="source_zscore",
   )

   predictions, segmentation_times = pipeline.segmentation(
       prepared["patch_data"],
       use_classifier=True,
       unet_threshold=0.65,
       classifier_threshold=0.725,
   )

   postprocessed, postprocess_times, contour_details = pipeline.postprocessing(
       predictions["segmented_result"],
       line_mode="asta",
       width_mode="none",
   )

   metrics = pipeline.evaluate_masks(
       predictions["segmented_result"],
       postprocessed,
       prepared["mask"],
   )

   output_paths = pipeline.export_outputs(
       prepared["image"],
       postprocessed,
       mask=prepared["mask"],
       metrics=metrics,
       export_png=True,
       export_plots=True,
       metrics_csv=True,
       output_dir="outputs",
       output_prefix="example_full",
   )

Single-call usage
-----------------

The module-level ``main`` function runs preprocessing, segmentation, postprocessing, optional evaluation, and optional exports in one call.

.. code-block:: python

   from segmentation import main
   from satellite_trail_segmentation.classifier_model.classifier import TrailClassifier
   from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint
   from satellite_trail_segmentation.unet_model.unet import UNet

   unet = UNet()
   load_checkpoint("results/models/unet/unet_weights.pt", unet)

   classifier = TrailClassifier(base_channels=16)
   load_checkpoint("results/models/classifier/classifier_weights.pt", classifier)

   times, results = main(
       unet,
       "data/png/example_full.fits_full.png",
       mask_path="data/png/example_full_mask.png",
       classifier_model=classifier,
       use_classifier=True,
       normalization="source_zscore",
       unet_threshold=0.65,
       classifier_threshold=0.725,
       output_dir="outputs",
       output_prefix="example_full",
   )

Command-line usage
------------------

Run the same full-field workflow from the repository root:

.. code-block:: bash

   python segmentation.py \
     --source-path data/png/example_full.fits_full.png \
     --mask-path data/png/example_full_mask.png \
     --unet-checkpoint results/models/unet/unet_weights.pt \
     --classifier-checkpoint results/models/classifier/classifier_weights.pt \
     --use-classifier \
     --normalization source_zscore \
     --unet-threshold 0.65 \
     --classifier-threshold 0.725 \
     --output-dir outputs \
     --output-prefix example_full

If no mask is available, omit ``--mask-path``. The pipeline will still export the predicted mask and plot, but it will not write metrics.

Outputs
-------

With the default export flags, outputs are written under ``output_dir`` using ``output_prefix``:

.. list-table::
   :header-rows: 1

   * - Output
     - Description
   * - ``<output_prefix>_mask.png``
     - Final postprocessed binary prediction mask.
   * - ``<output_prefix>_plot.png``
     - Side-by-side image and prediction plot, with the true mask included when provided.
   * - ``<output_prefix>_metrics.csv``
     - Raw prediction, postprocessed prediction, and delta metrics. Written only when a mask is provided.

The Python API also returns timing dictionaries for preprocessing, segmentation, and postprocessing. When a mask is provided, it returns metrics for the raw prediction and postprocessed mask.

Common settings
---------------

``use_classifier``
   When true, the classifier first decides which patches should be segmented by the U-Net. This can reduce work and suppress patch-level false positives, but missed classifier positives prevent those patches from being segmented.

``unet_threshold``
   Probability threshold used to binarize U-Net pixel predictions.

``classifier_threshold``
   Probability threshold used to decide whether a patch contains a trail.

``line_mode`` and ``width_mode``
   Postprocessing mode controls. ``line_mode="asta"`` with ``width_mode="none"`` is the default. See :doc:`postprocessing` for the available Hough-style options.

API reference
-------------

For the generated class and function reference, see :doc:`api/pipeline`.
