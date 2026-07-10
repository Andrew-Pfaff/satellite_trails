Postprocessing
==============

Postprocessing refines raw segmentation masks into cleaner line detections. The project uses Hough-style processing to reconnect broken trail segments and reduce false negatives caused by patch boundaries, stars, or small gaps.

The core postprocessing code lives in:

.. code-block:: text

   src/satellite_trail_segmentation/postprocess/

The full-field pipeline uses:

.. code-block:: python

   from satellite_trail_segmentation.postprocess.pipeline import postprocess_segmentation

Basic usage
-----------

The postprocessing entry point accepts a binary-like mask. Any nonzero value is treated as foreground, and the returned mask is a ``uint8`` array containing ``0`` and ``foreground_value``.

.. code-block:: python

   from satellite_trail_segmentation.postprocess.pipeline import postprocess_segmentation

   postprocessed = postprocess_segmentation(
       raw_prediction,
       mode="asta_only",
       hough_threshold=50,
       min_line_length=100,
       max_line_gap=250,
       morph_kernel_size=3,
       min_component_size=500,
   )

Available modes
---------------

The pipeline supports exactly two postprocessing configurations through ``mode``.

.. list-table::
   :header-rows: 1

   * - ``mode``
     - Behavior
   * - ``asta_only``
     - Detects Hough line segments, draws each detected segment with one-pixel thickness, then applies the shared cleanup stage. This is the default and matches the conservative ASTA-style setting.
   * - ``asta_gap_fill``
     - Starts from the ASTA-only Hough drawing, clusters detected segments into bounded representative centerlines, fills only support-anchored gaps, then applies the shared cleanup stage.

Mode guidance
-------------

Use ``mode="asta_only"`` as the conservative default. It draws real Hough-detected segments and does not estimate trail widths.

Use ``mode="asta_gap_fill"`` when the prediction has broken trails but you want to preserve the real ASTA Hough result. Width estimates are only used to draw bounded gap-fill segments where the fitted centerline is supported on both sides by existing foreground.

Gap-fill options
----------------

``width_samples``
   Number of positions sampled along each representative centerline when estimating the gap-fill drawing width.

``max_width_search``
   Perpendicular search radius used for sampled width estimation.

``fallback_width``
   Width used when sampled width estimation fails. The default is ``1``.

``min_fill_gap`` and ``max_fill_gap``
   Lower and upper bounds on support-anchored gaps that may be filled.

Hough detection options
-----------------------

``hough_threshold``
   Minimum accumulator threshold for OpenCV's probabilistic Hough transform. Higher values require stronger line evidence and usually produce fewer detections.

``min_line_length``
   Minimum Hough segment length in pixels. Increase this to ignore short fragments; decrease it if true trails are small.

``max_line_gap``
   Maximum gap OpenCV may bridge when forming Hough line segments.

Line clustering options
-----------------------

``asta_gap_fill`` clusters Hough segments before building representative centerlines.

``line_cluster_angle_degrees``
   Maximum orientation difference between segments in one cluster. Smaller values split trails more aggressively by angle.

``line_cluster_distance``
   Maximum perpendicular distance between segments in one cluster.

``line_cluster_max_along_gap``
   Maximum along-line gap between segments before they are split into separate clusters. Set to ``None`` in Python to disable this gap check.

``max_extension_ratio``
   Maximum representative-centerline span relative to the observed cluster endpoint span before clipping to the image boundary.

Cleanup options
---------------

After Hough drawing and optional gap filling, the pipeline applies cleanup steps:

``morph_kernel_size``
   Square kernel size for morphological closing. Values less than or equal to ``1`` effectively skip closing.

``min_component_size``
   Minimum connected foreground component area to keep. Increase this to remove more isolated small detections.

``contour_filter``
   Enables the final contour filter. The filter removes small contours below ``contour_area_threshold`` and can remove large contours that look too fragmented in Hough-angle space.

``contour_area_threshold``
   Minimum contour area kept by the final contour filter.

Contour details
---------------

Set ``contour_details=True`` when you want a structured summary of final detected contours:

.. code-block:: python

   mask, details = postprocess_segmentation(
       raw_prediction,
       contour_details=True,
       contour_min_area=10,
   )

The returned details dictionary contains ``contour_count`` and a ``contours`` list. Each contour record includes area, centroid, bounding box, length, width, angle, and approximate endpoint coordinates.

Full-field pipeline example
---------------------------

``segmentation.py`` passes postprocessing options through ``SatelliteTrailPipeline.postprocessing``:

.. code-block:: python

   postprocessed, timings, contour_details = pipeline.postprocessing(
       predictions["segmented_result"],
       mode="asta_gap_fill",
       hough_threshold=50,
       min_line_length=100,
       max_line_gap=250,
       max_extension_ratio=1.5,
       min_fill_gap=10,
       max_fill_gap=250,
       morph_kernel_size=3,
       min_component_size=500,
       contour_filter=True,
       contour_details=True,
   )

The same options are available on the command line through ``segmentation.py``:

.. code-block:: bash

   python segmentation.py \
     --source-path data/png/example_full.fits_full.png \
     --unet-checkpoint results/models/unet/unet_weights.pt \
     --postprocess-mode asta_gap_fill \
     --hough-threshold 50 \
     --min-line-length 100 \
     --max-line-gap 250 \
     --contour-details

Before/after evaluation
-----------------------

The final evaluation compares raw segmentation masks against postprocessed masks on the same full-field test examples.

Important metrics for postprocessing are:

* FNR reduction.
* Recall improvement.
* Precision loss, if any.
* IoU/Dice change.
* Visual continuity of broken trails.

The conservative ASTA-style setting can reduce FNR and improve recall, but it may also lower precision and IoU because Hough drawing introduces additional false positives. The gap-fill setting is available for cases where bounded, support-anchored filling is worth that trade-off.
