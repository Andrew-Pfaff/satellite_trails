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
       line_mode="asta",
       width_mode="none",
       hough_threshold=50,
       min_line_length=100,
       max_line_gap=250,
       morph_kernel_size=3,
       min_component_size=500,
   )

Available mode combinations
---------------------------

The two main options are ``line_mode`` and ``width_mode``.

.. list-table::
   :header-rows: 1

   * - ``line_mode``
     - ``width_mode``
     - Behavior
   * - ``asta``
     - ``none``
     - Detects Hough line segments and draws each detected segment with one-pixel thickness. This is the default.
   * - ``asta``
     - ``contour_width``
     - Starts with one-pixel ASTA-style Hough lines, then fills supported gaps along clustered centerlines using widths estimated from nearby contours.
   * - ``asta``
     - ``median_sampled_width``
     - Starts with one-pixel ASTA-style Hough lines, then fills supported gaps along clustered centerlines using widths sampled from the raw prediction.
   * - ``centerline``
     - ``contour_width``
     - Clusters Hough segments into representative centerlines and redraws each centerline with contour-estimated width.
   * - ``centerline``
     - ``median_sampled_width``
     - Clusters Hough segments into representative centerlines and redraws each centerline with median sampled width.

``line_mode="centerline"`` cannot be used with ``width_mode="none"`` because centerline mode needs a drawing width for each representative line.

Mode guidance
-------------

Use ``line_mode="asta", width_mode="none"`` as the conservative default. It extends detected trails without estimating a trail width, so it is less likely to thicken false positives.

Use an ``asta`` width mode when the raw prediction has broken trails but you want to preserve the one-pixel Hough result. In this mode, width estimates are only used to draw gap-fill segments where the centerline is supported on both sides by existing foreground.

Use ``line_mode="centerline"`` when you want a cleaner reconstructed line per Hough cluster. This can improve visual continuity, but it redraws full representative centerlines and can therefore change mask shape more aggressively than ASTA mode. In the final full-field evaluation, this mode increased false positives substantially and is therefore not recommended as the main reporting configuration.

Width modes
-----------

``contour_width``
   Measures connected foreground contours with minimum-area rectangles, matches each centerline to the nearest contour samples, and uses that contour width. ``max_contour_distance`` controls how far a line can be from a contour before ``fallback_width`` is used.

``median_sampled_width``
   Samples perpendicular profiles across each centerline and uses the median foreground run width. ``width_samples`` controls how many positions are sampled along the line, and ``max_width_search`` controls the perpendicular search radius.

``fallback_width``
   Used when width estimation fails or no matching foreground support is found. The default is ``1``.

Hough detection options
-----------------------

``hough_threshold``
   Minimum accumulator threshold for OpenCV's probabilistic Hough transform. Higher values require stronger line evidence and usually produce fewer detections.

``min_line_length``
   Minimum Hough segment length in pixels. Increase this to ignore short fragments; decrease it if true trails are small.

``max_line_gap``
   Maximum gap OpenCV may bridge when forming Hough line segments. This value is also used as the maximum gap length for ASTA width-mode gap filling.

Line clustering options
-----------------------

Width modes and ``centerline`` mode cluster Hough segments before building representative centerlines.

``line_cluster_angle_degrees``
   Maximum orientation difference between segments in one cluster. Smaller values split trails more aggressively by angle.

``line_cluster_distance``
   Maximum perpendicular distance between segments in one cluster.

``line_cluster_max_along_gap``
   Maximum along-line gap between segments before they are split into separate clusters. Set to ``None`` in Python to disable this gap check.

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
       line_mode="asta",
       width_mode="median_sampled_width",
       hough_threshold=50,
       min_line_length=100,
       max_line_gap=250,
       min_fill_gap=10,
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
     --line-mode asta \
     --width-mode median_sampled_width \
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

The conservative ASTA-style setting substantially reduces FNR and improves recall, but it also lowers precision and IoU because Hough drawing introduces additional false positives. The centerline variants were explored as alternatives, but they underperformed the ASTA-style setting because representative centerlines overdraw weak or fragmented detections.
