Postprocessing
==============

Postprocessing starts from a binary prediction and uses probabilistic Hough line detection, morphological closing, connected-component removal, and contour filtering to reconnect broken trail segments and reduce false negatives.

The core postprocessing code lives in:

.. code-block:: text

   src/satellite_trail_segmentation/postprocess/

Basic usage
-----------

The postprocessing entry point accepts a binary-like mask. Any nonzero value is treated as foreground, and the returned mask is a ``uint8`` array containing ``0`` and ``foreground_value``.

.. code-block:: python

   from satellite_trail_segmentation.postprocess.pipeline import postprocess_segmentation

   postprocessed = postprocess_segmentation(
       raw_prediction,
       hough_threshold=50,
       min_line_length=100,
       max_line_gap=125,
       morph_kernel_size=3,
       min_component_size=500,
       contour_area_threshold=1500,
   )

Pipeline
--------

The live postprocessing pipeline performs the following sequence:

* Detect probabilistic Hough line segments.
* Draw each detected segment with one-pixel thickness onto a copy of the input prediction.
* Apply morphological closing.
* Remove small connected components.
* Optionally remove small and highly fragmented contours using a contour-local Hough transform and DBSCAN angle clustering.

The experimental width-aware gap-fill mode used during development is not part of the live pipeline.

Final configurations
--------------------

The final full-field evaluator runs two configurations. The validation-selected configuration uses ``hough_threshold=50``, ``min_line_length=100``, ``max_line_gap=125``, ``morph_kernel_size=3``, ``min_component_size=500``, and ``contour_area_threshold=1500``. The released ASTA defaults use the same values except ``max_line_gap=250`` and ``contour_area_threshold=3000``. The selected configuration is the default in ``evaluate_final_full_field.py``; the general ``postprocess_segmentation`` API retains the released ASTA values as its function defaults.

Hough detection options
-----------------------

``hough_threshold``
   Minimum accumulator threshold for OpenCV's probabilistic Hough transform. Higher values require stronger line evidence and usually produce fewer detections.

``min_line_length``
   Minimum Hough segment length in pixels. Increase this to ignore short fragments; decrease it if true trails are small.

``max_line_gap``
   Maximum gap OpenCV may bridge when forming Hough line segments.

Cleanup options
---------------

After Hough drawing, the pipeline applies cleanup steps:

``morph_kernel_size``
   Square kernel size for morphological closing. Values less than or equal to ``1`` effectively skip closing.

``min_component_size``
   Minimum connected foreground component area to keep. Increase this to remove more isolated small detections.

``contour_filter``
   Enables the final contour filter. The filter removes small contours below ``contour_area_threshold`` and can remove large contours that look too fragmented in Hough-angle space.

``contour_area_threshold``
   Minimum contour area kept by the final contour filter.

The contour-local Hough and DBSCAN parameters are also exposed by ``postprocess_segmentation``. Their released ASTA defaults are Hough threshold ``50``, minimum line length ``100``, maximum line gap ``10``, DBSCAN ``eps=5``, ``min_samples=1``, and rejection at five orientation clusters.

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
       hough_threshold=50,
       min_line_length=100,
       max_line_gap=125,
       morph_kernel_size=3,
       min_component_size=500,
       contour_filter=True,
       contour_area_threshold=1500,
       contour_details=True,
   )

The same options are available on the command line through ``segmentation.py``:

.. code-block:: bash

   python segmentation.py \
     --source-path data/png/example_full.fits_full.png \
     --unet-checkpoint results/models/unet/unet_weights.pt \
     --hough-threshold 50 \
     --min-line-length 100 \
     --max-line-gap 125 \
     --contour-area-threshold 1500 \
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

On the held-out test set, the selected setting gives raw-U-Net IoU ``0.8095`` and FNR ``0.0513`` after postprocessing, compared with IoU ``0.7914`` and FNR ``0.0463`` under the ASTA defaults. The selected setting therefore preserves overlap better, while the ASTA defaults recover more missed trail pixels.
