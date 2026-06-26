Postprocessing
==============

Postprocessing refines raw segmentation masks into cleaner line detections. The project uses Hough-style processing to reconnect broken trail segments and reduce false negatives caused by patch boundaries, stars, or small gaps.

The core postprocessing code lives in:

.. code-block:: text

   src/satellite_trail_segmentation/postprocess/

The full-field pipeline uses:

.. code-block:: python

   from satellite_trail_segmentation.postprocess.hough import postprocess_segmentation

Before/after evaluation
-----------------------

For final reporting, compare raw segmentation masks against postprocessed masks on the same full-field examples and, where possible, aggregate metrics over the test split.

Important metrics for postprocessing are:

* FNR reduction.
* Recall improvement.
* Precision loss, if any.
* IoU/Dice change.
* visual continuity of broken trails.

The ASTA paper reported a substantial FNR reduction after Hough postprocessing. This project should report the same type of before/after comparison, while noting the PNG/FITS limitation.
