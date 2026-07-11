Quickstart
==========

Install the package in editable mode from the repository root:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .[dev,docs]

Run the test suite:

.. code-block:: bash

   python -m pytest -q

Build the HTML documentation:

.. code-block:: bash

   cd docs
   make html

The generated pages are written to ``docs/build/html``.

Full-field segmentation
-----------------------

The current full-field inference helper lives in ``segmentation.py``. It takes one full-field PNG image, splits it into ``528 x 528`` patches, optionally filters patches with the classifier, segments selected patches with a U-Net-style model, and optionally runs Hough postprocessing.

.. code-block:: python

   from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint
   from satellite_trail_segmentation.unet_model.unet import UNet
   from segmentation import SatelliteTrailPipeline

   model = UNet()
   load_checkpoint("results/models/unet/unet_weights.pt", model)

   pipeline = SatelliteTrailPipeline(unet_model=model)
   prepared = pipeline.preprocessing("data/png/example_full.png", normalization="source_zscore")
   prediction, timings = pipeline.segmentation(prepared["patch_data"], use_classifier=False, unet_threshold=0.65)
   postprocessed, postprocess_timings, contour_details = pipeline.postprocessing(
       prediction["segmented_result"],
       hough_threshold=50,
       min_line_length=100,
       max_line_gap=125,
       contour_area_threshold=1500,
   )

For more detailed full-field usage, see :doc:`full_field_pipeline`. 
