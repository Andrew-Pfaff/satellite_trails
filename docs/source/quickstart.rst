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
   prediction, timings = pipeline.segmentation(prepared["patch_data"], use_classifier=False, unet_threshold=0.5)
   postprocessed, postprocess_timings = pipeline.postprocessing(prediction["segmented_result"])

For report-quality evaluation, prefer the H5 evaluation scripts described in :doc:`evaluation`.

Important paths
---------------

.. list-table::
   :header-rows: 1

   * - Path
     - Purpose
   * - ``data/png/``
     - Full-field PNG images and masks.
   * - ``data/h5s/``
     - H5 datasets created from PNG images and masks.
   * - ``scripts/slurm/``
     - CSD3 job wrappers.
   * - ``scripts/python/train/``
     - Training entry points.
   * - ``scripts/python/parameter_search/``
     - Optuna tuning and ablation entry points.
   * - ``scripts/python/evaluate_models.py``
     - Patch-level classifier and pixel-level segmentation evaluator.
   * - ``results/``
     - Ignored runtime outputs, checkpoints, logs, summaries, and figures.
