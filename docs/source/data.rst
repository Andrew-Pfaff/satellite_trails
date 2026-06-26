Data
====

The project uses PNG-derived full-field images and masks. This is different from the ASTA paper, which trained and applied its model on FITS image data. The PNG format is easier to handle locally, but it does not preserve the full FITS intensity information.

H5 datasets
-----------

Training and evaluation use H5 files created from the PNG images and masks. The expected datasets include:

.. code-block:: text

   data/h5s/dataset.h5
   data/h5s/half_dataset.h5

Each H5 file stores patch arrays and metadata such as:

.. code-block:: text

   images
   masks
   source_index
   source_split
   source_mean
   source_std
   patch_has_trail

The split convention is:

.. list-table::
   :header-rows: 1

   * - Split
     - Code
   * - Train
     - ``0``
   * - Validation
     - ``1``
   * - Test
     - ``2``

Normalization modes
-------------------

The dataset supports three normalization modes:

``source_zscore``
   Standardize each patch using the mean and standard deviation of its source full-field image. This is the default training/evaluation mode.

``patch_zscore``
   Standardize each patch using its own mean and standard deviation.

``uint8``
   Scale PNG-derived pixel values to ``[0, 1]``.

Source-level statistics
-----------------------

``source_zscore`` requires ``source_mean`` and ``source_std`` in the H5 file. These values are computed during preprocessing and preserve the normalization convention used during model training.

PNG/FITS limitation
-------------------

The missing FITS files are a central limitation. PNG images are quantized and contrast-scaled products of the original data, so the model cannot see the same brightness distribution available to the paper's original FITS-based pipeline. This limitation should be reported alongside final metrics.
