Data
====

The project uses PNG-derived full-field images and masks from the ASTA data release. This is different from the ASTA paper, which trained and applied its model on FITS image data. The PNG format is easier to handle locally, but it does not preserve the full FITS intensity information.

Getting the ASTA PNG data
-------------------------

The data used to train the models in this reproducibility study was the PNG data released with the ASTA paper, *Automated Detection of Satellite Trails in Ground-Based Observations Using U-Net and Hough Transform*.

Links:

* Paper: `A&A article <https://www.aanda.org/articles/aa/full_html/2024/12/aa51663-24/aa51663-24.html>`__
* Preprint: `arXiv:2407.19461 <https://arxiv.org/abs/2407.19461>`__
* Data release: `Zenodo record 11642424 <https://zenodo.org/records/11642424>`__

Download the main PNG archive from Zenodo:

.. code-block:: bash

   mkdir -p data/raw
   wget -c -O data/raw/Processed.zip "https://zenodo.org/records/11642424/files/Processed.zip?download=1"

The archive is large, approximately 18 GB. The ``-c`` flag allows ``wget`` to resume the download if it is interrupted.

After downloading, extract the archive:

.. code-block:: bash

   unzip data/raw/Processed.zip -d data/raw/asta

The preprocessing scripts expect the full-field PNG images and mask PNGs to be available in the local project data area, typically ``data/png/``. Depending on where the archive extracts the files, either copy or symlink the PNGs into that directory. The local ``data/`` directory is intended for downloaded and generated data and should not be committed to git.

Zenodo also provides ``Satellites_Catalog_Application.csv`` with the application-section catalogue discussed in the ASTA paper. That table is useful for reproducing the catalogue-matching analysis, but it is not required for patch-level model training.

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
