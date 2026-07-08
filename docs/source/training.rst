Training
========

Training scripts live in ``scripts/python/train`` and are wrapped by SLURM scripts in ``scripts/slurm/training``.

Classifier
----------

The classifier predicts whether a ``528 x 528`` patch contains a satellite trail. It can be used as a gating step before running the segmentation model on full-field images.

Local entry point:

.. code-block:: bash

   python scripts/python/train/train_classifier.py --help

CSD3 wrapper:

.. code-block:: bash

   sbatch scripts/slurm/training/classifier_train.slurm

Baseline U-Net
--------------

The baseline U-Net performs pixel-level segmentation and outputs raw logits. Apply a sigmoid before thresholding the probabilities.

Local entry point:

.. code-block:: bash

   python scripts/python/train/train_unet.py --help

CSD3 wrapper:

.. code-block:: bash

   sbatch scripts/slurm/training/unet_train.slurm

Attention U-Net
---------------

The Attention U-Net uses additive attention gates on skip connections while otherwise following the cleaned baseline U-Net conventions.

Local entry point:

.. code-block:: bash

   python scripts/python/train/train_attention_unet.py --help

CSD3 wrapper:

.. code-block:: bash

   sbatch scripts/slurm/training/attention_unet_train.slurm

Shared training settings
------------------------

All three training entry points read the H5 dataset from ``--data-path`` and train on the ``train`` split while monitoring the ``val`` split. The scripts use AdamW, a warmup plus cosine learning-rate schedule, optional fixed-step weighted sampling, optional shift augmentation, gradient clipping, and optional early stopping.

The most important shared options are:

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning
   * - ``--epochs``
     - Maximum number of epochs. Early stopping can stop before this limit.
   * - ``--batch-size``
     - Number of patches per optimization step.
   * - ``--learning-rate``
     - Initial AdamW learning rate.
   * - ``--weight-decay``
     - AdamW weight decay. Defaults to ``1e-4``.
   * - ``--normalization``
     - Input normalization mode. Choose from ``source_zscore``, ``patch_zscore``, or ``uint8``. Final wrappers default to ``source_zscore``.
   * - ``--sampler-fraction``
     - Positive patch fraction requested by the fixed-step weighted sampler. If omitted, regular shuffled loading is used.
   * - ``--steps-per-epoch``
     - Number of fixed sampler steps per epoch, expressed through ``steps_per_epoch * batch_size`` sampled patches. Defaults to ``800``.
   * - ``--warmup-epochs``
     - Number of linear warmup epochs before cosine decay. The Python default is ``10``; the SLURM wrappers use ``5``.
   * - ``--eta-min``
     - Minimum learning rate reached by cosine decay. Defaults to ``1e-6``.
   * - ``--grad-clip-max-norm``
     - Maximum gradient norm before clipping. Defaults to ``1.0``.
   * - ``--early-stopping-patience``
     - Number of validation epochs without sufficient improvement before stopping. The Python default is disabled; the SLURM wrappers use ``10``.
   * - ``--early-stopping-min-delta``
     - Minimum validation improvement required to reset early stopping. Defaults to ``0.0``.
   * - ``--p-shift``, ``--min-shift``, ``--max-shift``
     - Probability and pixel range for shift augmentation on the training split. The Python default disables shifts with ``p_shift=0.0``; the SLURM wrappers use ``0.25`` with shifts from ``15`` to ``100`` pixels.
   * - ``--full-save-path``
     - Path for the full training checkpoint, including model and optimizer state.
   * - ``--weight-save-path``
     - Path for model weights only.
   * - ``--plot-path``
     - Optional output path for training and validation loss curves.
   * - ``--seed``
     - Random seed. Defaults to ``1``.

Segmentation model settings
---------------------------

``train_unet.py`` and ``train_attention_unet.py`` share the same segmentation-specific settings. Both train from raw logits with the combo loss:

.. math::

   \mathrm{loss} = \alpha \cdot \mathrm{BCEWithLogits} + (1 - \alpha) \cdot \mathrm{DiceLoss}

where ``--bce-weight-factor`` is :math:`\alpha`.

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning
   * - ``--pos-weight``
     - Positive-class weight used by the BCE term. The Python default is ``1.0``; the SLURM wrappers use ``2.0``.
   * - ``--bce-weight-factor``
     - Weight of the BCE term in combo loss. Defaults to ``0.5``.
   * - ``--label-smoothing``
     - Label smoothing applied inside combo loss. Defaults to ``0.0``.
   * - ``--dropout-rate``
     - Dropout used inside the U-Net blocks. Defaults to ``0.0``.
   * - ``--no-batchnorm``
     - Disables batch normalization. Batch normalization is enabled by default.

During validation, segmentation checkpoints are selected by the best validation IoU over thresholds from ``0.05`` to ``0.95`` in 37 evenly spaced steps. The SLURM wrappers save final outputs under ``results/models/unet/`` and ``results/models/attention_unet/`` by default.

Classifier settings
-------------------

``train_classifier.py`` trains the patch classifier with weighted BCE plus a soft false-negative penalty. The best checkpoint is selected by validation specificity with a recall penalty, so final classifier training favors high recall before optimizing specificity.

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning
   * - ``--base-channels``
     - Width multiplier for the classifier convolution stack. The SLURM wrapper uses ``16``.
   * - ``--dropout-rate``
     - Classifier dropout rate. The Python default is ``0.0``; the SLURM wrapper uses ``0.2``.
   * - ``--pos-weight``
     - Positive-class weight for the BCE term. Defaults to ``3.0``.
   * - ``--fn-penalty-weight``
     - Weight of the additional false-negative penalty. The Python default is ``1.0``; the SLURM wrapper uses ``3.0``.
   * - ``--min-recall``
     - Recall target used by checkpoint selection. The Python default is ``0.98``; the SLURM wrapper uses ``0.99``.
   * - ``--recall-penalty``
     - Penalty strength when validation recall is below ``--min-recall``. The Python default is ``2.0``; the SLURM wrapper uses ``3.0``.

During validation, classifier metrics are swept over thresholds from ``0.05`` to ``0.95`` in 37 evenly spaced steps. The SLURM wrapper saves final outputs under ``results/models/classifier/`` by default.

Training checkpoints, model weights, and loss plots are runtime outputs and should not be committed to the repository.
