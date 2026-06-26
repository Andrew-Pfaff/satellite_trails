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

The final training scripts support:

* ``normalization=source_zscore`` by default.
* fixed-step weighted sampling with a requested positive patch fraction.
* Combo loss for segmentation, controlled by ``bce_weight_factor``.
* AdamW optimization.
* cosine learning-rate scheduling with warmup.
* early stopping.
* gradient clipping.
* shift augmentation with configurable shift probability and shift range.

Final checkpoints are saved under ``results/models/<model_name>/`` and should not be committed to the repository.
