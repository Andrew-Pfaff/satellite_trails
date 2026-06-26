Evaluation
==========

The main evaluator is ``scripts/python/evaluate_models.py``. It supports:

* ``classifier``
* ``unet``
* ``attention_unet``

The CSD3 wrapper is:

.. code-block:: bash

   sbatch scripts/slurm/eval.slurm

Examples
--------

Evaluate a baseline U-Net on the validation split:

.. code-block:: bash

   MODEL_TYPE=unet SPLIT=val sbatch scripts/slurm/eval.slurm

Evaluate an Attention U-Net on the test split:

.. code-block:: bash

   MODEL_TYPE=attention_unet SPLIT=test sbatch scripts/slurm/eval.slurm

Evaluate a classifier:

.. code-block:: bash

   MODEL_TYPE=classifier SPLIT=val sbatch scripts/slurm/eval.slurm

Metrics
-------

Segmentation models are evaluated pixel-wise across thresholds. The main ranking metric is IoU. The summary also includes precision, recall, Dice/F1, specificity, FPR, FNR, ROC AUC, and selected thresholds.

The classifier is evaluated patch-wise across thresholds. The tuning objective is penalized specificity under a high-recall constraint.

Final protocol
--------------

The final report should use validation results for threshold selection and held-out test results for final comparison. A final combined evaluation script is still planned to automate this validation-to-test workflow.
