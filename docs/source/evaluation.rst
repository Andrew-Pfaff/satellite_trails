Evaluation
==========

The main evaluator is ``scripts/python/evaluate_models.py``. It supports:

* ``classifier``
* ``unet``
* ``attention_unet``

The CSD3 wrapper is:

.. code-block:: bash

   sbatch scripts/slurm/eval.slurm

It writes a one-row summary CSV, a per-threshold metrics CSV, and a threshold metric plot for all model types. Segmentation models also write an ROC plot.

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

Full-field evaluation
---------------------

``scripts/python/evaluate_final_full_field.py`` evaluates the final PNG pipeline on rows selected from ``master_split.csv``. It runs raw U-Net predictions and classifier-gated U-Net predictions, then applies both the released ASTA postprocessing defaults and the validation-selected parameters to each prediction.


The script writes one ``<method>_per_image_metrics.csv`` file per method and ``aggregate_metrics.csv`` in the output directory. Each row includes the confusion counts plus accuracy, precision, recall/sensitivity, specificity, FPR, FNR, IoU, and Dice/F1.

The six output methods are ``unet``, ``classifier_unet``, ``unet_postprocess_asta``, ``classifier_unet_postprocess_asta``, ``unet_postprocess_selected``, and ``classifier_unet_postprocess_selected``. Selected postprocessing values are supplied through the postprocessing CLI arguments; their defaults reproduce the final selected configuration.

Final protocol
--------------

The final report uses validation results for threshold selection and held-out test results for final comparison. ``evaluate_models.py`` is used for patch/H5 threshold selection and ``evaluate_final_full_field.py`` for final PNG full-field raw-versus-postprocessed comparisons.
