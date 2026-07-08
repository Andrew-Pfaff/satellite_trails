Reproducing The Project
=======================

This page summarizes the end-to-end workflow used to reproduce the project. The commands assume the repository layout described in the user guide; adjust paths only if data or checkpoint files are stored elsewhere.

1. Install the environment
--------------------------

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .[dev,docs]

2. Prepare H5 datasets
----------------------

Use the dataset preprocessing SLURM wrapper on CSD3:

.. code-block:: bash

   sbatch scripts/slurm/dataset/preprocess_dataset.slurm

Expected H5 outputs include ``dataset.h5`` and ``half_dataset.h5`` under ``data/h5s/``.

3. Run tuning and ablations
---------------------------

Run Optuna studies on ``half_dataset.h5``:

.. code-block:: bash

   sbatch scripts/slurm/parameter_search/classifier_tuning.slurm
   sbatch scripts/slurm/parameter_search/unet_tuning.slurm
   sbatch scripts/slurm/parameter_search/attention_unet_tuning.slurm

Run ablations for classifier and baseline U-Net:

.. code-block:: bash

   sbatch scripts/slurm/parameter_search/classifier_ablation.slurm
   sbatch scripts/slurm/parameter_search/unet_ablation.slurm

4. Train final models
---------------------

Train final selected models on ``dataset.h5``:

.. code-block:: bash

   sbatch scripts/slurm/training/classifier_train.slurm
   sbatch scripts/slurm/training/unet_train.slurm
   sbatch scripts/slurm/training/attention_unet_train.slurm

5. Evaluate final models
------------------------

Evaluate validation and test splits:

.. code-block:: bash

   MODEL_TYPE=classifier SPLIT=val sbatch scripts/slurm/eval.slurm
   MODEL_TYPE=classifier SPLIT=test sbatch scripts/slurm/eval.slurm
   MODEL_TYPE=unet SPLIT=val sbatch scripts/slurm/eval.slurm
   MODEL_TYPE=unet SPLIT=test sbatch scripts/slurm/eval.slurm
   MODEL_TYPE=attention_unet SPLIT=val sbatch scripts/slurm/eval.slurm
   MODEL_TYPE=attention_unet SPLIT=test sbatch scripts/slurm/eval.slurm

Use validation metrics for threshold selection and held-out test metrics for final comparison.

Run the final PNG full-field comparison:

.. code-block:: bash

   python scripts/python/evaluate_final_full_field.py --png-dir data/png --master-split-csv data/h5s/master_split.csv --unet-checkpoint results/models/unet/unet_weights.pt --classifier-checkpoint results/models/classifier/classifier_weights.pt --split-id 2 --unet-threshold 0.65 --classifier-threshold 0.725 --output-dir results/metrics/test
