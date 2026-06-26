Reproducing The Project
=======================

This page summarizes the intended end-to-end workflow for reproducing the project. Paths and final commands should be checked once final results are frozen.

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

The final report should use validation thresholds for model/threshold selection and held-out test metrics for the final comparison.
