Parameter Search
================

Hyperparameter tuning and ablation scripts live in ``scripts/python/parameter_search``. Their CSD3 wrappers live in ``scripts/slurm/parameter_search``.

Optuna tuning
-------------

Classifier tuning:

.. code-block:: bash

   sbatch scripts/slurm/parameter_search/classifier_tuning.slurm

Baseline U-Net tuning:

.. code-block:: bash

   sbatch scripts/slurm/parameter_search/unet_tuning.slurm

Attention U-Net tuning:

.. code-block:: bash

   sbatch scripts/slurm/parameter_search/attention_unet_tuning.slurm

Attention U-Net tuning used ``p_shift=0.25`` and ``use_batchnorm=True``, matching the selected baseline U-Net training configuration.

Ablations
---------

Classifier and baseline U-Net ablations use the top Optuna rows and rerun selected variants for longer.

Classifier:

.. code-block:: bash

   sbatch scripts/slurm/parameter_search/classifier_ablation.slurm

Baseline U-Net:

.. code-block:: bash

   sbatch scripts/slurm/parameter_search/unet_ablation.slurm

The classifier ablation compares shift augmentation off/on. The baseline U-Net ablation compares shift augmentation off/on and batch normalization on/off.

Outputs
-------

Tuning and ablation outputs are written under:

.. code-block:: text

   results/hyperparameter_tuning/dbs/
   results/hyperparameter_tuning/summaries/
   results/hyperparameter_tuning/logs/
   results/hyperparameter_tuning/ablations/

These files are runtime artifacts and are not committed.
