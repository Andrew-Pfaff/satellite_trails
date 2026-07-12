CSD3 Workflow
=============

The submitted CSD3 runs used the following project path. Adjust this path for another CSD3 account or workspace.

.. code-block:: text

   /home/anp50/rds/hpc-work/satellite_trails

The submitted runs also used this virtual environment path:

.. code-block:: text

   /home/anp50/rds/hpc-work/satellite_trails/.venv

Most scripts use the ``ampere`` partition and request one GPU. Tune/training jobs are intended for A100-class hardware.

Result layout
-------------

Runtime outputs are written under ``results/``:

.. code-block:: text

   results/models/
   results/hyperparameter_tuning/

Most generated tuning databases, logs, plots, and intermediate checkpoints are not committed. The final classifier and baseline U-Net checkpoints used for the submitted evaluation are deliberate exceptions and are tracked under ``results/models/classifier/`` and ``results/models/unet/``. No final Attention U-Net checkpoint was produced.

Safety note
-----------

Do not pull or change the CSD3 working tree while submitted jobs are pending or running. A pending SLURM job will use whatever code is present when the job starts.

Record the commit used for each run:

.. code-block:: bash

   git rev-parse HEAD
