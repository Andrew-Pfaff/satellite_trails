CSD3 Workflow
=============

The SLURM scripts assume the project lives at:

.. code-block:: text

   /home/anp50/rds/hpc-work/satellite_trails

and that the environment is available at:

.. code-block:: text

   /home/anp50/rds/hpc-work/satellite_trails/.venv

Most scripts use the ``ampere`` partition and request one GPU. Tune/training jobs are intended for A100-class hardware.

Result layout
-------------

Runtime outputs are written under ``results/``:

.. code-block:: text

   results/models/
   results/hyperparameter_tuning/

These outputs should be copied into report tables/figures as needed but should not be committed to the repository.

Safety note
-----------

Do not pull or change the CSD3 working tree while submitted jobs are pending or running. A pending SLURM job will use whatever code is present when the job starts.

Record the commit used for each run:

.. code-block:: bash

   git rev-parse HEAD
