Google Colab Workflow
=====================

Google Colab was used as a fallback runtime for the final training and final evaluation runs because CSD3 outages made the normal SLURM workflow unavailable. The Colab notebooks are kept in ``scripts/colab/`` so the fallback workflow can be reproduced or inspected alongside the CSD3 scripts.

To use these notebooks, open them in Google Colab from the repository, mount Google Drive when prompted, adjust the paths to match your Drive layout, and run the cells from top to bottom. The notebooks generally copy large datasets or PNGs from Drive to Colab's local runtime storage before running Python scripts, which is much faster than training or evaluating directly from Drive.

Notebook guide
--------------

.. list-table::
   :header-rows: 1

   * - Notebook
     - Purpose
   * - ``scripts/colab/git_clone.ipynb``
     - Mounts Google Drive and clones or refreshes the project repository in the Colab environment.
   * - ``scripts/colab/classifier_train_colab.ipynb``
     - Copies the H5 dataset to local Colab storage, trains the patch classifier, and writes classifier checkpoints and training curves back to Drive.
   * - ``scripts/colab/unet_train_colab.ipynb``
     - Copies the H5 dataset to local Colab storage, trains the baseline U-Net, and writes U-Net checkpoints and training curves back to Drive.
   * - ``scripts/colab/evaluate_models_colab.ipynb``
     - Runs the H5-based model evaluator for trained classifier and segmentation checkpoints, then saves summary and threshold metric outputs.
   * - ``scripts/colab/final_full_field_evaluation_colab.ipynb``
     - Runs the final full-field PNG evaluation workflow used for report metrics, including raw and postprocessed pipeline variants.
   * - ``scripts/colab/plot_all_pipelines_colab.ipynb``
     - Loads trained checkpoints and PNG examples, runs the available pipeline variants, and creates comparison plots for inspection or reporting.
   * - ``scripts/colab/pipeline_timing_colab.ipynb``
     - Benchmarks the end-to-end full-field pipeline on CPU and A100 GPU runtimes for the computation timing comparison.

Relationship to CSD3 scripts
----------------------------

The CSD3 scripts in ``scripts/slurm/`` remain the intended HPC workflow. The Colab notebooks mirror the same project scripts where possible, but wrap them with Drive mounting, local runtime copies, and notebook-specific output paths.
