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
   * - ``scripts/colab/save_validation_predictions_colab.ipynb``
     - Runs the final U-Net on ``data/val_images`` and saves one ``*_prediction.png`` beside each validation mask for postprocessing selection.
   * - ``scripts/colab/tune_postprocessing_colab.ipynb``
     - Runs the 36-configuration validation grid and writes aggregate confusion counts and metrics for every configuration plus the raw baseline.
   * - ``scripts/colab/final_full_field_evaluation_colab.ipynb``
     - Runs the final full-field PNG evaluation workflow used for report metrics, including raw predictions, released ASTA postprocessing defaults, and validation-selected postprocessing for both U-Net variants.
   * - ``scripts/colab/pipeline_timing_colab.ipynb``
     - Benchmarks the end-to-end full-field pipeline on CPU and A100 GPU runtimes for the computation timing comparison.

Relationship to CSD3 scripts
----------------------------

The final reported classifier and U-Net were trained with the Colab notebooks above after CSD3 access became unavailable. The scripts in ``scripts/slurm/`` provide alternative HPC wrappers, but their training defaults are generic and do not match every selected final-model setting. The Colab training notebooks are the record of the final training commands.
