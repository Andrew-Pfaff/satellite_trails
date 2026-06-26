Results
=======

Final results are pending while CSD3 tuning, ablation, and training jobs complete. This page is a placeholder for the final documentation tables and should be updated from the generated summary CSV files.

Final model metrics
-------------------

.. list-table::
   :header-rows: 1

   * - Model
     - Split
     - Threshold
     - IoU
     - Dice/F1
     - Precision
     - Recall
     - Specificity
     - FPR
     - FNR
   * - Classifier
     - Test
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
   * - Baseline U-Net
     - Test
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
   * - Attention U-Net
     - Test
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD
     - TBD

Planned figures
---------------

* Threshold sweep curves for each final model.
* Full-field reconstruction examples.
* Raw versus postprocessed segmentation masks.
* Failure cases: false positives, false negatives, patch-boundary errors, and postprocessing tradeoffs.
* Timing table by pipeline stage.

Comparison to ASTA
------------------

The final report should compare against the ASTA paper's reported patch-level precision/recall, threshold behavior, postprocessing FNR reduction, and processing-time table. The comparison must state that the original FITS data were unavailable.
