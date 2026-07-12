Limitations
===========

FITS data unavailable
---------------------

The most important limitation is the lack of original FITS images. The ASTA paper trained and applied the released model on FITS data, while this project uses PNG-derived full-field images and masks. PNG images have reduced and transformed brightness information, so final metrics should not be expected to match the paper exactly.

No large-scale catalogue reproduction
-------------------------------------

The original paper applied ASTA to around 200,000 full-field MeerLICHT images and compared detections against satellite catalogues. That final survey-scale step is outside the scope of this project because the required FITS/full-survey data are not available.

Patch-based context
-------------------

The models operate on ``528 x 528`` patches. Trails can cross patch boundaries, which can create fragmented predictions. Hough postprocessing is used to reduce this problem, but overlapping patches or full-field models would provide a more direct solution.

Incomplete Attention U-Net experiment
-------------------------------------

The Attention U-Net architecture and its training utilities were implemented, but the tuning and training study could not be completed after access to data and computational resources was lost. No final Attention U-Net checkpoint or test metrics are included, so the reported model comparison is limited to the baseline U-Net and classifier-gated U-Net.
