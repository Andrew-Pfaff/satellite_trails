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

External RECA masks
-------------------

The optional RECA/satmetrics evaluation uses Hough-derived masks rather than the same manual masks used for model training. Those results should be treated as an external sanity check rather than the primary benchmark.
