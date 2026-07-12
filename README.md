# Satellite Trail Segmentation

This repository contains a reproducibility study of satellite trail detection in astronomical images. It implements patch preprocessing, a patch classifier, baseline U-Net segmentation, an experimental Attention U-Net, and probabilistic-Hough postprocessing. The Attention U-Net was not trained to completion or included in the final evaluation.


## Documentation

The Sphinx documentation is the main user and API documentation surface.
It should be treated as the primary documentation entry point for installation, data preparation, training, evaluation, results, and API reference material.

Build it locally with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev,docs]
.venv/bin/sphinx-build -b html docs/source docs/build/html
```

Open:

```text
docs/build/html/index.html
```

The docs include quickstart usage, data notes, training, parameter search, evaluation, postprocessing, CSD3 and Colab workflows, limitations, and API documentation.


## Quick Setup

```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/anp50.git satellite_trails
cd satellite_trails
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Optionally
```bash
pip install -e .[dev,docs]
python -m pytest -q
```

## Data

The models in this repository were trained on the PNG data released with the ASTA paper, "Automated Detection of Satellite Trails in Ground-Based Observations Using U-Net and Hough Transform".

- Paper: [A&A article](https://www.aanda.org/articles/aa/full_html/2024/12/aa51663-24/aa51663-24.html) and [arXiv preprint](https://arxiv.org/abs/2407.19461)
- Dataset: [Zenodo record 11642424](https://zenodo.org/records/11642424)

Download the main ASTA PNG archive with:

```bash
mkdir -p data/raw
wget -c -O data/raw/Processed.zip "https://zenodo.org/records/11642424/files/Processed.zip?download=1"
```

The archive is large, about 18 GB. After downloading, extract it and place or symlink the PNG images and masks into the local `data/png/` layout used by the preprocessing scripts. The original ASTA paper trained and applied its pipeline on FITS images; this reproducibility work uses the released PNG data.

## Main Workflows

Preprocess data on CSD3:

```bash
sbatch scripts/slurm/dataset/preprocess_dataset.slurm
```

Run model tuning:

```bash
sbatch scripts/slurm/parameter_search/classifier_tuning.slurm
sbatch scripts/slurm/parameter_search/unet_tuning.slurm
sbatch scripts/slurm/parameter_search/attention_unet_tuning.slurm
```

Run model ablations:

```bash
sbatch scripts/slurm/parameter_search/classifier_ablation.slurm
sbatch scripts/slurm/parameter_search/unet_ablation.slurm
```

Reproduce the final report training runs with the Colab notebooks:

- `scripts/colab/classifier_train_colab.ipynb`
- `scripts/colab/unet_train_colab.ipynb`

These notebooks contain the exact selected settings used for the reported classifier and U-Net. In particular, the final U-Net used 200 steps per epoch, positive weight 1.0, no shift augmentation, and batch normalisation. The training SLURM wrappers are generic CSD3 templates whose defaults do not reproduce the final checkpoints. They can be adapted by overriding their environment variables with the values recorded in the Colab notebooks.

Evaluate the completed models:

```bash
MODEL_TYPE=classifier SPLIT=test sbatch scripts/slurm/eval.slurm
MODEL_TYPE=unet SPLIT=test sbatch scripts/slurm/eval.slurm
```

Run the final PNG full-field comparison with:

```bash
python scripts/python/evaluate_final_full_field.py --png-dir data/png --master-split-csv data/h5s/master_split.csv --unet-checkpoint results/models/unet/unet_weights.pt --classifier-checkpoint results/models/classifier/classifier_weights.pt --split-id 2 --output-dir results/metrics/test --unet-threshold 0.65 --classifier-threshold 0.725 --hough-threshold 50 --min-line-length 100 --max-line-gap 125 --morph-kernel-size 3 --min-component-size 500 --contour-area-threshold 1500
```

The command writes per-image CSVs for the two raw predictions, both predictions with released ASTA defaults, and both predictions with the selected parameters, plus ``aggregate_metrics.csv`` containing all six aggregate rows.

## Final Test Metrics

Aggregate full-field metrics on 26 held-out test images:

Selected operating thresholds were `UNET_THRESHOLD = 0.65` and `CLASSIFIER_THRESHOLD = 0.725`. The evaluator reports both the released ASTA postprocessing defaults and the validation-selected configuration (`hough_threshold=50`, `min_line_length=100`, `max_line_gap=125`, `morph_kernel_size=3`, `min_component_size=500`, and `contour_area_threshold=1500`).

| Method | IoU | Dice/F1 | Precision | Recall | FNR | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classifier U-Net | 0.8202 | 0.9012 | 0.8893 | 0.9135 | 0.0865 | 7.02e-05 |
| U-Net | 0.8131 | 0.8969 | 0.8801 | 0.9144 | 0.0856 | 7.68e-05 |
| Classifier U-Net + selected postprocessing | 0.8100 | 0.8950 | 0.8484 | 0.9470 | 0.0530 | 1.04e-04 |
| U-Net + selected postprocessing | 0.8095 | 0.8947 | 0.8465 | 0.9487 | 0.0513 | 1.06e-04 |
| Classifier U-Net + ASTA defaults | 0.7961 | 0.8865 | 0.8283 | 0.9534 | 0.0466 | 1.22e-04 |
| U-Net + ASTA defaults | 0.7914 | 0.8836 | 0.8231 | 0.9537 | 0.0463 | 1.27e-04 |

## Submission Notes

The large ASTA PNG archive, generated H5 datasets, and most runtime result directories are not stored in version control because of size. The best model checkpoints used for the submitted results are included with the submission, and the remaining generated artifacts can be reproduced through the documented workflows.

## Repository Layout

```text
satellite_trails/
├── src/
│   └── satellite_trail_segmentation/   Python package
├── scripts/
│   ├── python/                         CLI entry points
│   └── slurm/                          CSD3 job wrappers
├── docs/                               Sphinx documentation
├── tests/                              Unit tests
├── data/                               Local data, ignored except small metadata
├── results/
│   └── models/                         Committed final classifier and U-Net checkpoints
```

## Note on the Use of Autogeneration Tools

AI tools, such as OpenAI Codex and Gemini, were used during development for the following tasks:

- Reviewing code and suggesting debugging fixes
- Drafting and revising documentation, comments, and docstrings
- Helping structure data preprocessing, HDF5 dataset generation, and split metadata workflows
- Helping implement seeding and checkpointing utilities
- Helping set up the postprocessing pipeline
- Writing and modifying plotting utilities
- Optimizing torch training code
- Updating UNet training code and evaluation code to work with Attention U-Net
- Writing command-line interfaces for scripts
- Adding and updating SLURM scripts, including command-line interfaces
- Writing tests using proposer critic workflow
- Porting workflows to Google Colab notebooks
- Checking repository structure and submission requirements

All code, results, and written claims were reviewed and edited by me.
