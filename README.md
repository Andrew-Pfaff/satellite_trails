# Satellite Trail Segmentation

This repository contains a reproducibility study of satellite trail detection in astronomical images. It implements: patch preprocessing, a patch classifier, baseline U-Net segmentation, Attention U-Net segmentation, and Hough-style postprocessing.


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
git clone <repo-url> satellite_trails
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

Train final models:

```bash
sbatch scripts/slurm/training/classifier_train.slurm
sbatch scripts/slurm/training/unet_train.slurm
sbatch scripts/slurm/training/attention_unet_train.slurm
```

Evaluate models:

```bash
MODEL_TYPE=classifier SPLIT=test sbatch scripts/slurm/eval.slurm
MODEL_TYPE=unet SPLIT=test sbatch scripts/slurm/eval.slurm
MODEL_TYPE=attention_unet SPLIT=test sbatch scripts/slurm/eval.slurm
```

Run the final PNG full-field comparison with:

```bash
python scripts/python/evaluate_final_full_field.py --png-dir data/png --master-split-csv data/h5s/master_split.csv --unet-checkpoint results/models/unet/unet_weights.pt --classifier-checkpoint results/models/classifier/classifier_weights.pt --split-id 2 --output-dir final_eval_outputs --unet-threshold 0.65 --classifier-threshold 0.725
```

## Final Test Metrics

Aggregate full-field metrics on 26 held-out test images:

| Method | IoU | Dice/F1 | Precision | Recall | FNR | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classifier U-Net | 0.8202 | 0.9012 | 0.8893 | 0.9135 | 0.0865 | 7.02e-05 |
| U-Net | 0.8131 | 0.8969 | 0.8801 | 0.9144 | 0.0856 | 7.68e-05 |
| Classifier U-Net + ASTA-only | 0.7961 | 0.8865 | 0.8283 | 0.9534 | 0.0466 | 1.22e-04 |
| U-Net + ASTA-only | 0.7914 | 0.8836 | 0.8231 | 0.9537 | 0.0463 | 1.27e-04 |
| Classifier U-Net + ASTA gap-fill | 0.7869 | 0.8808 | 0.8185 | 0.9533 | 0.0467 | 1.30e-04 |
| U-Net + ASTA gap-fill | 0.7814 | 0.8773 | 0.8112 | 0.9551 | 0.0449 | 1.37e-04 |

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
└── results/                            Runtime outputs, ignored
```
