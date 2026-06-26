# Satellite Trail Segmentation

This repository contains a reproducibility study of satellite trail detection in astronomical images. It implements: patch preprocessing, a patch classifier, baseline U-Net segmentation, Attention U-Net segmentation, and Hough-style postprocessing.


## Documentation

The Sphinx documentation is the main user and API documentation surface.

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

The docs include quickstart usage, data notes, training, parameter search, evaluation, postprocessing, CSD3 workflow, limitations, and API documentation.


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
├── results/                            Runtime outputs, ignored
└── reca_streaks_eval/                  Optional external RECA/satmetrics evaluation tooling
```
