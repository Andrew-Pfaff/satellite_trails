# Automated detection of satellite trails in astronomical images using deep learning

## Environment Setup

To run this pipeline locally or on a cluster, set up a Python virtual environment and install the package in editable mode. The package requires Python 3.9+.

1. **Clone the repository:**
   ```bash
   git clone ...
   cd satellite_trails
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the package:**
   Install the package in editable mode so that changes to the source code apply immediately without needing to reinstall.
   ```bash
   pip install -e .
   ```

4. **Optional: install development dependencies:**
   This installs the extra tools declared in `pyproject.toml` for local analysis and notebooks.
   ```bash
   pip install -e .[dev]
   ```

#### Optional: Setting up for Jupyter Notebooks
If you want to use this virtual environment inside a Jupyter Notebook, link the environment to your Jupyter kernels. If you installed `.[dev]`, `jupyterlab` is already included. Make sure your virtual environment is active, then run:

```bash
pip install ipykernel
python -m ipykernel install --user --name=satellite_env --display-name "Python (Satellite Trails)"
```
Now, when you open a notebook, you can select "Python (Satellite Trails)" as your kernel, and it will have access to your local package.
