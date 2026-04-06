# Automated detection of satellite trails in astronomical images using deep learning

## Environment Setup

To run this pipeline locally or on a cluster, you need to set up a Python virtual environment and install the package in editable mode.

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
   Install the package in "editable" mode so that changes to the source code apply immediately without needing to reinstall.
   ```bash
   pip install -e .
   ```

#### Optional: Setting up for Jupyter Notebooks
If you want to use this virtual environment inside a Jupyter Notebook, you need to link the environment to your Jupyter kernels. Make sure your virtual environment is active, then run:

```bash
pip install ipykernel
python -m ipykernel install --user --name=satellite_env --display-name "Python (Satellite Trails)"
```
Now, when you open a notebook, you can select "Python (Satellite Trails)" as your kernel, and it will have access to your local package.