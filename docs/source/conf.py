# Configuration file for the Sphinx documentation builder.

# -- Source code path --
import os, sys, subprocess
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information --
project = 'Satellite Trail Segmentation'
copyright = '2026, Andrew Pfaff'
author = 'Andrew Pfaff'
release = 'v0.1'


# -- General configuration --
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output --
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# -- Auto-generate API docs --
def run_apidoc(_):
    cur_dir = os.path.dirname(__file__)
    lib_dir = os.path.abspath(os.path.join(cur_dir, '../../src/satellite_trail_segmentation'))
    out_dir = os.path.join(cur_dir, 'api')
    subprocess.run([
        'sphinx-apidoc',
        '--force',
        '--module-first',
        '-o', out_dir,
        lib_dir,
    ])

def setup(app):
    app.connect('builder-inited', run_apidoc)



autodoc_mock_imports = [
    "torch",
    "torchvision",
    "h5py",
    "optuna",
    "PIL",
    "cv2",
    "numpy",
    "sklearn",
]


html_title = "Satellite Trail Segmentation"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
add_module_names = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
    "private-members": False,
}


import re
import os

def fix_rst_titles(app):
    cur_dir = os.path.dirname(__file__)
    api_dir = os.path.join(cur_dir, 'api')
    
    # Ensure the directory exists before trying to iterate through it
    if not os.path.exists(api_dir):
        return

    for rst_file in os.listdir(api_dir):
        if not rst_file.endswith('.rst'):
            continue
        path = os.path.join(api_dir, rst_file)
        with open(path, 'r') as f:
            content = f.read()
        
        # 1. Sphinx escapes underscores in headings. We must un-escape them first.
        # Example: 'satellite\_trail\_segmentation' becomes 'satellite_trail_segmentation'
        content = content.replace(r'\_', '_')
        
        # 2. Shorten sub-package and module titles
        # Example: 'satellite_trail_segmentation.classifier_model.classifier module' -> 'classifier'
        content = re.sub(
            r'^satellite_trail_segmentation(?:[.a-zA-Z0-9_]*\.)?([a-zA-Z0-9_]+) (module|package)$',
            r'\1',
            content,
            flags=re.MULTILINE
        )
        
        # 3. Handle the root package title edge-case
        # Example: 'satellite_trail_segmentation package' -> 'satellite_trail_segmentation'
        content = re.sub(
            r'^satellite_trail_segmentation package$',
            r'satellite_trail_segmentation',
            content,
            flags=re.MULTILINE
        )
        
        with open(path, 'w') as f:
            f.write(content)

# Consolidate your setup functions into this single one
def setup(app):
    app.connect('builder-inited', run_apidoc)
    app.connect('builder-inited', fix_rst_titles)