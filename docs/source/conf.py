"""Sphinx configuration for the satellite trail segmentation documentation."""

from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "docs" / "build" / "matplotlib"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


project = "Satellite Trail Segmentation"
author = "Andrew Pfaff"
copyright = "2026, Andrew Pfaff"
release = "0.1"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []


html_theme = "sphinx_rtd_theme"
html_title = "Satellite Trail Segmentation"
html_static_path = ["_static"]


autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
    "private-members": False,
}

add_module_names = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
