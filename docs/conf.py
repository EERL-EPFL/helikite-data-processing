import os
import sys

sys.path.insert(0, os.path.abspath("../helikite"))  # Fix module path

project = "helikite-data-processing"
copyright = "2025, Evan Thomas"
author = "Evan Thomas"
release = "1.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "autoapi.extension",
    # "sphinxcontrib.autodoc_pydantic",
]

html_theme = "sphinx_rtd_theme"

# AutoAPI Configuration
autoapi_type = "python"
autoapi_dirs = ["../helikite"]
autoapi_ignore = [
    "**/helikite.py",
    "**/tests/*.py",
    "**/test_*.py",
    "**/__pycache__",
    "**/conftest.py",
    "**/*checkpoint*",
    "helikite/instruments/__init__.py",
]

autodoc_pydantic_model_show_json = True
autodoc_pydantic_settings_show_json = False
