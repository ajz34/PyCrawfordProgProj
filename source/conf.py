# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Programming Tutorial in Chemistry by Python'
copyright = '2020, ajz34'
author = 'ajz34'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinxcontrib.bibtex",
    "sphinx_thebe",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
thml_title = "Programming Tutorial in Chemistry by Python"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for HTML output -------------------------------------------------

# https://myst-nb.readthedocs.io/en/latest/use/execute.html
jupyter_execute_notebooks = "off"

thebe_config = {
    "repository_url": "https://github.com/ajz34/PyCrawfordProgProj",
    "repository_branch": "master",
}

html_theme_options = {
    "repository_url": "https://github.com/ajz34/PyCrawfordProgProj",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "thebe": True,
    },
    "path_to_docs": "source",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "expand_sections": ["reference/index"]
}
