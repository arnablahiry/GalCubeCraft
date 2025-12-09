# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = ''
copyright = '2025, Arnab Lahiry'
author = 'Arnab Lahiry'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))  # points to GalCubeCraft/src

# Path for static files
html_static_path = ['_static']

# Logo in the top left of the sidebar
html_logo = '_static/logo.png'

# Optional: favicon
html_favicon = '_static/favicon.png'


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # for Google/Numpy-style docstrings
    'sphinx.ext.viewcode',  # add links to highlighted source code
    'myst_parser',
    'sphinx.ext.mathjax',
]


myst_enable_extensions = [
    "amsmath",    # support for display math
    "dollarmath", # support for $...$ inline and $$...$$ display math
]

intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


html_theme = 'sphinx_book_theme'