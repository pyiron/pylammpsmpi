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
import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../../pylammpsmpi/'))

# -- Project information -----------------------------------------------------

project = 'pylammpsmpi'
copyright = '2020, Jan Janssen, Sarath Menon'
author = 'Jan Janssen, Sarath Menon'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['../_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_logo = "../_static/pyiron_logo.png"
html_theme_options = {
    'logo_only' : True,
    'canonical_url' : 'https://pylammpsmpi.readthedocs.io/',
}

html_extra_path = ['../_static']

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'
pygments_style = None

html_static_path = ['../_static']
def setup(app):
    app.add_stylesheet("theme_extra.css")

latex_documents = [
    (master_doc, 'pylammpsmpi.tex', u'pylammpsmpi Documentation',
     u'Sarath Menon, Jan Janssen', 'manual'),
]

man_pages = [
    (master_doc, 'pylammpsmpi', u'pylammpsmpi Documentation',
     [author], 1)
]
