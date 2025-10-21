# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pylammpsmpi"
copyright = "2020, Jan Janssen, Sarath Menon"
author = "Jan Janssen, Sarath Menon"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]


# -- Generate API documentation ----------------------------------------------
# https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html

from sphinx.ext.apidoc import main

main(["-e", "-o", "apidoc", "../../src/pylammpsmpi/", "--force"])
