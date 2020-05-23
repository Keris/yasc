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
import sphinx_bootstrap_theme
import matplotlib as mpl
from subprocess import check_call as sh
mpl.use("Agg")
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = u'yasc'
copyright = '2020, Liqiang Du'
author = 'Liqiang Du'

# The short X.Y version.
sys.path.insert(0, os.path.abspath(os.path.pardir))
import yasc
version = yasc.__version__
# The full version, including alpha/beta/rc tags.
release = yasc.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_bootstrap_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
]

# Include the example source for plots in API docs
plot_include_source = True
# plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False

# The master toctree document.
master_doc = 'index'

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'source_link_position': "footer",
    'bootswatch_theme': "paper",
    'navbar_sidebarrel': False,
    'bootstrap_version': "3",
    'navbar_links': [
                    #  ("Gallery", "examples/index"),
                     ("Tutorial", "tutorial"),
                     ("API", "api"),
                     ],

    }

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# Add the 'copybutton' javascript, to hide/show the prompt in code
# examples, originally taken from scikit-learn's doc/conf.py
def setup(app):
    app.add_js_file('copybutton.js')
    app.add_css_file('style.css')

    # Build tutorials
    sh(["make -C tutorial"], shell=True)