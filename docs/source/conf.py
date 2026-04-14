# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# import sphinx_book_theme

# -- Project information -----------------------------------------------------

project = "tiamat"
copyright = "2025, Forschungszentrum Juelich GmbH"
author = "Big Data Analytics Group, Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Juelich GmbH"
language = "en"

# -- General configuration ---------------------------------------------------

source_suffix = [".rst", ".md"]

# The master toctree document.
root_doc = "index"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/legacy"]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_design",
    "sphinx_copybutton",  # adds a copy button for code fields
    "sphinx.ext.graphviz",  # to allow drawing diagrams
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_show_sourcelink = False
html_show_sphinx = True
html_permalinks = False
html_favicon = "_static/images/favicon.ico"
html_title = ""
html_scaled_image_link = False

html_theme_options = {
    "logo": {
        "image_light": "_static/images/logo-lightmode.png",
        "image_dark": "_static/images/logo-darkmode.png",
    },
    "repository_url": "https://jugit.fz-juelich.de/inm-1/bda/software/data_access/tiamat",
    "repository_provider": "gitlab",
    "use_repository_button": True,
    "use_download_button": False,
    "use_fullscreen_button": False,
}
