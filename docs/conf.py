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
import importlib
import inspect
import os
import sys

from pallets_sphinx_themes import get_version

sys.path.insert(0, os.path.abspath(".."))
sys.path.append(os.path.abspath("./_ext"))


# -- Project information -----------------------------------------------------

project = "aclick"
copyright = "2022, Jonáš Kulhánek"
author = "Jonáš Kulhánek"
release, version = get_version("aclick")
if release == "develop":
    version = "develop"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinxcontrib.log_cabinet",
    "pallets_sphinx_themes",
    "sphinx_issues",
    "sphinx_tabs.tabs",
    "run_code",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "flask"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_sidebars = {
    '**': ['localtoc.html', 'toc.html', 'searchbox.html']
}


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        filename = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    module = obj.__module__.split(".")[0]
    filename = os.path.relpath(filename, os.path.abspath(".."))
    start, end = lines[1], lines[1] + len(lines[0]) - 1
    if release != "develop":
        baseurl = f"https://github.com/jkulhanek/{module}/blob/v{release}"
    else:
        baseurl = f"https://github.com/jkulhanek/{module}/tree/master"
    return f"{baseurl}/{filename}#L{start}-L{end}"


def setup(app):
    app.add_css_file("run_code.css")
