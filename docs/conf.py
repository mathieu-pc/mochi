import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.append(os.path.abspath("_ext"))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MOCHI'
copyright = '2026, Mathieu Perron-Cormier'
author = 'Mathieu Perron-Cormier'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Trigger sphinx-apidoc automatically before sphinx-build starts

def run_apidoc(app):
	from sphinx.ext.apidoc import main

	cur_dir = os.path.abspath(os.path.dirname(__file__))
	# Path to where your package code is located
	module_dir = os.path.join(cur_dir, "..", "src/Mochi")
	# Path where you want the .rst files saved
	output_dir = os.path.join(cur_dir, "api")

	# Options match command line: sphinx-apidoc -f -o <output> <module>
	cmd_args = ["-f", "-e", "-o", output_dir, module_dir, "--force"]
	main(cmd_args)

def rm_api(app, exception):
	if exception is not None:
		return
	if os.name == "nt":
		os.system("rd /s /q api")
	else:
		s.system("rm -r api")


def setup(app):
	app.connect("builder-inited", run_apidoc)
	app.connect("build-finished", rm_api)



intersphinx_mapping = {
	"python": ("https://docs.python.org/3", None),
	"numpy": ("https://numpy.org/doc/stable", None),
	"astropy": ("https://docs.astropy.org/en/stable", None),
	"martini": ("https://martini.readthedocs.io/en/stable", None),
	"radio_beam": ("https://radio-beam.readthedocs.io/en/stable/", None)
}


extensions = [
	"sphinx.ext.viewcode",
	"sphinx.ext.doctest",
	"sphinx.ext.napoleon",
	"sphinx.ext.mathjax",
	"sphinx.ext.autosummary",
	"sphinx.ext.intersphinx",
	"auto_parameter_description",
	"sphinx_copybutton",
]


#templates_path = ['_templates']
#exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = "sphinx_rtd_theme"
#htm_static_path = ["_static"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
