# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'REBA 3D'
copyright = '2025, Laboratoire Ergonomie'
author = 'Équipe REBA 3D'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Langue française
language = 'fr'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Logo et favicon (optionnel)
# html_logo = '_static/logo.png'
# html_favicon = '_static/favicon.ico'

# Options du thème Read the Docs
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
    'sticky_navigation': True,
}
