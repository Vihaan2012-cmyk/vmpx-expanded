project = "vmpx"
copyright = "2026, vmpx contributors"
author = "vmpx contributors"
release = "0.1.0"
version = "0.1"

extensions = []

templates_path = []
exclude_patterns = ["_build", "extra"]
master_doc = "index"
source_suffix = ".rst"
language = "en"

pygments_style = "sphinx"
highlight_language = "none"

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = []
html_extra_path = ["extra"]
html_title = "vmpx documentation"
html_short_title = "vmpx"
html_show_sphinx = False
htmlhelp_basename = "vmpxdoc"
