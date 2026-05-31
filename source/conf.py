# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MyNote'
copyright = '2024, jiangyao'
author = 'jiangyao'
release = 'release'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    'sphinx.ext.mathjax'
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'  # 设置语言为简体中文

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'analytics_anonymize_ip': False,
    'logo_only': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

# 注入在线 Mermaid 脚本
html_js_files = [
    'https://cloudflare.com'
]

# 允许在 Markdown 中无缝使用 mermaid 语法
myst_fence_as_directive = ["mermaid"]

