"""Конфигурация Sphinx для документации по коду.

Собирает HTML из docstring-ов всех модулей ядра. Сборка:

    py -m sphinx -b html docs docs/_build

Либо одной командой (включая просмотр результата в браузере):

    py docs/build_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Предиктор задержек наземного транспорта"
author = "Команда хакатона"
release = "1.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",       # Google/NumPy docstring
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",     # ссылки на numpy/pandas
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = []

# Каскад и батчинг содержат математику в docstring-ах -- reStructuredText ее понимает.
default_role = "py:obj"
nitpicky = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_mock_imports = ["catboost", "polars"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
