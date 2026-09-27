"""Сборка HTML-документации по коду ядра.

Запуск:
    py docs/build_docs.py
Результат:
    docs/_build/index.html
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "_build"


def main() -> int:
    try:
        import sphinx  # noqa: F401
    except ImportError:
        print("Sphinx не установлен. Установите: py -m pip install sphinx", flush=True)
        return 1
    if BUILD.exists():
        shutil.rmtree(BUILD)
    command = [sys.executable, "-m", "sphinx", "-b", "html", str(HERE), str(BUILD)]
    print("сборка: " + " ".join(command), flush=True)
    code = subprocess.call(command)
    index = BUILD / "index.html"
    if code == 0 and index.exists():
        print(f"готово: {index}", flush=True)
    elif code == 0:
        print(f"сборка завершена, но {index} не найден", flush=True)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
