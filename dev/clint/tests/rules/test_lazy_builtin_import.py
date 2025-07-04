from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules import LazyBuiltinImport


def test_lazy_builtin_import(index: SymbolIndex, config: Config, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
def f():
    # Bad
    import sys
    import pandas as pd

# Good
import os
"""
    )
    results = lint_file(tmp_file, config, index)
    assert len(results) == 1
    assert isinstance(results[0].rule, LazyBuiltinImport)
    assert results[0].loc == Location(3, 4)
