from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules import LazyBuiltinImport


def test_lazy_builtin_import(index_path: Path) -> None:
    code = """
def f():
    # Bad
    import sys
    import pandas as pd

# Good
import os
"""
    config = Config(select={LazyBuiltinImport.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, LazyBuiltinImport)
    assert results[0].loc == Location(3, 4)
