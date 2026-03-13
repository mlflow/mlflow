from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import LazyImport


def test_lazy_import(index_path: Path) -> None:
    code = """
def f():
    # Bad
    import sys
    import pandas as pd

# Good
import os
"""
    config = Config(select={LazyImport.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert isinstance(results[0].rule, LazyImport)
    assert results[0].range == Range(Position(3, 4))
