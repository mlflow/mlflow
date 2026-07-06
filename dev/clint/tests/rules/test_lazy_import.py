from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Position, Range, lint_file
from clint.rules import LazyImport


def test_lazy_import(index: SymbolIndex) -> None:
    code = """
def f():
    # Bad
    import sys
    import pandas as pd

# Good
import os
"""
    config = Config(select={LazyImport.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 1
    assert isinstance(results[0].rule, LazyImport)
    assert results[0].range == Range(Position(3, 4))


def test_lazy_import_third_party(index: SymbolIndex) -> None:
    code = """
def f():
    # Bad - always-available third-party packages
    import pydantic
    from packaging.version import Version
    from cachetools import TTLCache

# Good - top-level imports
import pydantic
from packaging.version import Version
from cachetools import TTLCache

def g():
    # Good - databricks is not in the always-available set
    import databricks
"""
    config = Config(select={LazyImport.name})
    results = lint_file(Path("test.py"), code, config, index)
    assert len(results) == 3
    assert all(isinstance(r.rule, LazyImport) for r in results)
    assert results[0].range == Range(Position(3, 4))
    assert results[1].range == Range(Position(4, 4))
    assert results[2].range == Range(Position(5, 4))
