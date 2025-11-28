from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules.duplicate_import import DuplicateImport


def test_duplicate_import_from_style(index_path: Path) -> None:
    code = """
from foo import bar


def f():
    from foo import bar

    print(bar)
"""
    config = Config(select={DuplicateImport.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, DuplicateImport) for v in violations)
    assert violations[0].range == Range(Position(5, 4))
    rule = violations[0].rule
    assert isinstance(rule, DuplicateImport)
    assert rule.import_name == "foo.bar"


def test_duplicate_import_import_style(index_path: Path) -> None:
    code = """
import os


def f():
    import os

    print(os.getcwd())
"""
    config = Config(select={DuplicateImport.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, DuplicateImport) for v in violations)
    assert violations[0].range == Range(Position(5, 4))
    rule = violations[0].rule
    assert isinstance(rule, DuplicateImport)
    assert rule.import_name == "os"


def test_duplicate_import_nested_module(index_path: Path) -> None:
    code = """
from mlflow.tracking import log_metric


def f():
    from mlflow.tracking import log_metric

    log_metric("key", 1)
"""
    config = Config(select={DuplicateImport.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, DuplicateImport) for v in violations)
    assert violations[0].range == Range(Position(5, 4))
    rule = violations[0].rule
    assert isinstance(rule, DuplicateImport)
    assert rule.import_name == "mlflow.tracking.log_metric"


def test_no_duplicate_import_different_names(index_path: Path) -> None:
    code = """
from foo import bar


def f():
    from foo import baz

    print(bar, baz)
"""
    config = Config(select={DuplicateImport.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 0


def test_no_duplicate_import_different_modules(index_path: Path) -> None:
    code = """
from foo import bar


def f():
    from qux import bar

    print(bar)
"""
    config = Config(select={DuplicateImport.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 0


def test_no_duplicate_import_only_top_level(index_path: Path) -> None:
    code = """
from foo import bar

print(bar)
"""
    config = Config(select={DuplicateImport.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 0


def test_duplicate_import_multiple_functions(index_path: Path) -> None:
    code = """
from foo import bar


def f1():
    from foo import bar

    print(bar)


def f2():
    from foo import bar

    print(bar)
"""
    config = Config(select={DuplicateImport.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, DuplicateImport) for v in violations)
    assert violations[0].range == Range(Position(5, 4))
    assert violations[1].range == Range(Position(11, 4))
    for v in violations:
        rule = v.rule
        assert isinstance(rule, DuplicateImport)
        assert rule.import_name == "foo.bar"


def test_no_duplicate_import_in_type_checking(index_path: Path) -> None:
    code = """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foo import bar


def f():
    from foo import bar

    print(bar)
"""
    config = Config(select={DuplicateImport.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    # Should not flag as duplicate because top-level import is inside TYPE_CHECKING
    assert len(violations) == 0
