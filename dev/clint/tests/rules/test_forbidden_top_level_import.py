from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.forbidden_top_level_import import ForbiddenTopLevelImport


def test_forbidden_top_level_import(index_path: Path) -> None:
    code = """
# Bad
import foo
from foo import bar

# Good
import baz
"""
    config = Config(
        select={ForbiddenTopLevelImport.name},
        forbidden_top_level_imports={"*": ["foo"]},
    )
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, ForbiddenTopLevelImport) for v in violations)
    assert violations[0].loc == Location(2, 0)
    assert violations[1].loc == Location(3, 0)
