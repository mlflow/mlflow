from pathlib import Path

from clint.config import Config
from clint.linter import lint_file
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
    assert (violations[0].loc.lineno, violations[0].loc.col_offset) == (2, 0)
    assert (violations[1].loc.lineno, violations[1].loc.col_offset) == (3, 0)
