from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
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
    assert violations[0].range == Range(Position(2, 0))
    assert violations[1].range == Range(Position(3, 0))


def test_nested_if_in_type_checking_block(index_path: Path) -> None:
    code = """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    if True:
        pass
    import databricks  # Should NOT be flagged
    from databricks import foo  # Should NOT be flagged
"""
    config = Config(
        select={ForbiddenTopLevelImport.name},
        forbidden_top_level_imports={"*": ["databricks"]},
    )
    violations = lint_file(Path("test.py"), code, config, index_path)
    # Should have no violations since imports are inside TYPE_CHECKING
    assert len(violations) == 0
