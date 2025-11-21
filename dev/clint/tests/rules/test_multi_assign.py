from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import MultiAssign


def test_multi_assign(index_path: Path) -> None:
    code = """
# Bad
x, y = 1, 2

# Good
a, b = func()
"""
    config = Config(select={MultiAssign.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert all(isinstance(r.rule, MultiAssign) for r in results)
    assert results[0].range == Range(Position(2, 0))
