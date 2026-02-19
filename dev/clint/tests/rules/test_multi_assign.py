from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import MultiAssign


def test_multi_assign(index_path: Path) -> None:
    code = """
# Bad - non-constant values
x, y = func1(), func2()

# Good - unpacking from function
a, b = func()

# Good - all constants (allowed)
c, d = 1, 1
e, f, g = 0, 0, 0
h, i = "test", "test"
"""
    config = Config(select={MultiAssign.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 1
    assert all(isinstance(r.rule, MultiAssign) for r in results)
    assert results[0].range == Range(Position(2, 0))
