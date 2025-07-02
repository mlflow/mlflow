from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules import MultiAssign


def test_multi_assign(index: SymbolIndex, config: Config, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
# Bad
x, y = 1, 2

# Good
a, b = func()
"""
    )
    results = lint_file(tmp_file, config, index)
    assert len(results) == 1
    assert all(isinstance(r.rule, MultiAssign) for r in results)
    assert results[0].loc == Location(2, 0)
