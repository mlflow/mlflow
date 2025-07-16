from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules import ImplicitOptional


def test_implicit_optional(index: SymbolIndex, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
from typing import Optional

# Bad
bad: int = None
class Bad:
    x: str = None

# Good
good: Optional[int] = None
class Good:
    x: Optional[str] = None
"""
    )
    config = Config(select={ImplicitOptional.name})
    results = lint_file(tmp_file, config, index)
    assert len(results) == 2
    assert all(isinstance(r.rule, ImplicitOptional) for r in results)
    assert results[0].loc == Location(4, 5)
    assert results[1].loc == Location(6, 7)
