from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import ImplicitOptional


def test_implicit_optional(index_path: Path) -> None:
    code = """
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
    config = Config(select={ImplicitOptional.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 2
    assert all(isinstance(r.rule, ImplicitOptional) for r in results)
    assert results[0].range == Range(Position(4, 5))
    assert results[1].range == Range(Position(6, 7))
