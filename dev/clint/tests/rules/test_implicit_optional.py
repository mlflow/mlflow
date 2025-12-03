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


def test_implicit_optional_stringified(index_path: Path) -> None:
    code = """
from typing import Optional

# Bad - stringified without Optional or None union
bad1: "int" = None
bad2: "str" = None
class Bad:
    x: "int" = None

# Good - stringified with Optional
good1: "Optional[int]" = None
good2: "Optional[str]" = None
class Good1:
    x: "Optional[str]" = None

# Good - stringified with | None
good3: "int | None" = None
good4: "str | None" = None
good5: "int|None" = None
class Good2:
    x: "SomeClass | None" = None
"""
    config = Config(select={ImplicitOptional.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert len(results) == 3
    assert all(isinstance(r.rule, ImplicitOptional) for r in results)
    assert results[0].range == Range(Position(4, 6))  # bad1
    assert results[1].range == Range(Position(5, 6))  # bad2
    assert results[2].range == Range(Position(7, 7))  # Bad.x
