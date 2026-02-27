from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import ExceptBoolOp


def test_except_bool_op(index_path: Path) -> None:
    code = """
# Bad - or in except
try:
    pass
except ValueError or KeyError:
    pass

# Bad - and in except
try:
    pass
except ValueError and KeyError:
    pass

# Bad - chained or
try:
    pass
except ValueError or KeyError or TypeError:
    pass

# Good - tuple syntax
try:
    pass
except (ValueError, KeyError):
    pass

# Good - single exception
try:
    pass
except ValueError:
    pass

# Good - bare except
try:
    pass
except Exception:
    pass
"""
    config = Config(select={ExceptBoolOp.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert all(isinstance(r.rule, ExceptBoolOp) for r in results)
    assert [r.range for r in results] == [
        Range(Position(4, 0)),
        Range(Position(10, 0)),
        Range(Position(16, 0)),
    ]
