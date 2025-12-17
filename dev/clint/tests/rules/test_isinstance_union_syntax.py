from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules import IsinstanceUnionSyntax


def test_isinstance_union_syntax(index_path: Path) -> None:
    code = """
# Bad - basic union syntax
isinstance(obj, str | int)
isinstance(value, int | str | float)

# Bad - parenthesized union in tuple
isinstance(x, ((str | int),))

# Good - tuple syntax (recommended)
isinstance(obj, (str, int))
isinstance(value, (int, str, float))

# Good - single type
isinstance(obj, str)
isinstance(obj, int)

# Good - Union type annotation (different syntax)
isinstance(obj, Union[str, int])

# Good - other functions with union syntax
other_func(obj, str | int)
some_call(x | y)

# Good - invalid isinstance calls, not our concern
isinstance()
isinstance(obj)
"""
    config = Config(select={IsinstanceUnionSyntax.name})
    results = lint_file(Path("test.py"), code, config, index_path)
    assert all(isinstance(r.rule, IsinstanceUnionSyntax) for r in results)
    assert [r.range for r in results] == [
        Range(Position(2, 0)),
        Range(Position(3, 0)),
        Range(Position(6, 0)),
    ]
