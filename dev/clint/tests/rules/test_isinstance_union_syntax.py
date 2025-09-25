from pathlib import Path

from clint.config import Config
from clint.linter import lint_file
from clint.rules import IsinstanceUnionSyntax


def test_isinstance_union_syntax(index_path: Path) -> None:
    code = """
# Should be flagged - basic union syntax
isinstance(obj, str | int)
isinstance(value, int | str | float)

# Should be flagged - parenthesized union in tuple
isinstance(x, ((str | int),))

# Should NOT be flagged - tuple syntax (recommended)
isinstance(obj, (str, int))
isinstance(value, (int, str, float))

# Should NOT be flagged - single type
isinstance(obj, str)
isinstance(obj, int)

# Should NOT be flagged - Union type annotation (different syntax)
isinstance(obj, Union[str, int])

# Should NOT be flagged - other functions with union syntax
other_func(obj, str | int)
some_call(x | y)

# Should NOT be flagged - isinstance with single argument (malformed but not our concern)
isinstance(obj)
"""
    config = Config(select={IsinstanceUnionSyntax.name})
    results = lint_file(Path("test.py"), code, config, index_path)

    # Should flag lines 3, 4, and 7 (isinstance with union syntax)
    assert len(results) == 3
    assert all(isinstance(r.rule, IsinstanceUnionSyntax) for r in results)

    # Check specific line numbers (1-indexed in code, 0-indexed in Location)
    expected_lines = [2, 3, 6]  # Lines 3, 4, 7 in 1-based indexing
    actual_lines = sorted(r.loc.lineno for r in results)
    assert actual_lines == expected_lines


def test_isinstance_union_syntax_nested_cases(index_path: Path) -> None:
    """Test more complex nested union cases"""
    code = """
# Complex nested union - should be flagged
isinstance(obj, (str | int) | float)
isinstance(obj, str | (int | float))

# Mixed valid and invalid - should be flagged
isinstance(obj, str | list)

# Valid nested tuples - should NOT be flagged
isinstance(obj, ((str, int), (float, bool)))
"""
    config = Config(select={IsinstanceUnionSyntax.name})
    results = lint_file(Path("test.py"), code, config, index_path)

    # Should flag lines 3, 4, and 7
    assert len(results) == 3
    assert all(isinstance(r.rule, IsinstanceUnionSyntax) for r in results)
