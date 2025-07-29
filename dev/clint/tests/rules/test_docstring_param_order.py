from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules.docstring_param_order import DocstringParamOrder


def test_docstring_param_order(index: SymbolIndex, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
# Bad
def f(x: int, y: str) -> None:
    '''
    Args:
        y: Second param.
        x: First param.
    '''

# Good
def f(a: int, b: str) -> None:
    '''
    Args:
        a: First param.
        b: Second param.
    '''
"""
    )

    config = Config(select={DocstringParamOrder.name})
    violations = lint_file(tmp_file, config, index)
    assert len(violations) == 1
    assert all(isinstance(v.rule, DocstringParamOrder) for v in violations)
    assert violations[0].loc == Location(2, 0)
