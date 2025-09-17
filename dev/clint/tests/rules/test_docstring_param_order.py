from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.docstring_param_order import DocstringParamOrder


def test_docstring_param_order(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    code = """
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
    tmp_file.write_text(code)

    config = Config(select={DocstringParamOrder.name})
    violations = lint_file(tmp_file, code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, DocstringParamOrder) for v in violations)
    assert violations[0].loc == Location(2, 0)
