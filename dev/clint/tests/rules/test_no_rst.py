from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules.no_rst import NoRst


def test_no_rst(index: SymbolIndex, config: Config, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
def bad(y: int) -> str:
    '''
    :param y: The parameter

    :returns: The result
    '''

def good(x: int) -> str:
    '''
    Args:
        x: The parameter.

    Returns:
        The result.
    '''
"""
    )

    violations = lint_file(tmp_file, config, index)
    assert len(violations) == 1
    assert all(isinstance(v.rule, NoRst) for v in violations)
    assert violations[0].loc == Location(2, 4)
