from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.no_rst import NoRst


def test_no_rst(index_path: Path) -> None:
    code = """
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
    config = Config(select={NoRst.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, NoRst) for v in violations)
    assert violations[0].loc == Location(2, 4)
