from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.extraneous_docstring_param import ExtraneousDocstringParam


def test_extraneous_docstring_param(index_path: Path) -> None:
    code = '''
def bad_function(param1: str) -> None:
    """
    Example function docstring.

    Args:
        param1: First parameter
        param2: This parameter doesn't exist in function signature
        param3: Another non-existent parameter
    """

def good_function(param1: str, param2: int) -> None:
    """
    Good function with matching parameters.

    Args:
        param1: First parameter
        param2: Second parameter
    """
'''
    config = Config(select={ExtraneousDocstringParam.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, ExtraneousDocstringParam) for v in violations)
    assert violations[0].loc == Location(1, 0)
