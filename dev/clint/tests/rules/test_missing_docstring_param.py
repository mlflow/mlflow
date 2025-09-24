from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.missing_docstring_param import MissingDocstringParam


def test_missing_docstring_param(index_path: Path) -> None:
    code = '''
def bad_function(param1: str, param2: int, param3: bool) -> None:
    """
    Example function with missing parameters in docstring.

    Args:
        param1: First parameter described
    """

def good_function(param1: str, param2: int) -> None:
    """
    Good function with all parameters documented.

    Args:
        param1: First parameter
        param2: Second parameter
    """
'''
    config = Config(select={MissingDocstringParam.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MissingDocstringParam) for v in violations)
    assert violations[0].loc == Location(1, 0)
