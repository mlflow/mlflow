from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Position, Range, lint_file
from clint.rules.missing_docstring_param import MissingDocstringParam


def test_missing_docstring_param(index: SymbolIndex) -> None:
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
    violations = lint_file(Path("test.py"), code, config, index)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MissingDocstringParam) for v in violations)
    assert violations[0].range == Range(Position(1, 0))


def test_missing_docstring_param_init(index: SymbolIndex) -> None:
    code = '''
class MyClass:
    def __init__(self, param1: str, param2: int) -> None:
        """
        Initialize MyClass.

        Args:
            param1: First parameter
        """
        pass

class GoodClass:
    def __init__(self, param1: str, param2: int) -> None:
        """
        Initialize GoodClass.

        Args:
            param1: First parameter
            param2: Second parameter
        """
        pass
'''
    config = Config(select={MissingDocstringParam.name})
    violations = lint_file(Path("test.py"), code, config, index)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MissingDocstringParam) for v in violations)
    assert violations[0].range == Range(Position(2, 4))


def test_missing_docstring_param_name_mangled(index: SymbolIndex) -> None:
    code = '''
class MyClass:
    def __private_helper(self, param1: str, param2: int) -> None:
        """
        Private name-mangled method (starts with __ but doesn't end with __).
        Should be skipped by clint.

        Args:
            param1: First parameter
        """
        pass

    def __init__(self, param1: str) -> None:
        """
        Initialize MyClass.

        Args:
            param1: First parameter
        """
        pass
'''
    config = Config(select={MissingDocstringParam.name})
    violations = lint_file(Path("test.py"), code, config, index)
    # Only __init__ should be checked, __private_helper should be skipped
    assert len(violations) == 0
