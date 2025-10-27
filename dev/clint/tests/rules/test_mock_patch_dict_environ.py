from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.mock_patch_dict_environ import MockPatchDictEnviron


def test_mock_patch_dict_environ_with_string_literal(index_path: Path) -> None:
    code = """
import os
from unittest import mock

# Bad - string literal
def test_func():
    with mock.patch.dict("os.environ", {"FOO": "True"}):
        pass
"""
    config = Config(select={MockPatchDictEnviron.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MockPatchDictEnviron) for v in violations)
    assert violations[0].loc == Location(6, 9)


def test_mock_patch_dict_environ_with_expression(index_path: Path) -> None:
    code = """
import os
from unittest import mock

# Bad - os.environ as expression
def test_func():
    with mock.patch.dict(os.environ, {"FOO": "bar"}):
        pass
"""
    config = Config(select={MockPatchDictEnviron.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MockPatchDictEnviron) for v in violations)
    assert violations[0].loc == Location(6, 9)


def test_mock_patch_dict_environ_as_decorator(index_path: Path) -> None:
    code = """
import os
from unittest import mock

# Bad - as decorator
@mock.patch.dict("os.environ", {"FOO": "value"})
def test_func():
    pass
"""
    config = Config(select={MockPatchDictEnviron.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MockPatchDictEnviron) for v in violations)
    assert violations[0].loc == Location(5, 1)


def test_mock_patch_dict_environ_with_clear(index_path: Path) -> None:
    code = """
import os
from unittest import mock

# Bad - with clear=True
def test_func():
    with mock.patch.dict(os.environ, {}, clear=True):
        pass
"""
    config = Config(select={MockPatchDictEnviron.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MockPatchDictEnviron) for v in violations)
    assert violations[0].loc == Location(6, 9)


def test_mock_patch_dict_non_environ(index_path: Path) -> None:
    code = """
from unittest import mock

# Good - not os.environ
def test_func():
    with mock.patch.dict("some.other.dict", {"key": "value"}):
        pass
"""
    config = Config(select={MockPatchDictEnviron.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_mock_patch_dict_environ_non_test_file(index_path: Path) -> None:
    code = """
import os
from unittest import mock

# Good - not in test file
def normal_func():
    with mock.patch.dict("os.environ", {"FOO": "True"}):
        pass
"""
    config = Config(select={MockPatchDictEnviron.name})
    violations = lint_file(Path("normal_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_mock_patch_dict_environ_with_mock_alias(index_path: Path) -> None:
    code = """
import os
from unittest import mock as mock_lib

# Bad - with alias
def test_func():
    with mock_lib.patch.dict("os.environ", {"FOO": "bar"}):
        pass
"""
    config = Config(select={MockPatchDictEnviron.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MockPatchDictEnviron) for v in violations)
    assert violations[0].loc == Location(6, 9)


def test_mock_patch_dict_environ_nested_function_not_caught(index_path: Path) -> None:
    code = """
import os
from unittest import mock

def test_outer():
    def inner_function():
        with mock.patch.dict("os.environ", {"FOO": "True"}):
            pass
    inner_function()
"""
    config = Config(select={MockPatchDictEnviron.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 0
