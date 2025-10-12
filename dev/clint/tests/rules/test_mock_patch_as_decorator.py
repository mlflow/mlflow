from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.mock_patch_as_decorator import MockPatchAsDecorator


def test_mock_patch_as_decorator_unittest_mock(index_path: Path) -> None:
    code = """
import unittest.mock

@unittest.mock.patch("foo.bar")
def test_foo(mock_bar):
    ...
"""
    config = Config(select={MockPatchAsDecorator.name})
    violations = lint_file(Path("test_mock_patch.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MockPatchAsDecorator) for v in violations)
    assert violations[0].loc == Location(3, 1)


def test_mock_patch_as_decorator_from_unittest_import_mock(index_path: Path) -> None:
    code = """
from unittest import mock

@mock.patch("foo.bar")
def test_foo(mock_bar):
    ...
"""
    config = Config(select={MockPatchAsDecorator.name})
    violations = lint_file(Path("test_mock_patch.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MockPatchAsDecorator) for v in violations)
    assert violations[0].loc == Location(3, 1)


def test_mock_patch_object_as_decorator(index_path: Path) -> None:
    code = """
from unittest import mock

@mock.patch.object(SomeClass, "method")
def test_foo(mock_method):
    ...
"""
    config = Config(select={MockPatchAsDecorator.name})
    violations = lint_file(Path("test_mock_patch.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MockPatchAsDecorator) for v in violations)
    assert violations[0].loc == Location(3, 1)


def test_mock_patch_dict_as_decorator(index_path: Path) -> None:
    code = """
from unittest import mock

@mock.patch.dict("os.environ", {"FOO": "bar"})
def test_foo():
    ...
"""
    config = Config(select={MockPatchAsDecorator.name})
    violations = lint_file(Path("test_mock_patch.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, MockPatchAsDecorator) for v in violations)
    assert violations[0].loc == Location(3, 1)


def test_mock_patch_as_context_manager_is_ok(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar") as mock_bar:
        ...
"""
    config = Config(select={MockPatchAsDecorator.name})
    violations = lint_file(Path("test_mock_patch.py"), code, config, index_path)
    assert len(violations) == 0


def test_non_test_file_not_checked(index_path: Path) -> None:
    code = """
from unittest import mock

@mock.patch("foo.bar")
def foo(mock_bar):
    ...
"""
    config = Config(select={MockPatchAsDecorator.name})
    violations = lint_file(Path("mock_patch.py"), code, config, index_path)
    assert len(violations) == 0


def test_multiple_patch_decorators(index_path: Path) -> None:
    code = """
from unittest import mock

@mock.patch("foo.bar")
@mock.patch("foo.baz")
def test_foo(mock_baz, mock_bar):
    ...
"""
    config = Config(select={MockPatchAsDecorator.name})
    violations = lint_file(Path("test_mock_patch.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, MockPatchAsDecorator) for v in violations)
    assert violations[0].loc == Location(3, 1)
    assert violations[1].loc == Location(4, 1)
