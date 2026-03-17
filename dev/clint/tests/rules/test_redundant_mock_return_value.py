from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules.redundant_mock_return_value import RedundantMockReturnValue


def test_patch_with_magic_mock_return_value(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar", return_value=mock.MagicMock()):
        ...
"""
    config = Config(select={RedundantMockReturnValue.name})
    violations = lint_file(Path("test_foo.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, RedundantMockReturnValue) for v in violations)
    assert violations[0].range == Range(Position(4, 9))


def test_patch_with_mock_return_value(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar", return_value=mock.Mock()):
        ...
"""
    config = Config(select={RedundantMockReturnValue.name})
    violations = lint_file(Path("test_foo.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, RedundantMockReturnValue) for v in violations)


def test_patch_object_with_magic_mock_return_value(index_path: Path) -> None:
    code = """
from unittest import mock

class Foo:
    pass

def test_foo():
    with mock.patch.object(Foo, "method", return_value=mock.MagicMock()):
        ...
"""
    config = Config(select={RedundantMockReturnValue.name})
    violations = lint_file(Path("test_foo.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, RedundantMockReturnValue) for v in violations)


def test_patch_with_meaningful_return_value_is_ok(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar", return_value=42):
        ...
"""
    config = Config(select={RedundantMockReturnValue.name})
    violations = lint_file(Path("test_foo.py"), code, config, index_path)
    assert len(violations) == 0


def test_patch_with_magic_mock_with_args_is_ok(index_path: Path) -> None:
    code = """
from unittest import mock

class Foo:
    pass

def test_foo():
    with mock.patch("foo.bar", return_value=mock.MagicMock(spec=Foo)):
        ...
"""
    config = Config(select={RedundantMockReturnValue.name})
    violations = lint_file(Path("test_foo.py"), code, config, index_path)
    assert len(violations) == 0


def test_patch_without_return_value_is_ok(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar"):
        ...
"""
    config = Config(select={RedundantMockReturnValue.name})
    violations = lint_file(Path("test_foo.py"), code, config, index_path)
    assert len(violations) == 0


def test_non_test_file_not_checked(index_path: Path) -> None:
    code = """
from unittest import mock

def foo():
    with mock.patch("foo.bar", return_value=mock.MagicMock()):
        ...
"""
    config = Config(select={RedundantMockReturnValue.name})
    violations = lint_file(Path("foo.py"), code, config, index_path)
    assert len(violations) == 0


def test_unittest_mock_import_style(index_path: Path) -> None:
    code = """
import unittest.mock

def test_foo():
    with unittest.mock.patch("foo.bar", return_value=unittest.mock.MagicMock()):
        ...
"""
    config = Config(select={RedundantMockReturnValue.name})
    violations = lint_file(Path("test_foo.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, RedundantMockReturnValue) for v in violations)
