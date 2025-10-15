from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.nested_mock_patch import NestedMockPatch


def test_nested_mock_patch_unittest_mock(index_path: Path) -> None:
    code = """
import unittest.mock

def test_foo():
    with unittest.mock.patch("foo.bar"):
        with unittest.mock.patch("foo.baz"):
            ...
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, NestedMockPatch) for v in violations)
    assert violations[0].loc == Location(4, 4)


def test_nested_mock_patch_from_unittest_import_mock(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar"):
        with mock.patch("foo.baz"):
            ...
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, NestedMockPatch) for v in violations)
    assert violations[0].loc == Location(4, 4)


def test_nested_mock_patch_object(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch.object(SomeClass, "method"):
        with mock.patch.object(AnotherClass, "method"):
            ...
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, NestedMockPatch) for v in violations)
    assert violations[0].loc == Location(4, 4)


def test_nested_mock_patch_dict(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch.dict("os.environ", {"FOO": "bar"}):
        with mock.patch.dict("os.environ", {"BAZ": "qux"}):
            ...
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, NestedMockPatch) for v in violations)
    assert violations[0].loc == Location(4, 4)


def test_nested_mock_patch_mixed(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar"):
        with mock.patch.object(SomeClass, "method"):
            ...
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, NestedMockPatch) for v in violations)
    assert violations[0].loc == Location(4, 4)


def test_multiple_context_managers_is_ok(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar"), mock.patch("foo.baz"):
        ...
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 0


def test_multiple_context_managers_with_object_is_ok(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar"), mock.patch.object(SomeClass, "method"):
        ...
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 0


def test_nested_with_but_not_mock_patch_is_ok(index_path: Path) -> None:
    code = """
def test_foo():
    with open("file.txt"):
        with open("file2.txt"):
            ...
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 0


def test_nested_with_only_one_mock_patch_is_ok(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar"):
        with open("file.txt"):
            ...
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 0


def test_non_nested_mock_patches_are_ok(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar"):
        pass
    with mock.patch("foo.baz"):
        pass
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 0


def test_non_test_file_not_checked(index_path: Path) -> None:
    code = """
from unittest import mock

def foo():
    with mock.patch("foo.bar"):
        with mock.patch("foo.baz"):
            ...
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 0


def test_nested_with_code_after_is_ok(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar"):
        with mock.patch("foo.baz"):
            ...

        assert True
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    assert len(violations) == 0


def test_deeply_nested_mock_patch(index_path: Path) -> None:
    code = """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar"):
        with mock.patch("foo.baz"):
            with mock.patch("foo.qux"):
                ...
"""
    config = Config(select={NestedMockPatch.name})
    violations = lint_file(Path("test_nested_mock_patch.py"), code, config, index_path)
    # Should detect both levels of nesting
    assert len(violations) == 2
    assert all(isinstance(v.rule, NestedMockPatch) for v in violations)
    assert violations[0].loc == Location(4, 4)
    assert violations[1].loc == Location(5, 8)
