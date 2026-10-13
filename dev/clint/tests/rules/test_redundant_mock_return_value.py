from pathlib import Path

import pytest
from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import lint_file
from clint.rules.redundant_mock_return_value import RedundantMockReturnValue

CONFIG = Config(select={RedundantMockReturnValue.name})
TEST_FILE = Path("test_foo.py")


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar", return_value=mock.MagicMock()):
        ...
""",
            id="patch-return_value-MagicMock",
        ),
        pytest.param(
            """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar", return_value=mock.Mock()):
        ...
""",
            id="patch-return_value-Mock",
        ),
        pytest.param(
            """
from unittest import mock

class Foo:
    pass

def test_foo():
    with mock.patch.object(Foo, "method", return_value=mock.MagicMock()):
        ...
""",
            id="patch.object-return_value-MagicMock",
        ),
        pytest.param(
            """
import unittest.mock

def test_foo():
    with unittest.mock.patch("foo.bar", return_value=unittest.mock.MagicMock()):
        ...
""",
            id="unittest.mock-import-style",
        ),
    ],
)
def test_violation(code: str, index: SymbolIndex) -> None:
    violations = lint_file(TEST_FILE, code, CONFIG, index)
    assert len(violations) == 1
    assert isinstance(violations[0].rule, RedundantMockReturnValue)


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar", return_value=42):
        ...
""",
            id="meaningful-return_value",
        ),
        pytest.param(
            """
from unittest import mock

class Foo:
    pass

def test_foo():
    with mock.patch("foo.bar", return_value=mock.MagicMock(spec=Foo)):
        ...
""",
            id="MagicMock-with-args",
        ),
        pytest.param(
            """
from unittest import mock

def test_foo():
    with mock.patch("foo.bar"):
        ...
""",
            id="no-return_value",
        ),
    ],
)
def test_no_violation(code: str, index: SymbolIndex) -> None:
    violations = lint_file(TEST_FILE, code, CONFIG, index)
    assert len(violations) == 0


def test_non_test_file_not_checked(index: SymbolIndex) -> None:
    code = """
from unittest import mock

def foo():
    with mock.patch("foo.bar", return_value=mock.MagicMock()):
        ...
"""
    violations = lint_file(Path("foo.py"), code, CONFIG, index)
    assert len(violations) == 0
