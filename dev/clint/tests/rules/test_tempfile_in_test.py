from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules.tempfile_in_test import TempfileInTest


def test_tempfile_in_test_temporary_directory(index_path: Path) -> None:
    code = """
import tempfile

# Bad
def test_func():
    tempfile.TemporaryDirectory()

# Good
def non_test_func():
    tempfile.TemporaryDirectory()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TempfileInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))


def test_tempfile_in_test_named_temporary_file(index_path: Path) -> None:
    code = """
import tempfile

# Bad
def test_func():
    tempfile.NamedTemporaryFile()

# Good
def non_test_func():
    tempfile.NamedTemporaryFile()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TempfileInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))


def test_tempfile_in_test_temporary_file(index_path: Path) -> None:
    code = """
import tempfile

# Bad
def test_func():
    tempfile.TemporaryFile()

# Good
def non_test_func():
    tempfile.TemporaryFile()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TempfileInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))


def test_tempfile_in_test_mkdtemp(index_path: Path) -> None:
    code = """
import tempfile

# Bad
def test_func():
    tempfile.mkdtemp()

# Good
def non_test_func():
    tempfile.mkdtemp()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TempfileInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))


def test_tempfile_in_test_with_from_import_temporary_directory(index_path: Path) -> None:
    code = """
from tempfile import TemporaryDirectory

# Bad
def test_func():
    TemporaryDirectory()

# Good
def non_test_func():
    TemporaryDirectory()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TempfileInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))


def test_tempfile_in_test_with_from_import_named_temporary_file(index_path: Path) -> None:
    code = """
from tempfile import NamedTemporaryFile

# Bad
def test_func():
    NamedTemporaryFile()

# Good
def non_test_func():
    NamedTemporaryFile()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TempfileInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))


def test_tempfile_in_test_with_from_import_temporary_file(index_path: Path) -> None:
    code = """
from tempfile import TemporaryFile

# Bad
def test_func():
    TemporaryFile()

# Good
def non_test_func():
    TemporaryFile()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TempfileInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))


def test_tempfile_in_test_with_from_import_mkdtemp(index_path: Path) -> None:
    code = """
from tempfile import mkdtemp

# Bad
def test_func():
    mkdtemp()

# Good
def non_test_func():
    mkdtemp()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TempfileInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))


def test_tempfile_in_test_no_violation_outside_test(index_path: Path) -> None:
    code = """
import tempfile

def normal_function():
    tempfile.TemporaryDirectory()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("non_test_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_tempfile_in_test_with_alias(index_path: Path) -> None:
    code = """
import tempfile as tf

# Bad - should still catch aliased import
def test_func():
    tf.TemporaryDirectory()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TempfileInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))


def test_tempfile_in_test_nested_functions_not_caught(index_path: Path) -> None:
    """
    Nested functions are not considered to be "in test" - this matches
    the behavior of other test-specific rules like os.environ.
    """
    code = """
import tempfile

def test_outer():
    def inner_function():
        tempfile.TemporaryDirectory()  # Not caught since inner_function is not a test function
    inner_function()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_tempfile_not_tempfile_module(index_path: Path) -> None:
    code = """
class FakeTempfile:
    @staticmethod
    def TemporaryDirectory():
        pass

fake_tempfile = FakeTempfile()

def test_func():
    # Should not trigger since it's not tempfile.TemporaryDirectory
    fake_tempfile.TemporaryDirectory()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_tempfile_in_test_with_context_manager(index_path: Path) -> None:
    code = """
import tempfile

# Bad - using with statement
def test_func():
    with tempfile.TemporaryDirectory() as tmpdir:
        pass
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TempfileInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 9))


def test_tempfile_in_test_assigned_to_variable(index_path: Path) -> None:
    code = """
import tempfile

# Bad - assigned to variable
def test_func():
    tmpdir = tempfile.TemporaryDirectory()
"""
    config = Config(select={TempfileInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, TempfileInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 13))
