# Python Style Guide

This guide documents Python coding conventions that go beyond what [ruff](https://docs.astral.sh/ruff/) and [clint](../../dev/clint/) can enforce. The practices below require human judgment to implement correctly and improve code readability, maintainability, and testability across the MLflow codebase.

## Avoid Redundant Test Docstrings

Omit docstrings that merely echo the function name without adding value. Test names should be self-documenting.

```python
# Bad
def test_foo():
    """Test foo"""
    ...


# Good
def test_foo():
    ...
```

## Use Type Hints for All Functions

Add type hints to all function parameters and return values. This enables better IDE support, catches bugs early, and serves as inline documentation.

```python
# Bad
def foo(s):
    return len(s)


# Good
def foo(s: str) -> int:
    return len(s)
```

## Minimize Try-Catch Block Scope

Wrap only the specific operations that can raise exceptions. Keep safe operations outside the try block to improve debugging and avoid masking unexpected errors.

```python
# Bad
try:
    never_fails()
    can_fail()
except ...:
    handle_error()

# Good
never_fails()
try:
    can_fail()
except ...:
    handle_error()
```

## Use Dataclasses Instead of Complex Tuples

Replace tuples with 3+ elements with named dataclasses. This improves code clarity, prevents positional argument errors, and enables type checking on individual fields.

```python
# Bad
def get_user() -> tuple[str, int, str]:
    return "Alice", 30, "Engineer"


# Good
from dataclasses import dataclass


@dataclass
class User:
    name: str
    age: int
    occupation: str


def get_user() -> User:
    return User(name="Alice", age=30, occupation="Engineer")
```

## Always Verify Mock Calls with Assertions

Every mocked function must have an assertion (`assert_called`, `assert_called_once`, etc.) to verify it was invoked correctly. Without assertions, tests may pass even when the mocked code isn't executed.

```python
from unittest import mock


# Bad
def test_foo():
    with mock.patch("foo.bar"):
        calls_bar()


# Good
def test_bar():
    with mock.patch("foo.bar") as mock_bar:
        calls_bar()
        mock_bar.assert_called_once()
```

## Set Mock Behaviors in Patch Declaration

Define `return_value` and `side_effect` directly in the `patch()` call rather than assigning them afterward. This keeps mock configuration explicit and reduces setup code.

```python
from unittest import mock


# Bad
def test_foo():
    with mock.patch("foo.bar") as mock_bar:
        mock_bar.return_value = 42
        calls_bar()

    with mock.patch("foo.bar") as mock_bar:
        mock_bar.side_effect = Exception("Error")
        calls_bar()


# Good
def test_foo():
    with mock.patch("foo.bar", return_value=42) as mock_bar:
        calls_bar()

    with mock.patch("foo.bar", side_effect=Exception("Error")) as mock_bar:
        calls_bar()
```

## Use Pytest's Monkeypatch for Directory Changes

Use `monkeypatch.chdir()` instead of manual `os.chdir()` with try/finally blocks. Pytest automatically restores the original directory after the test, preventing side effects.

```python
import os
import pytest


# Bad
def test_foo():
    cwd = os.getcwd()
    try:
        os.chdir("some/directory")
    finally:
        os.chdir(cwd)


# Good
def test_foo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir("some/directory")
```

## Parametrize Tests with Multiple Input Cases

Use `@pytest.mark.parametrize` to test multiple inputs instead of repeating assertions. This creates separate test cases for each input, making failures easier to diagnose and tests more maintainable.

```python
# Bad
def test_foo():
    assert foo("a") == 0
    assert foo("b") == 1
    assert foo("c") == 2


# Good
@pytest.mark.parametrize(
    ("input", "expected"),
    [
        ("a", 0),
        ("b", 1),
        ("c", 2),
    ],
)
def test_foo(input: str, expected: int):
    assert foo(input) == expected
```

## Use Pytest's Monkeypatch for Mocking Environment Variables

Use `monkeypatch.setenv()` instead of `mock.patch.dict()` for environment variables. Pytest's monkeypatch fixture automatically restores the original environment after the test, providing cleaner and more reliable test isolation.

```python
# Bad
def test_foo():
    with mock.patch.dict("os.environ", {"FOO": "True"}):
        ...


# Good
def test_foo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FOO", "True")
    ...
```

## Use Pytest's tmp_path Fixture for Temporary Files

Use `tmp_path` fixture instead of manual `tempfile.TemporaryDirectory()` for handling temporary files and directories in tests. Pytest automatically cleans up the temporary directory after the test, provides better test isolation, and integrates seamlessly with pytest's fixture system.

```python
# Bad
import tempfile


def test_foo():
    with tempfile.TemporaryDirectory() as tmpdir:
        ...


# Good
from pathlib import Path


def test_foo(tmp_path: Path):
    ...
```
