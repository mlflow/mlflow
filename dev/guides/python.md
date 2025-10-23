# Python Style Guide

This guide documents Python coding conventions that go beyond what [ruff](https://docs.astral.sh/ruff/) and [clint](../../dev/clint/) can enforce. The practices below require human judgment to implement correctly and improve code readability, maintainability, and testability across the MLflow codebase.

## Avoid Redundant Docstrings

Omit docstrings that merely repeat the function name or provide no additional value. Function names should be self-documenting.

```python
# Bad
def calculate_sum(a: int, b: int) -> int:
    """Calculate sum"""
    return a + b


# Good
def calculate_sum(a: int, b: int) -> int:
    return a + b
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

### Exceptions

**Test functions:** The `-> None` return type can be omitted for test functions since they implicitly return `None` and the return value is not used. However, **parameter type hints are still required** for all test function parameters.

```python
# Good - has parameter type hints but no return type hint
def test_foo(s: str):
    ...


# Good - has both parameter and return type hints
def test_foo(s: str) -> None:
    ...


# Bad - missing parameter type hints
def test_foo(s):
    ...
```

**`__init__` methods:** The `-> None` return type can be omitted for `__init__` methods since they always return `None` by definition.

```python
# Acceptable
class Foo:
    def __init__(self, s: str):
        ...


# Also acceptable (but not required)
class Foo:
    def __init__(self, s: str) -> None:
        ...
```

### Prefer `typing.Literal` for Fixed-String Parameters

When a parameter only accepts a fixed set of string values, use `typing.Literal` instead of a plain `str` type hint. This improves type-checking, enables IDE autocompletion, and documents allowed values at the type level.

```python
# Bad
def f(app: str) -> None:
    """
    Args:
        app: Application type. Either "fastapi" or "flask".
    """
    ...


# Good
from typing import Literal


def f(app: Literal["fastapi", "flask"]) -> None:
    """
    Args:
        app: Application type. Either "fastapi" or "flask".
    """
    ...
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

## Use `pathlib` Methods Instead of `os` Module Functions

When you have a `pathlib.Path` object, use its built-in methods instead of `os` module functions. This is more readable, type-safe, and follows object-oriented principles.

```python
from pathlib import Path

path = Path("some/file.txt")

# Bad
import os

os.path.exists(path)
os.remove(path)

# Good
path.exists()
path.unlink()
```

## Pass `pathlib.Path` Objects Directly to `subprocess`

Avoid converting `pathlib.Path` objects to strings when passing them to `subprocess` functions. Modern Python (3.8+) accepts Path objects directly, making the code cleaner and more type-safe.

```python
import subprocess
from pathlib import Path

path = Path("some/script.py")

# Bad
subprocess.check_call(["foo", "bar", str(path)])

# Good
subprocess.check_call(["foo", "bar", path])
```

## Use next() to Find First Match Instead of Loop-and-Break

Use the `next()` builtin function with a generator expression to find the first item that matches a condition. This is more concise and functional than manually looping with break statements.

```python
# Bad
result = None
for item in items:
    if item.name == "target":
        result = item
        break

# Good
result = next((item for item in items if item.name == "target"), None)
```

## Use Pattern Matching for String Splitting

When splitting strings into a fixed number of parts, use pattern matching instead of direct unpacking or verbose length checks. Pattern matching provides concise, safe extraction that clearly handles both expected and unexpected cases.

```python
# Bad: unsafe
a, b = some_str.split(".")

# Bad: safe but verbose
if some_str.count(".") == 1:
    a, b = some_str.split(".")
else:
    raise ValueError(f"Invalid format: {some_str!r}")

# Bad: safe but verbose
splits = some_str.split(".")
if len(splits) == 2:
    a, b = splits
else:
    raise ValueError(f"Invalid format: {some_str!r}")

# Good
match some_str.split("."):
    case [a, b]:
        ...
    case _:
        raise ValueError(f"Invalid format: {some_str!r}")
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
