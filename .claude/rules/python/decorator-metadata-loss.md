---
paths: "**/*.py"
---

# Decorator Metadata Loss

When writing decorators, always use `@functools.wraps` to preserve function metadata (like `__name__` and `__doc__`), and use `typing.ParamSpec` and `typing.TypeVar` to preserve the function's type information for accurate type checking and autocompletion in IDEs.

```python
# Bad
from typing import Any, Callable


def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ...  # Pre-execution logic (e.g., logging, validation, setup)
        res = f(*args, **kwargs)
        ...  # Post-execution logic (e.g., cleanup, result transformation)
        return res

    return wrapper


# Good
import functools
from typing import Callable, ParamSpec, TypeVar

_P = ParamSpec("P")
_R = TypeVar("R")


def decorator(f: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        ...  # Pre-execution logic (e.g., logging, validation, setup)
        res = f(*args, **kwargs)
        ...  # Post-execution logic (e.g., cleanup, result transformation)
        return res

    return wrapper
```
