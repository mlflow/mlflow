---
paths: "**/*.py"
---

# Untyped String Literals

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
