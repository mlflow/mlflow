---
paths: "**/*.py"
---

# Mock Setup After Patch

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
