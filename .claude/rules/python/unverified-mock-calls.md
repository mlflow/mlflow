---
paths: "**/*.py"
---

# Unverified Mock Calls

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
