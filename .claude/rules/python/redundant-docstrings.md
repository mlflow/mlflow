---
paths: "**/*.py"
---

# Redundant Docstrings

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
