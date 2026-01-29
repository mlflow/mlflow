---
paths: "**/*.py"
---

# Unsafe String Unpacking

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
