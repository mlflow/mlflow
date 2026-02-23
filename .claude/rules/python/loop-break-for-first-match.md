---
paths: "**/*.py"
---

# Loop-Break for First Match

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
