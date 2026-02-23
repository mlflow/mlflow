---
paths: "**/*.py"
---

# Custom Assert Messages

Pytest's assertion introspection provides detailed failure information automatically. Avoid adding custom messages to `assert` statements in tests unless absolutely necessary.

```python
# Bad
def test_list_items():
    items = list_items()
    assert len(items) == 3, f"Expected 3 items, got {len(items)}"


# Good
def test_list_items():
    items = list_items()
    assert len(items) == 3
```
