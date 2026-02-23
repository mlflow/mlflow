---
paths: "**/*.py"
---

# Repeated Test Assertions

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
