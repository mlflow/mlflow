# `pytest-raises-without-match`

This custom pylint rule disallows calling `pytest.raises` without a `match` argument
to avoid capturing unintended exceptions and eliminate false-positive tests.

## Example

Suppose we want to test this function throws `Exception("bar")` when `condition2` is satisfied.

```python
def func():
    if condition1:
        raise Exception("foo")

    if condition2:
        raise Exception("bar")
```

### Bad

```python
def test_func():
    with pytest.raises(Exception):
        func()
```

- This test passes when `condition1` is unintentionally satisfied.
- Future code readers will struggle to identify which exception `pytest.raises` should match.

### Good

```python
def test_func():
    with pytest.raises(Exception, match="bar"):
        func()
```

- This test fails when `condition1` is unintentionally satisfied.
- Future code readers can quickly identify which exception `pytest.raises` should match by searching `bar`.

## How to disable this rule

```python
def test_func():
    with pytest.raises(Exception):  # pylint: disable=pytest-raises-without-match
        func()
```

## References

- https://docs.pytest.org/en/latest/how-to/assert.html#assertions-about-expected-exceptions
- https://docs.pytest.org/en/latest/reference/reference.html#pytest.raises
