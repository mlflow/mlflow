# Clint

A custom linter for mlflow to enforce rules that ruff doesn't cover.

## Installation

```
pip install -e dev/clint
```

## Usage

```bash
clint file.py ...
```

## Integrating with Visual Studio Code

1. Install [the Pylint extension](https://marketplace.visualstudio.com/items?itemName=ms-python.pylint)
2. Add the following setting in your `settings.json` file:

```json
{
  "pylint.path": ["${interpreter}", "-m", "clint"]
}
```

## Ignoring Rules for Specific Files or Lines

**To ignore a rule on a specific line (recommended):**

```python
foo()  # clint: disable=<rule_name>
```

Replace `<rule_name>` with the actual rule you want to disable.

**To ignore a rule for an entire file:**

Add the file path to the `exclude` list in your `pyproject.toml`:

```toml
[tool.clint]
exclude = [
  # ...existing entries...
  "path/to/file.py",
]
```

## Testing

```bash
pytest dev/clint
```

## Rules

### redundant-test-docstring

Test functions and classes should avoid single-line docstrings that are typically redundant.
Multi-line docstrings that provide substantial documentation are allowed.

**Rationale:** Single-line docstrings in tests like "Test X" or "Tests for Y" are typically low-value and redundant with the test name. However, multi-line docstrings that explain complex test scenarios, backwards compatibility requirements, or critical behaviors are valuable and should be preserved.

**Good:**

```python
def test_user_authentication_fails_with_invalid_password():
    # Test implementation
    pass


def test_backwards_compatibility_with_v1_api():
    """
    This test ensures that the v2 API maintains backwards compatibility
    with v1 client libraries by accepting both old and new parameter formats.

    The test specifically validates edge cases documented in RFC-123.
    """
    # Test implementation
    pass
```

**Bad:**

```python
def test_auth():
    """Test that user authentication fails with invalid password."""
    pass


def test_database_connection():
    """Tests database connection."""
    pass
```

**Using NB Comments for Critical Documentation:**

When you need to document something unusual or critical in a test, use NB (nota bene) comments:

```python
def test_complex_edge_case():
    # NB: This test uses a workaround for issue #12345 because
    # the standard approach fails due to timing constraints.
    # We mock the database connection instead of using a real one.
    with mock.patch("db.connect") as mock_conn:
        result = process_with_timeout()

    # NB: We expect 42 here, not 41, because of the special rounding
    # behavior documented in RFC-789. This is intentional and correct.
    assert result == 42
```

The NB comment convention makes it clear that the comment contains important information that future maintainers need to know, distinguishing it from regular explanatory comments.
