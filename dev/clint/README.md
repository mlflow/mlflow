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

**For multi-line constructs (docstrings, etc.), place the disable comment on the closing line:**

```python
def func():
    """
    Docstring with [markdown link](url).
    """  # clint: disable=markdown-link
    pass
```

This works because the linter checks both the start and end lines of the violation range.

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
