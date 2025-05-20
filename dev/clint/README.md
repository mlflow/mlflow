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

## Want to ignore a rule for a specific file or line?

Option 1: Disable a rule on a specific line (recommended).

```python
foo()  # clint: disable=<rule_name>
```

Option 2: Update the `exclude` list in the `pyproject.toml` file.

```toml
[tool.clint]
exclude = [
  ...,
  "path/to/file",
]
```
