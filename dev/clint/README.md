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

## How to Add a New Rule

1. **Create rule** in `dev/clint/src/clint/rules/your_rule.py`:

```python
from clint.rules.base import Rule


class YourRule(Rule):
    def _message(self) -> str:
        return "Violation message"

    @staticmethod
    def check(file_content: str) -> bool:
        return "bad_pattern" in file_content
```

2. **Call rule** in `dev/clint/src/clint/linter.py`:

```python
if rules.YourRule.check(src):
    self._check(Range(Position(0, 0)), rules.YourRule())
```

3. **Import rule** in `dev/clint/src/clint/rules/__init__.py` and add to `__all__`.

4. **Add test** in `dev/clint/tests/rules/test_your_rule.py` (see existing tests for examples).

5. **Run**: `pytest dev/clint/tests/rules/test_your_rule.py`

## Testing

```bash
pytest dev/clint
```
