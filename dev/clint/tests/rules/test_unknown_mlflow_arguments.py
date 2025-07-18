from pathlib import Path

import pytest
from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules.unknown_mlflow_arguments import UnknownMlflowArguments


def test_unknown_mlflow_arguments(index: SymbolIndex, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        '''
def bad():
    """
    .. code-block:: python

        import mlflow

        mlflow.log_param(foo="bar")

    """

def good():
    """
    .. code-block:: python

        import mlflow

        mlflow.log_param(key="k", value="v")
    """
'''
    )
    config = Config(
        select={UnknownMlflowArguments.name},
        example_rules=[UnknownMlflowArguments.name],
    )
    violations = lint_file(tmp_file, config, index)
    assert len(violations) == 1
    assert all(isinstance(v.rule, UnknownMlflowArguments) for v in violations)
    assert violations[0].loc == Location(7, 8)


@pytest.mark.parametrize("suffix", [".md", ".mdx"])
def test_unknown_mlflow_arguments_markdown(index: SymbolIndex, tmp_path: Path, suffix: str) -> None:
    tmp_file = (tmp_path / "test").with_suffix(suffix)
    tmp_file.write_text(
        """
# Bad

```python
import mlflow

mlflow.log_param(foo="bar")
```

# Good

```python
import mlflow

mlflow.log_param(key="k", value="v")
```
"""
    )
    config = Config(
        select={UnknownMlflowArguments.name},
        example_rules=[UnknownMlflowArguments.name],
    )
    violations = lint_file(tmp_file, config, index)
    assert len(violations) == 1
    assert all(isinstance(v.rule, UnknownMlflowArguments) for v in violations)
    assert violations[0].loc == Location(6, 0)
