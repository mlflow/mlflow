from pathlib import Path

import pytest
from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules.unknown_mlflow_function import UnknownMlflowFunction


def test_unknown_mlflow_function(index: SymbolIndex, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        '''
def bad():
    """
    .. code-block:: python

        import mlflow

        mlflow.foo()

    """


def good():

    """
    .. code-block:: python

        import mlflow

        mlflow.log_param("k", "v")

    """
'''
    )
    config = Config(select={UnknownMlflowFunction.name}, example_rules=[UnknownMlflowFunction.name])
    violations = lint_file(tmp_file, config, index)
    assert len(violations) == 1
    assert all(isinstance(v.rule, UnknownMlflowFunction) for v in violations)
    assert violations[0].loc == Location(7, 8)


@pytest.mark.parametrize("suffix", [".md", ".mdx"])
def test_unknown_mlflow_function_markdown(index: SymbolIndex, tmp_path: Path, suffix: str) -> None:
    tmp_file = (tmp_path / "test").with_suffix(suffix)
    tmp_file.write_text(
        """
# Bad

```python
import mlflow

mlflow.foo()
```

# Good

```python
import mlflow

mlflow.log_param("k", "v")
```

"""
    )
    config = Config(
        select={UnknownMlflowFunction.name},
        example_rules=[UnknownMlflowFunction.name],
    )
    violations = lint_file(tmp_file, config, index)
    assert len(violations) == 1
    assert all(isinstance(v.rule, UnknownMlflowFunction) for v in violations)
    assert violations[0].loc == Location(6, 0)
