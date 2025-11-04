from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.lazy_module import LazyModule


def test_lazy_module(index_path: Path) -> None:
    # Create a file that looks like mlflow/__init__.py for the rule to apply
    code = """
from mlflow.utils.lazy_load import LazyLoader
from typing import TYPE_CHECKING

# Bad - LazyLoader module not imported in TYPE_CHECKING block
anthropic = LazyLoader("mlflow.anthropic", globals(), "mlflow.anthropic")

# Good - LazyLoader with corresponding TYPE_CHECKING import
sklearn = LazyLoader("mlflow.sklearn", globals(), "mlflow.sklearn")

if TYPE_CHECKING:
    from mlflow import sklearn  # Good - this one is imported
"""
    config = Config(select={LazyModule.name})
    violations = lint_file(Path("mlflow", "__init__.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, LazyModule) for v in violations)
    assert violations[0].loc == Location(5, 12)  # anthropic LazyLoader
