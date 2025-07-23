from pathlib import Path

import pytest
from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules.lazy_module import LazyModule


def test_lazy_module(index: SymbolIndex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a file that looks like mlflow/__init__.py for the rule to apply

    monkeypatch.chdir(tmp_path)
    (tmp_path / "mlflow").mkdir()
    tmp_file = tmp_path / "mlflow" / "__init__.py"
    tmp_file.write_text(
        """
from mlflow.utils.lazy_load import LazyLoader
from typing import TYPE_CHECKING

# Bad - LazyLoader module not imported in TYPE_CHECKING block
anthropic = LazyLoader("mlflow.anthropic", globals(), "mlflow.anthropic")

# Good - LazyLoader with corresponding TYPE_CHECKING import
sklearn = LazyLoader("mlflow.sklearn", globals(), "mlflow.sklearn")

if TYPE_CHECKING:
    from mlflow import sklearn  # Good - this one is imported
"""
    )
    config = Config(select={LazyModule.name})
    violations = lint_file(tmp_file.relative_to(tmp_path), config, index)
    assert len(violations) == 1
    assert all(isinstance(v.rule, LazyModule) for v in violations)
    assert violations[0].loc == Location(5, 12)  # anthropic LazyLoader
