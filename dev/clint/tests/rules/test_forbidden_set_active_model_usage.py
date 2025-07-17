from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules.forbidden_set_active_model_usage import ForbiddenSetActiveModelUsage


def test_forbidden_set_active_model_usage(index: SymbolIndex, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test.py"
    tmp_file.write_text(
        """
import mlflow

# Bad
mlflow.set_active_model("model_name")

# Good
mlflow._set_active_model("model_name")

# Bad - with aliasing
from mlflow import set_active_model
set_active_model("model_name")

# Good - with aliasing
from mlflow import _set_active_model
_set_active_model("model_name")
"""
    )

    config = Config(select={ForbiddenSetActiveModelUsage.name})
    violations = lint_file(tmp_file, config, index)
    assert len(violations) == 3
    assert all(isinstance(v.rule, ForbiddenSetActiveModelUsage) for v in violations)
    assert violations[0].loc == Location(4, 0)  # mlflow.set_active_model call
    assert violations[1].loc == Location(10, 0)  # from mlflow import set_active_model
    assert violations[2].loc == Location(11, 0)  # direct set_active_model call
