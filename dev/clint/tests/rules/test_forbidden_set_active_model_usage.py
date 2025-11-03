from pathlib import Path

from clint.config import Config
from clint.linter import lint_file
from clint.rules.forbidden_set_active_model_usage import ForbiddenSetActiveModelUsage


def test_forbidden_set_active_model_usage(index_path: Path) -> None:
    code = """
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
    config = Config(select={ForbiddenSetActiveModelUsage.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 3
    assert all(isinstance(v.rule, ForbiddenSetActiveModelUsage) for v in violations)
    assert (violations[0].loc.lineno, violations[0].loc.col_offset) == (
        4,
        0,
    )  # mlflow.set_active_model call
    assert (violations[1].loc.lineno, violations[1].loc.col_offset) == (
        10,
        0,
    )  # from mlflow import set_active_model
    assert (violations[2].loc.lineno, violations[2].loc.col_offset) == (
        11,
        0,
    )  # direct set_active_model call
