from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
from clint.rules.mlflow_class_name import MlflowClassName


def test_mlflow_class_name(index_path: Path) -> None:
    code = """
# Bad - using MLflow
class MLflowClient:
    pass

# Bad - using MLFlow
class MLFlowLogger:
    pass

# Bad - nested occurrence of MLflow
class CustomMLflowHandler:
    pass

# Bad - nested occurrence of MLFlow
class BaseMLFlowTracker:
    pass

# Good - using Mlflow
class MlflowModel:
    pass

# Good - no MLflow patterns
class DataHandler:
    pass
"""
    config = Config(select={MlflowClassName.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 4
    assert all(isinstance(v.rule, MlflowClassName) for v in violations)
    assert violations[0].range == Range(Position(2, 0))  # MLflowClient
    assert violations[1].range == Range(Position(6, 0))  # MLFlowLogger
    assert violations[2].range == Range(Position(10, 0))  # CustomMLflowHandler
    assert violations[3].range == Range(Position(14, 0))  # BaseMLFlowTracker
