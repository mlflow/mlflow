from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.log_model_artifact_path import LogModelArtifactPath


def test_log_model_artifact_path(index_path: Path) -> None:
    code = """
import mlflow

# Bad - using deprecated artifact_path positionally
mlflow.sklearn.log_model(model, "model")

# Bad - using deprecated artifact_path as keyword
mlflow.tensorflow.log_model(model, artifact_path="tf_model")

# Good - using the new 'name' parameter
mlflow.sklearn.log_model(model, name="my_model")

# Good - spark flavor is exempted from this rule
mlflow.spark.log_model(spark_model, "spark_model")

# Bad - another flavor with artifact_path
mlflow.pytorch.log_model(model, artifact_path="pytorch_model")
"""
    config = Config(select={LogModelArtifactPath.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 3
    assert all(isinstance(v.rule, LogModelArtifactPath) for v in violations)
    assert violations[0].loc == Location(4, 0)
    assert violations[1].loc == Location(7, 0)
    assert violations[2].loc == Location(16, 0)
