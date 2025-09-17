from pathlib import Path

import pytest
from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules import GetArtifactUri


def test_get_artifact_uri_in_rst_example(index_path: Path) -> None:
    code = """
Documentation
=============

Here's an example:

.. code-block:: python

    import mlflow

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        model_uri = mlflow.get_artifact_uri("model")
        print(model_uri)
"""
    config = Config(select={GetArtifactUri.name}, example_rules=[GetArtifactUri.name])
    violations = lint_file(Path("test.rst"), code, config, index_path)
    assert len(violations) == 1
    assert violations[0].rule.name == GetArtifactUri.name
    assert violations[0].loc == Location(12, 20)


@pytest.mark.parametrize("suffix", [".md", ".mdx"])
def test_get_artifact_uri_in_markdown_example(index_path: Path, suffix: str) -> None:
    code = """
# Documentation

Here's an example:

```python
import mlflow

with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    model_uri = mlflow.get_artifact_uri("model")
    print(model_uri)
```
"""
    config = Config(select={GetArtifactUri.name}, example_rules=[GetArtifactUri.name])
    violations = lint_file(Path("test").with_suffix(suffix), code, config, index_path)
    assert len(violations) == 1
    assert violations[0].rule.name == GetArtifactUri.name
    assert violations[0].loc == Location(10, 16)


def test_get_artifact_uri_not_in_regular_python_files(index_path: Path) -> None:
    code = """
import mlflow

with mlflow.start_run():
    model_uri = mlflow.get_artifact_uri("model")
    print(model_uri)
"""
    config = Config(select={GetArtifactUri.name}, example_rules=[GetArtifactUri.name])
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 0


def test_get_artifact_uri_without_log_model_allowed(index_path: Path) -> None:
    """Test that mlflow.get_artifact_uri is allowed when no log_model is present."""
    code = """
Documentation
=============

Here's an example:

.. code-block:: python

    import mlflow

    # This should be allowed - no log_model in the example
    model_uri = mlflow.get_artifact_uri("some_model")
    loaded_model = mlflow.sklearn.load_model(model_uri)
"""
    config = Config(select={GetArtifactUri.name}, example_rules=[GetArtifactUri.name])
    violations = lint_file(Path("test.rst"), code, config, index_path)
    assert len(violations) == 0
