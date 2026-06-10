"""Private implementation of ``@mlflow.test`` + ``mlflow.genai.assert_behavior``.

Public surface:
- ``mlflow.test`` -- a no-op marker so the pytest plugin (auto-registered via
  the ``pyproject.toml`` entry point) can bundle/parallelize the test and group
  its traces under a regression-test run.
- ``mlflow.genai.assert_behavior`` -- the imperative assertion called in the body.
"""

from mlflow._assertions.api import assert_behavior
from mlflow._assertions.decorator import test

__all__ = ["assert_behavior", "test"]
