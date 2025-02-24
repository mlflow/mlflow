import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("code", "should_warn"),
    [
        ("import mlflow.recipes", True),
        ("from mlflow.recipes import Recipe", True),
        ("import mlflow", False),
    ],
)
def test_deprecation_warning(code: str, should_warn: bool) -> None:
    stdout = subprocess.check_output(
        [sys.executable, "-c", code], stderr=subprocess.STDOUT, text=True
    )
    assert ("FutureWarning: MLflow Recipes is deprecated" in stdout) is should_warn
