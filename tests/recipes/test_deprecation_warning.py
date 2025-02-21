import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("code", "warns"),
    [
        ("import mlflow.recipes", True),
        ("from mlflow.recipes import Recipe", True),
        ("import mlflow", False),
    ],
)
def test_deprecation_warning(code: str, warns: bool) -> None:
    stdout = subprocess.check_output(
        [sys.executable, "-c", code], stderr=subprocess.STDOUT, text=True
    )
    assert ("FutureWarning: MLflow Recipes is deprecated" in stdout) is warns
