import subprocess
import sys

import pytest

from mlflow.recipes import Recipe

WARNING_MESSAGE = "MLflow Recipes is deprecated"


def test_no_warns() -> None:
    stdout = subprocess.check_output(
        [sys.executable, "-c", "import mlflow"], stderr=subprocess.STDOUT, text=True
    )
    assert WARNING_MESSAGE not in stdout

    prc = subprocess.run(
        [sys.executable, "-m", "mlflow", "models", "serve", "-m", "foo"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # As there is no model 'foo', the command will fail.
        check=False,
    )
    assert WARNING_MESSAGE not in prc.stdout


def test_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir("examples/recipes/regression")

    with pytest.warns(FutureWarning, match=WARNING_MESSAGE):
        Recipe(profile="local")

    stdout = subprocess.check_output(
        [sys.executable, "-m", "mlflow", "recipes", "--help"], stderr=subprocess.STDOUT, text=True
    )
    assert WARNING_MESSAGE in stdout

    stdout = subprocess.check_output(
        [sys.executable, "-m", "mlflow", "recipes", "clean", "--profile", "local"],
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert WARNING_MESSAGE in stdout
